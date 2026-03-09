"""
Road Surface Classification — TS2Vec Embedding & Clustering
Uses pre-windowed CSV files (labeled + unlabeled) as input.

Steps
-----
1. Load windowed CSVs  →  labeled train/test + unlabeled
2. Train TS2Vec on train split (labeled + unlabeled combined)
3. Generate embeddings for labeled train, test, and unlabeled sets
4. PCA reduce (fit on labeled train embeddings)
5. Best-K sweep (silhouette on labeled train PCA)
6. Cluster with 8 methods on labeled train embeddings
7. Evaluate on labeled test split (Silhouette, DB, CH, ARI, NMI, Dunn)
8. Also cluster unlabeled embeddings
9. Save predictions CSVs
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import normalize
from ts2vec import TS2Vec

# ── Timestamp & paths ─────────────────────────────────────────────────────────
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
SCRIPT_DIR = Path(__file__).parent
LOGS_DIR   = SCRIPT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE   = LOGS_DIR / f"{TIMESTAMP}_ts2vec.log"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Surface names ─────────────────────────────────────────────────────────────
SURFACE_NAMES = {
     1: "Paving Blocks (Smooth) (Red)",
     2: "Concrete Sidewalk",
     3: "Smooth Brick (High Street)",
     4: "Rough brick (High Street)",
     5: "Asphalt / Tar surface",
     6: "Indoor Carpet (low-pile)",
     7: "Indoor Linoleum",
     8: "Indoor Tile",
     9: "Curb Up",
    10: "Curb Down",
    11: "Rectangular Paving Tiles",
    12: "Paving Blocks (Rough)",
}

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "windowed_csv_labeled"  : Path("../../Datasets/ExtractedFeatures/labeled_accelerometer_raw_windows.csv"),
    "windowed_csv_unlabeled": Path("../../Datasets/ExtractedFeatures/unlabeled_accelerometer_raw_windows.csv"),
    "unlabeled_id"          : -1,
    "test_size"             : 0.2,
    "seed"                  : 42,
    "acc_epochs"            : 100,
    "output_dims"           : 128,
    "pca_variance"          : 0.95,
    "output_dir"            : SCRIPT_DIR,
    "models_dir"            : SCRIPT_DIR / "models",
}

# ── Method config: name → dict-key ────────────────────────────────────────────
# All methods use cosine distance on L2-normalised embeddings (no PCA).
_METHOD_CFG = {
    "KMeans"    : "kmeans",
    "Agg"       : "agg",
    "GMM"       : "gmm",
    "SBScan"    : "sbscan",
    "POS"       : "pos",
    "RandAssign": "rand_assign",
    "GSA"       : "gsa",
    "RandClust" : "rand_clust",
}


def _cosine_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return (N, K) cosine distance matrix; input need not be normalised."""
    X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    C_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    return 1.0 - X_n @ C_n.T


# ── Label remapping ───────────────────────────────────────────────────────────
def remap_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary positive surface IDs to consecutive 0-based integers."""
    unique  = sorted(np.unique(labels))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.vectorize(mapping.__getitem__)(labels)


# ── Stratified split ──────────────────────────────────────────────────────────
def stratified_split(windows, labels, test_size, seed):
    rng = np.random.default_rng(seed)
    tr_i, te_i = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        idx = rng.permutation(idx)
        n   = max(1, int(len(idx) * test_size))
        te_i.extend(idx[:n])
        tr_i.extend(idx[n:])
        name = SURFACE_NAMES.get(cls, f"Surface {cls}")
        logger.info("    %-32s: %5d train  %4d test", name, len(idx) - n, n)
    return windows[np.array(tr_i)], labels[np.array(tr_i)], \
           windows[np.array(te_i)], labels[np.array(te_i)]


# ── Load windowed CSV ─────────────────────────────────────────────────────────
def _load_windows(csv_path, has_labels):
    """Return (arr, labels) where arr is (N, 3, T), z-normalised per channel."""
    logger.info("  Loading: %s", csv_path)
    raw = pd.read_csv(csv_path)
    logger.info("  Rows: %d  Windows: %d", len(raw), raw["window_id"].nunique())
    windows, labels = [], []
    for _, group in raw.groupby("window_id", sort=True):
        xyz = group[["valueX", "valueY", "valueZ"]].to_numpy(dtype=np.float32).T  # (3, T)
        windows.append(xyz)
        labels.append(int(group["surface_id"].iloc[0]) if has_labels else -1)
    arr = np.stack(windows).astype(np.float32)
    mu  = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1,  keepdims=True).clip(1e-8)
    return (arr - mu) / std, np.array(labels, dtype=int)


def load_windowed_data(cfg):
    """
    Returns:
        tr_X  (N_tr, 3, T)  — labeled-train + unlabeled
        tr_y  (N_tr,)       — labels; -1 for unlabeled
        te_X  (N_te, 3, T)  — labeled-test
        te_y  (N_te,)
        X_u   (N_u,  3, T)  — unlabeled only (reference)
    """
    arr_l, labels = _load_windows(cfg["windowed_csv_labeled"], has_labels=True)
    valid  = (labels != cfg["unlabeled_id"]) & (labels > 0)
    X_l, y_l = arr_l[valid], remap_labels(labels[valid])
    logger.info("  Labeled windows: %d  Classes: %s", len(X_l), sorted(np.unique(y_l)))

    X_u, _ = _load_windows(cfg["windowed_csv_unlabeled"], has_labels=False)
    logger.info("  Unlabeled windows: %d", len(X_u))

    logger.info("  Stratified split (labeled only):")
    tr_X_l, tr_y_l, te_X, te_y = stratified_split(X_l, y_l, cfg["test_size"], cfg["seed"])

    if len(X_u) > 0:
        tr_X = np.concatenate([tr_X_l, X_u], axis=0)
        tr_y = np.concatenate([tr_y_l, np.full(len(X_u), -1, dtype=int)], axis=0)
        logger.info("  Training set: %d labeled + %d unlabeled = %d total",
                    len(tr_X_l), len(X_u), len(tr_X))
    else:
        tr_X, tr_y = tr_X_l, tr_y_l
        logger.info("  Training set: %d labeled (no unlabeled data)", len(tr_X))

    return tr_X, tr_y, te_X, te_y, X_u


# ── Stdout → logger bridge ────────────────────────────────────────────────────
class _LoggerWriter:
    def __init__(self, level):
        self._level = level
        self._buf   = ""

    def write(self, msg):
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._level(line)

    def flush(self):
        if self._buf.strip():
            self._level(self._buf.strip())
            self._buf = ""


# ── TS2Vec training ───────────────────────────────────────────────────────────
def train_ts2vec(windows_nct: np.ndarray, n_epochs: int, device: str):
    """Input (N, C, T); TS2Vec expects (N, T, C)."""
    windows_ntc = windows_nct.transpose(0, 2, 1)
    logger.info("Training TS2Vec — shape=%s  epochs=%d", windows_ntc.shape, n_epochs)
    model = TS2Vec(input_dims=windows_ntc.shape[2], device=device,
                   output_dims=CONFIG["output_dims"])
    old_stdout = sys.stdout
    sys.stdout = _LoggerWriter(logger.info)
    try:
        loss_log = model.fit(windows_ntc, n_epochs=n_epochs, verbose=True)
    finally:
        sys.stdout = old_stdout
    logger.info("Training done. Final loss: %.6f", loss_log[-1])
    return model, loss_log


# ── Embeddings ────────────────────────────────────────────────────────────────
def generate_embeddings(model: TS2Vec, windows_nct: np.ndarray, label: str) -> np.ndarray:
    windows_ntc = windows_nct.transpose(0, 2, 1)
    logger.info("  Generating embeddings for %s (shape=%s)...", label, windows_ntc.shape)
    emb = model.encode(windows_ntc, encoding_window="full_series")
    logger.info("  Embeddings shape for %s: %s", label, emb.shape)
    return emb.astype(np.float32)


# ── PCA reduction ─────────────────────────────────────────────────────────────
def pca_reduce(emb_tr: np.ndarray, emb_te: np.ndarray, variance: float):
    n_tr = normalize(emb_tr, norm="l2")
    n_te = normalize(emb_te, norm="l2")
    pca  = PCA(n_components=variance, svd_solver="full").fit(n_tr)
    logger.info("  PCA: %dd → %dd (explained_var=%.3f)",
                emb_tr.shape[1], pca.n_components_, pca.explained_variance_ratio_.sum())
    return n_tr, pca.transform(n_tr), n_te, pca.transform(n_te), pca


# ── Best K ────────────────────────────────────────────────────────────────────
def best_k(norm_emb: np.ndarray, k_min: int, k_max: int) -> int:
    scores = {}
    for k in range(k_min, k_max + 1):
        lbl       = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(norm_emb)
        scores[k] = silhouette_score(norm_emb, lbl, metric="cosine")
        logger.info("    K=%2d  sil=%.4f", k, scores[k])
    bk = max(scores, key=scores.get)
    logger.info("  Best K=%d  sil=%.4f", bk, scores[bk])
    return bk


# ── Additional clustering algorithms ─────────────────────────────────────────

class SBScanClustering:
    """DBSCAN with automatic epsilon selection via k-NN distance elbow."""
    def __init__(self, n_clusters_hint: int = 5, min_samples: int = 5):
        self.n_clusters_hint = n_clusters_hint
        self.min_samples     = min_samples
        self.labels_         = None
        self.eps_            = None
        self._knn_clf        = None

    def fit(self, X):
        k        = self.min_samples
        nbrs     = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
        dists, _ = nbrs.kneighbors(X)
        k_dists  = np.sort(dists[:, -1])
        n        = len(k_dists)
        xs       = np.linspace(0.0, 1.0, n)
        rng_d    = k_dists[-1] - k_dists[0]
        ys       = (k_dists - k_dists[0]) / (rng_d + 1e-10)
        knee_idx = int(np.argmax(np.abs(ys - xs)))
        self.eps_ = float(k_dists[knee_idx])
        if self.eps_ < 1e-6:
            self.eps_ = float(np.median(k_dists))
        logger.info("    SBScan auto-eps=%.4f  (knee idx=%d/%d)", self.eps_, knee_idx, n)
        db = DBSCAN(eps=self.eps_, min_samples=self.min_samples, metric="cosine").fit(X)
        self.labels_ = db.labels_
        mask    = self.labels_ != -1
        n_valid = int(mask.sum())
        n_cls   = len(np.unique(self.labels_[mask])) if n_valid > 0 else 0
        logger.info("    SBScan: %d clusters, %d non-noise / %d total", n_cls, n_valid, n)
        if n_valid > 1 and n_cls > 1:
            self._knn_clf = KNeighborsClassifier(
                n_neighbors=min(5, n_valid), metric="cosine"
            ).fit(X[mask], self.labels_[mask])
        return self

    def predict(self, X):
        if self._knn_clf is None:
            return np.zeros(len(X), dtype=int)
        return self._knn_clf.predict(X)


class PSOClustering:
    """Particle Swarm Optimisation clustering (minimise WCSS)."""
    def __init__(self, n_clusters: int = 5, n_particles: int = 15,
                 max_iter: int = 50, seed: int = 42):
        self.n_clusters  = n_clusters
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.seed        = seed
        self.centroids_  = None
        self.labels_     = None

    def _assign(self, X, centroids):
        return _cosine_distances(X, centroids).argmin(axis=1)

    def _wcss(self, X, centroids):
        asgn = self._assign(X, centroids)
        return sum(_cosine_distances(X[asgn == k], centroids[k:k+1]).sum()
                   for k in range(len(centroids)) if (asgn == k).any())

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        D, K, N = X.shape[1], self.n_clusters, self.n_particles
        particles  = X[rng.choice(len(X), size=N * K, replace=True)].reshape(N, K, D)
        velocities = rng.uniform(-0.1, 0.1, (N, K, D))
        pbest      = particles.copy()
        pbest_fit  = np.array([self._wcss(X, p) for p in particles])
        gbest      = pbest[pbest_fit.argmin()].copy()
        gbest_fit  = pbest_fit.min()
        w, c1, c2  = 0.5, 1.5, 1.5
        for _ in range(self.max_iter):
            r1 = rng.uniform(0, 1, (N, K, D))
            r2 = rng.uniform(0, 1, (N, K, D))
            velocities = (w * velocities
                          + c1 * r1 * (pbest - particles)
                          + c2 * r2 * (gbest[None] - particles))
            particles += velocities
            fits = np.array([self._wcss(X, p) for p in particles])
            imp  = fits < pbest_fit
            pbest[imp]     = particles[imp]
            pbest_fit[imp] = fits[imp]
            if fits.min() < gbest_fit:
                gbest     = particles[fits.argmin()].copy()
                gbest_fit = fits.min()
        self.centroids_ = gbest
        self.labels_    = self._assign(X, gbest)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


class RandomAssignClustering:
    """Randomly assigns cluster IDs without regard to data geometry."""
    def __init__(self, n_clusters: int = 5, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed       = seed
        self.labels_    = None

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.labels_ = rng.integers(0, self.n_clusters, size=len(X))
        return self

    def predict(self, X):
        rng = np.random.default_rng(self.seed + 1)
        return rng.integers(0, self.n_clusters, size=len(X))


class GravitationalSearchClustering:
    """Gravitational Search Algorithm (GSA) clustering."""
    def __init__(self, n_clusters: int = 5, n_agents: int = 15,
                 max_iter: int = 50, G0: float = 100.0, alpha: float = 20.0,
                 seed: int = 42):
        self.n_clusters = n_clusters
        self.n_agents   = n_agents
        self.max_iter   = max_iter
        self.G0         = G0
        self.alpha      = alpha
        self.seed       = seed
        self.centroids_ = None
        self.labels_    = None

    def _assign(self, X, centroids):
        return _cosine_distances(X, centroids).argmin(axis=1)

    def _fitness(self, X, centroids):
        asgn = self._assign(X, centroids)
        return sum(_cosine_distances(X[asgn == k], centroids[k:k+1]).sum()
                   for k in range(len(centroids)) if (asgn == k).any())

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        D, K, N = X.shape[1], self.n_clusters, self.n_agents
        agents     = X[rng.choice(len(X), size=N * K, replace=True)].reshape(N, K, D)
        velocities = np.zeros((N, K, D))
        best_agent, best_fit = None, np.inf
        for t in range(self.max_iter):
            fits  = np.array([self._fitness(X, a) for a in agents])
            G     = self.G0 * np.exp(-self.alpha * t / self.max_iter)
            f_rng = fits.max() - fits.min()
            masses = (np.ones(N) / N if f_rng < 1e-10
                      else (fits.max() - fits) / (f_rng + 1e-10))
            masses /= masses.sum()
            if fits.min() < best_fit:
                best_fit   = fits.min()
                best_agent = agents[fits.argmin()].copy()
            k_best  = max(2, int(N * (1 - t / self.max_iter)))
            top_idx = np.argsort(fits)[:k_best]
            acc = np.zeros((N, K, D))
            for i in range(N):
                for j in top_idx:
                    if i == j:
                        continue
                    r       = np.linalg.norm(agents[j] - agents[i]) + 1e-10
                    acc[i] += G * masses[j] * (agents[j] - agents[i]) / r
            velocities = rng.uniform(0, 1, (N, K, D)) * velocities + acc
            agents    += velocities
        self.centroids_ = best_agent
        self.labels_    = self._assign(X, best_agent)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


class RandomClustering:
    """Random centroid initialisation + nearest-neighbour assignment (no iteration)."""
    def __init__(self, n_clusters: int = 5, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed       = seed
        self.centroids_ = None
        self.labels_    = None

    def _assign(self, X, centroids):
        return _cosine_distances(X, centroids).argmin(axis=1)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        idxs = rng.choice(len(X), size=self.n_clusters, replace=False)
        self.centroids_ = X[idxs]
        self.labels_    = self._assign(X, self.centroids_)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


# ── Clustering ────────────────────────────────────────────────────────────────
def cluster_all(tr_norm, tr_pca, te_norm, te_pca, k, xu_norm=None, xu_pca=None):
    km     = KMeans(n_clusters=k, random_state=42, n_init=20).fit(tr_norm)
    agg    = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(tr_norm)
    gmm    = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(tr_norm)
    sbscan = SBScanClustering(n_clusters_hint=k, min_samples=5).fit(tr_norm)
    pos    = PSOClustering(n_clusters=k, seed=42).fit(tr_norm)
    rand_a = RandomAssignClustering(n_clusters=k, seed=42).fit(tr_norm)
    gsa    = GravitationalSearchClustering(n_clusters=k, seed=42).fit(tr_norm)
    rand_c = RandomClustering(n_clusters=k, seed=42).fit(tr_norm)

    # Agg prediction: nearest centroid in cosine space
    agg_centroids = np.vstack([tr_norm[agg.labels_ == c].mean(axis=0) for c in range(k)])
    agg_centroids = normalize(agg_centroids, norm="l2")

    def _agg_predict(X_norm):
        return _cosine_distances(X_norm, agg_centroids).argmin(axis=1)

    tr_pred = {
        "kmeans"     : km.labels_,
        "agg"        : agg.labels_,
        "gmm"        : gmm.predict(tr_norm),
        "sbscan"     : sbscan.labels_,
        "pos"        : pos.labels_,
        "rand_assign": rand_a.labels_,
        "gsa"        : gsa.labels_,
        "rand_clust" : rand_c.labels_,
    }
    te_pred = {
        "kmeans"     : km.predict(te_norm),
        "agg"        : _agg_predict(te_norm),
        "gmm"        : gmm.predict(te_norm),
        "sbscan"     : sbscan.predict(te_norm),
        "pos"        : pos.predict(te_norm),
        "rand_assign": RandomAssignClustering(n_clusters=k, seed=99).fit(te_norm).labels_,
        "gsa"        : gsa.predict(te_norm),
        "rand_clust" : RandomClustering(n_clusters=k, seed=99).fit(te_norm).labels_,
    }
    xu_pred = None
    if xu_norm is not None and len(xu_norm) > 0:
        xu_pred = {
            "kmeans"     : km.predict(xu_norm),
            "agg"        : _agg_predict(xu_norm),
            "gmm"        : gmm.predict(xu_norm),
            "sbscan"     : sbscan.predict(xu_norm),
            "pos"        : pos.predict(xu_norm),
            "rand_assign": RandomAssignClustering(n_clusters=k, seed=77).fit(xu_norm).labels_,
            "gsa"        : gsa.predict(xu_norm),
            "rand_clust" : RandomClustering(n_clusters=k, seed=77).fit(xu_norm).labels_,
        }
    return tr_pred, te_pred, xu_pred, km


# ── Evaluation ────────────────────────────────────────────────────────────────
def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels[labels != -1])
    if len(unique) < 2:
        return np.nan
    clusters  = [X[labels == lbl] for lbl in unique]
    centroids = np.array([c.mean(axis=0) for c in clusters])
    min_inter = np.inf
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = float(_cosine_distances(centroids[i:i+1], centroids[j:j+1])[0, 0])
            if d < min_inter:
                min_inter = d
    max_intra = max(
        (_cosine_distances(c, cent[None]).mean() * 2
         for c, cent in zip(clusters, centroids) if len(c) > 0),
        default=0.0,
    )
    return np.nan if max_intra < 1e-10 else float(min_inter / max_intra)


def evaluate(emb, pred, gt, metric="euclidean"):
    pred = np.asarray(pred)
    ok   = pred != -1
    ev, lv, gv = emb[ok], pred[ok], np.asarray(gt)[ok]
    if len(np.unique(lv)) < 2:
        return dict(Silhouette=np.nan, DB=np.nan, CH=np.nan,
                    ARI=np.nan, NMI=np.nan, Dunn=np.nan)
    return dict(
        Silhouette = silhouette_score(ev, lv, metric=metric),
        DB         = davies_bouldin_score(ev, lv),
        CH         = calinski_harabasz_score(ev, lv),
        ARI        = adjusted_rand_score(gv, lv),
        NMI        = normalized_mutual_info_score(gv, lv),
        Dunn       = dunn_index(ev, lv),
    )


def evaluate_unsupervised(emb, pred, pseudo_gt=None, metric="euclidean"):
    pred = np.asarray(pred)
    ok   = pred != -1
    ev, lv = emb[ok], pred[ok]
    if len(np.unique(lv)) < 2:
        return dict(Silhouette=np.nan, DB=np.nan, CH=np.nan, Dunn=np.nan,
                    ARI_vs_KM=np.nan, NMI_vs_KM=np.nan)
    result = dict(
        Silhouette = silhouette_score(ev, lv, metric=metric),
        DB         = davies_bouldin_score(ev, lv),
        CH         = calinski_harabasz_score(ev, lv),
        Dunn       = dunn_index(ev, lv),
        ARI_vs_KM  = np.nan,
        NMI_vs_KM  = np.nan,
    )
    if pseudo_gt is not None:
        gv = np.asarray(pseudo_gt)[ok]
        result["ARI_vs_KM"] = adjusted_rand_score(gv, lv)
        result["NMI_vs_KM"] = normalized_mutual_info_score(gv, lv)
    return result


def log_metrics(tr_norm, tr_pred, tr_lbl,
                te_norm, te_pred, te_lbl,
                xu_norm=None, xu_pred=None):
    for split, emb, pred, gt in [("TRAIN", tr_norm, tr_pred, tr_lbl),
                                  ("TEST",  te_norm, te_pred, te_lbl)]:
        rows = {name: evaluate(emb, pred[key], gt, metric="cosine")
                for name, key in _METHOD_CFG.items()}
        logger.info("\n── %s ──────────────────────────────────────\n\n%s",
                    split, pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))

    logger.info("\nGeneralisation gap  (train ARI − test ARI):")
    for name, key in _METHOD_CFG.items():
        tr_v = evaluate(tr_norm, tr_pred[key], tr_lbl, metric="cosine")["ARI"] or 0
        te_v = evaluate(te_norm, te_pred[key], te_lbl, metric="cosine")["ARI"] or 0
        gap  = (tr_v or 0) - (te_v or 0)
        flag = "good" if gap < 0.10 else "overfit" if gap > 0.20 else "acceptable"
        logger.info("  %-12s  train=%.4f  test=%.4f  gap=%+.4f  [%s]",
                    name, tr_v, te_v, gap, flag)

    if xu_pred is not None and xu_norm is not None:
        km_ref = xu_pred["kmeans"]
        rows = {name: evaluate_unsupervised(xu_norm, xu_pred[key],
                                            pseudo_gt=km_ref, metric="cosine")
                for name, key in _METHOD_CFG.items()}
        logger.info("\n── UNLABELED (unsupervised + ARI/NMI vs KMeans pseudo-GT) ──────────")
        logger.info("\n%s", pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("Log file: %s", LOG_FILE)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Device: %s", device)
    logger.info("Run timestamp: %s", TIMESTAMP)

    out_dir    = CONFIG["output_dir"]
    models_dir = CONFIG["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── [0] Surface names ─────────────────────────────────────────────────────
    logger.info("\n[0] Surface names")
    for sid, name in SURFACE_NAMES.items():
        logger.info("  %3d -> %s", sid, name)

    # ── [1] Load windowed data ────────────────────────────────────────────────
    logger.info("\n[1] Load windowed data (labeled + unlabeled)")
    tr_X, tr_y, te_X, te_y, X_u = load_windowed_data(CONFIG)

    labeled_mask = tr_y != -1
    tr_X_l = tr_X[labeled_mask]
    tr_y_l = tr_y[labeled_mask]

    # ── [2] Labeled train distribution ───────────────────────────────────────
    logger.info("\n[2] Labeled-train class distribution:")
    for cls in sorted(np.unique(tr_y_l)):
        logger.info("  Class %2d: %d windows", cls, int((tr_y_l == cls).sum()))

    # ── [3] Train TS2Vec (labeled + unlabeled) ────────────────────────────────
    logger.info("\n[3] Train TS2Vec on full training set (labeled + unlabeled)")
    model, loss_log = train_ts2vec(tr_X, CONFIG["acc_epochs"], device)

    model_path = models_dir / f"ts2vec_{TIMESTAMP}.pth"
    torch.save(model.net.state_dict(), model_path)
    logger.info("\n[3b] Model saved: %s", model_path)

    loss_path = out_dir / f"ts2vec_{TIMESTAMP}_loss_log.csv"
    pd.DataFrame({"epoch": range(1, len(loss_log) + 1), "loss": loss_log}).to_csv(
        loss_path, index=False)
    logger.info("Loss log saved: %s", loss_path)

    # ── [4] Embeddings + PCA ──────────────────────────────────────────────────
    logger.info("\n[4] Embeddings + PCA (labeled train/test only)")
    tr_emb_l = generate_embeddings(model, tr_X_l, "labeled-train")
    te_emb   = generate_embeddings(model, te_X,   "test")
    tr_norm, tr_pca, te_norm, te_pca, pca = pca_reduce(
        tr_emb_l, te_emb, CONFIG["pca_variance"])

    np.save(out_dir / f"ts2vec_{TIMESTAMP}_emb_train_labeled.npy", tr_emb_l)
    np.save(out_dir / f"ts2vec_{TIMESTAMP}_emb_test.npy",          te_emb)
    np.save(out_dir / f"ts2vec_{TIMESTAMP}_labels_train.npy",      tr_y_l)
    np.save(out_dir / f"ts2vec_{TIMESTAMP}_labels_test.npy",       te_y)

    # ── [4b] Embed unlabeled ──────────────────────────────────────────────────
    xu_norm = xu_pca = None
    if len(X_u) > 0:
        logger.info("\n[4b] Embed unlabeled windows (%d)", len(X_u))
        u_emb  = generate_embeddings(model, X_u, "unlabeled")
        xu_norm = normalize(u_emb, norm="l2")
        xu_pca  = pca.transform(xu_norm)
        logger.info("  Unlabeled embeddings: %s  PCA: %s", u_emb.shape, xu_pca.shape)
        np.save(out_dir / f"ts2vec_{TIMESTAMP}_emb_unlabeled.npy", u_emb)
    else:
        logger.info("\n[4b] No unlabeled windows to embed.")

    # ── [5] Best K ────────────────────────────────────────────────────────────
    n_classes = len(np.unique(tr_y_l))
    k_min, k_max = max(2, n_classes - 2), n_classes + 2
    logger.info("\n[5] Best-K sweep: K=%d..%d  (n_classes=%d)", k_min, k_max, n_classes)
    k = best_k(tr_norm, k_min, k_max)

    # ── [6] Clustering ────────────────────────────────────────────────────────
    logger.info("\n[6] Clustering with K=%d  (8 methods)", k)
    tr_pred, te_pred, xu_pred, km = cluster_all(
        tr_norm, tr_pca, te_norm, te_pca, k, xu_norm, xu_pca)

    logger.info("\n  K-Means cluster → surface (train):")
    for cid in sorted(set(tr_pred["kmeans"])):
        mask     = tr_pred["kmeans"] == cid
        majority = int(pd.Series(tr_y_l[mask]).mode()[0])
        logger.info("    cluster %2d → class %d  (%d windows)", cid, majority, mask.sum())

    logger.info("\n  SBScan cluster → surface (train):")
    for cid in sorted(set(tr_pred["sbscan"])):
        mask = tr_pred["sbscan"] == cid
        if cid == -1:
            logger.info("    cluster %2d → noise  (%d windows)", cid, mask.sum())
        else:
            majority = int(pd.Series(tr_y_l[mask]).mode()[0])
            logger.info("    cluster %2d → class %d  (%d windows)", cid, majority, mask.sum())

    # ── [7] Metrics ───────────────────────────────────────────────────────────
    logger.info("\n[7] Metrics")
    log_metrics(tr_norm, tr_pred, tr_y_l,
                te_norm, te_pred, te_y,
                xu_norm, xu_pred)

    # ── [8] Save predictions ──────────────────────────────────────────────────
    logger.info("\n[8] Saving predictions")

    te_out = pd.DataFrame({"surface_id_gt": te_y} |
                          {f"cluster_{m}": te_pred[m] for m in te_pred})
    te_csv = out_dir / f"ts2vec_{TIMESTAMP}_test_predictions.csv"
    te_out.to_csv(te_csv, index=False)
    logger.info("Test predictions saved: %s  (%d rows)", te_csv, len(te_out))

    tr_out = pd.DataFrame({"surface_id_gt": tr_y_l} |
                          {f"cluster_{m}": tr_pred[m] for m in tr_pred})
    tr_csv = out_dir / f"ts2vec_{TIMESTAMP}_train_predictions.csv"
    tr_out.to_csv(tr_csv, index=False)
    logger.info("Train predictions saved: %s  (%d rows)", tr_csv, len(tr_out))

    if xu_pred is not None:
        xu_out = pd.DataFrame({f"cluster_{m}": xu_pred[m] for m in xu_pred})
        xu_csv = out_dir / f"ts2vec_{TIMESTAMP}_unlabeled_predictions.csv"
        xu_out.to_csv(xu_csv, index=False)
        logger.info("Unlabeled predictions saved: %s  (%d rows)", xu_csv, len(xu_out))

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    main()

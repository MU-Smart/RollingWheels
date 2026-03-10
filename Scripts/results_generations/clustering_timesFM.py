"""
Road Surface Classification — TimesFM Embedding + Clustering Pipeline

Model      : google/timesfm-2.0-500m-pytorch (HuggingFace, no fine-tuning)
Embeddings : last_hidden_state mean-pooled over patches  → (N, 1280)

Steps
-----
1. Load windowed CSV  →  Z-normalise  →  stratified 80/20 split (labeled only)
2. Load TimesFM model from HuggingFace
3. Compute per-window magnitude  →  extract TimesFM embeddings  →  StandardScaler  →  PCA
4. Cluster (KMeans / Agglomerative / GMM / SBScan / PSO / GSA / Random)
5. Evaluate on labeled test split (Silhouette / DB / CH / ARI / NMI / Dunn)
6. Visualise: t-SNE + UMAP  (grid + individual plots)
7. Fixed-K experiment (K = 3, 5, 7, 11) — t-SNE + UMAP
8. Repeat clustering / evaluation / visualisation on unlabeled data (unsupervised)
"""

import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import TimesFmModel

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

# ── Logging setup ─────────────────────────────────────────────────────────────
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(log_dir: Path = Path(".")) -> logging.Logger:
    """Set up logger writing to both console and a timestamped log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{RUN_TS}_timesFM_cosine_results.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("vcn")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path.resolve()}")
    return logger

logger = logging.getLogger("vcn")   # module-level; populated after setup_logging()

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "surface_types_csv"      : Path("../../Datasets/surface_types.csv"),
    "output_csv"             : Path("timesFM_cosine_results_unlabeled_predictions.csv"),
    "unlabeled_id"           : 0,

    # merge map type: "A" (5 classes), "B" (7 classes), "C" (9 classes)
    "merge_map_type"         : "B",

    # split
    "test_size"              : 0.2,
    "seed"                   : 42,

    # PCA
    "pca_variance"           : 0.95,

    # raw windowed data
    "windowed_csv_labeled"   : Path("../../Datasets/ExtractedFeatures/labeled_accelerometer_raw_windows.csv"),
    "windowed_csv_unlabeled" : Path("../../Datasets/ExtractedFeatures/unlabeled_accelerometer_raw_windows.csv"),

    # TimesFM model
    "timesfm_model_id"       : "google/timesfm-2.0-500m-pytorch",
    "timesfm_batch_size"     : 32,

    # Output directories
    "figures_base"           : Path("figures"),
    "models_dir"             : Path("models"),
}

# ── Surface name lookup ───────────────────────────────────────────────────────
SURFACE_NAMES: dict = {}

def load_surface_names(path):
    global SURFACE_NAMES
    try:
        df       = pd.read_csv(path)
        id_col   = next(c for c in df.columns if "id"   in c.lower())
        name_col = next(c for c in df.columns if "name" in c.lower())
        SURFACE_NAMES = {int(r[id_col]): str(r[name_col]) for _, r in df.iterrows()}
        for sid, name in sorted(SURFACE_NAMES.items()):
            logger.info(f"    {sid:3d} -> {name}")
    except Exception as e:
        logger.warning(f"  WARNING: {e}")

def sname(sid):
    return SURFACE_NAMES.get(int(sid), f"Surface {sid}")

# ── Surface merging ───────────────────────────────────────────────────────────
MERGE_MAP_TYPE_A = {
    1:  0,   # Paving Blocks (Smooth) (Red)
    2:  1,   # Concrete Sidewalk
    3:  0,   # Smooth Brick (High Street)
    4:  2,   # Rough Brick (High Street)
    5:  2,   # Asphalt / Tar surface
    6:  2,   # Indoor Carpet (low-pile)
    7:  0,   # Indoor Linoleum
    8:  0,   # Indoor Tile
    9:  3,   # Curb Up
   10:  3,   # Curb Down
   11:  4,   # Rectangular Paving Tiles
   12:  4,   # Paving Blocks (Rough)
}

MERGE_MAP_TYPE_B = {
    1:  0,   # Paving Blocks (Smooth) (Red)
    2:  1,   # Concrete Sidewalk
    3:  2,   # Smooth Brick (High Street)
    4:  2,   # Rough Brick (High Street)
    5:  3,   # Asphalt / Tar surface
    6:  3,   # Indoor Carpet (low-pile)
    7:  4,   # Indoor Linoleum
    8:  0,   # Indoor Tile
    9:  6,   # Curb Up
   10:  6,   # Curb Down
   11:  7,   # Rectangular Paving Tiles
   12:  7,   # Paving Blocks (Rough)
}

MERGE_MAP_TYPE_C = {
    1:  0,   # Paving Blocks (Smooth) (Red)
    2:  1,   # Concrete Sidewalk
    3:  2,   # Smooth Brick (High Street)
    4:  3,   # Rough Brick (High Street)
    5:  4,   # Asphalt / Tar surface
    6:  4,   # Indoor Carpet (low-pile)
    7:  5,   # Indoor Linoleum
    8:  6,   # Indoor Tile
    9:  7,   # Curb Up
   10:  7,   # Curb Down
   11:  8,   # Rectangular Paving Tiles
   12:  8,   # Paving Blocks (Rough)
}

SUPER_NAMES_TYPE_A = {
    0: "Paving Blocks / Smooth Brick / Linoleum / Indoor Tile",
    1: "Concrete Sidewalk",
    2: "Rough Brick + Asphalt + Indoor Carpet",
    3: "Curb (Up + Down)",
    4: "Rect. Tiles / Paving Blocks (Rough)",
}

SUPER_NAMES_TYPE_B = {
    0: "Paving Blocks (Smooth) / Indoor Tile",
    1: "Concrete Sidewalk",
    2: "Brick (Smooth + Rough)",
    3: "Asphalt + Indoor Carpet",
    4: "Indoor Linoleum",
    # index 5 unused in this mapping
    6: "Curb (Up + Down)",
    7: "Rect. Tiles / Paving Blocks (Rough)",
}

SUPER_NAMES_TYPE_C = {
    0: "Paving Blocks (Smooth)",
    1: "Concrete Sidewalk",
    2: "Smooth Brick",
    3: "Rough Brick",
    4: "Asphalt + Indoor Carpet",
    5: "Indoor Linoleum",
    6: "Indoor Tile",
    7: "Curb (Up + Down)",
    8: "Rect. Tiles / Paving Blocks (Rough)",
}

_MERGE_MAPS   = {"A": MERGE_MAP_TYPE_A, "B": MERGE_MAP_TYPE_B, "C": MERGE_MAP_TYPE_C}
_SUPER_NAMES  = {"A": SUPER_NAMES_TYPE_A, "B": SUPER_NAMES_TYPE_B, "C": SUPER_NAMES_TYPE_C}

MERGE_MAP       = _MERGE_MAPS[CONFIG["merge_map_type"]]
SUPER_NAMES     = _SUPER_NAMES[CONFIG["merge_map_type"]]
N_SUPER_CLASSES = len(SUPER_NAMES)

def remap_labels(y):
    """Map original surface_id -> super-class id.  Unknown ids kept as-is."""
    return np.array([MERGE_MAP.get(int(sid), int(sid)) for sid in y])

def super_name(sid):
    return SUPER_NAMES.get(int(sid), sname(sid))

# ── Stratified split ──────────────────────────────────────────────────────────
def stratified_split(X, y, test_size, seed):
    rng  = np.random.default_rng(seed)
    tr_i, te_i = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        n   = max(1, int(len(idx) * test_size))
        te_i.extend(idx[:n]);  tr_i.extend(idx[n:])
        logger.info(f"    {sname(cls):30s}: {len(idx)-n:5d} train  {n:4d} test")
    return (X[tr_i], y[tr_i], X[te_i], y[te_i])

# ── TimesFM embedding extraction ──────────────────────────────────────────────
def compute_magnitude(X: np.ndarray) -> np.ndarray:
    """Convert (N, 3, T) 3-axis windows to (N, T) L2 magnitude."""
    return np.linalg.norm(X, axis=1).astype(np.float32)


def extract_timesfm_embeddings(model, X_mag: np.ndarray, device,
                                batch_size: int = 32) -> np.ndarray:
    """
    Extract embeddings from TimesFM.

    Args:
        X_mag : (N, T) float32 magnitude array
    Returns:
        (N, hidden_dim) float32 numpy array (mean-pooled over patches)
    """
    model.eval()
    all_emb = []
    for i in tqdm(range(0, len(X_mag), batch_size), desc="TimesFM embeddings"):
        batch = torch.tensor(X_mag[i:i + batch_size], dtype=torch.float32).to(device)
        padding = torch.zeros(batch.shape, dtype=torch.long).to(device)
        freq    = torch.zeros(batch.shape[0], dtype=torch.long).view(-1, 1).to(device)
        with torch.no_grad():
            out = model(past_values=batch,
                        past_values_padding=padding,
                        freq=freq)
        # last_hidden_state: (B, n_patches, hidden_dim) → mean over patches
        pooled = out.last_hidden_state.mean(dim=1)
        all_emb.append(pooled.cpu().numpy())
    return np.vstack(all_emb)


# ── Load windowed csv ──────────────────────────────────────────────────────────
def _load_windows(csv_path, has_labels):
    """Load a windowed CSV → (arr (N,3,T), labels).
    If has_labels=False, labels is an all-(-1) array."""
    logger.info(f"  Loading: {csv_path}")
    raw = pd.read_csv(csv_path)
    logger.info(f"  Rows: {len(raw)}  Windows: {raw['window_id'].nunique()}")
    windows, labels = [], []
    for _, group in raw.groupby("window_id", sort=True):
        xyz = group[["valueX", "valueY", "valueZ"]].to_numpy(dtype=np.float32).T  # (3, T)
        windows.append(xyz)
        labels.append(int(group["surface_id"].iloc[0]) if has_labels else -1)
    arr = np.stack(windows).astype(np.float32)
    mu  = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1,  keepdims=True).clip(1e-8)
    arr = (arr - mu) / std
    return arr, np.array(labels, dtype=int)


def load_windowed_data(cfg):
    """
    Load labeled + unlabeled CSVs.
    Returns tr_X, tr_y, te_X, te_y, X_u
      - tr_X / tr_y : labeled training windows + super-class labels
      - te_X / te_y : labeled test windows + super-class labels
      - X_u         : unlabeled windows
    """
    arr_l, labels = _load_windows(cfg["windowed_csv_labeled"], has_labels=True)
    valid  = (labels != cfg["unlabeled_id"]) & (labels > 0)
    X_l, y_l = arr_l[valid], remap_labels(labels[valid])
    logger.info(f"  Labeled windows: {len(X_l)}  Classes: {sorted(np.unique(y_l))}")

    X_u, _ = _load_windows(cfg["windowed_csv_unlabeled"], has_labels=False)
    logger.info(f"  Unlabeled windows: {len(X_u)}")

    logger.info("  Stratified split (labeled only):")
    tr_X, tr_y, te_X, te_y = stratified_split(X_l, y_l, cfg["test_size"], cfg["seed"])
    return tr_X, tr_y, te_X, te_y, X_u


# ── PCA ───────────────────────────────────────────────────────────────────────
def pca_reduce(emb_tr, emb_te, variance=0.95):
    n_tr = normalize(emb_tr, norm="l2")
    n_te = normalize(emb_te, norm="l2")
    pca  = PCA(n_components=variance, svd_solver="full").fit(n_tr)
    logger.info(f"  PCA: {emb_tr.shape[1]}d → {pca.n_components_}d "
                f"(var={pca.explained_variance_ratio_.sum():.3f})")
    return n_tr, pca.transform(n_tr), n_te, pca.transform(n_te), pca


# ── Additional clustering algorithms ─────────────────────────────────────────
class SBScanClustering:
    """DBSCAN with automatic epsilon selection via k-NN cosine distance elbow."""
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
        xs = np.linspace(0.0, 1.0, n)
        rng_d = k_dists[-1] - k_dists[0]
        ys = (k_dists - k_dists[0]) / (rng_d + 1e-10)
        knee_idx  = int(np.argmax(np.abs(ys - xs)))
        self.eps_ = float(k_dists[knee_idx])
        if self.eps_ < 1e-6:
            self.eps_ = float(np.median(k_dists))
        logger.info(f"    SBScan auto-eps={self.eps_:.4f}  (knee idx={knee_idx}/{n})")
        db = DBSCAN(eps=self.eps_, min_samples=self.min_samples, metric="cosine").fit(X)
        self.labels_ = db.labels_
        mask    = self.labels_ != -1
        n_valid = int(mask.sum())
        n_cls   = len(np.unique(self.labels_[mask])) if n_valid > 0 else 0
        logger.info(f"    SBScan: {n_cls} clusters, {n_valid} non-noise / {n} total")
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
    """Particle Swarm Optimisation clustering (minimise within-cluster cosine distance)."""
    def __init__(self, n_clusters: int = 5, n_particles: int = 15,
                 max_iter: int = 50, seed: int = 42):
        self.n_clusters  = n_clusters
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.seed        = seed
        self.centroids_  = None
        self.labels_     = None

    def _assign(self, X, centroids):
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        C_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        return (1.0 - X_n @ C_n.T).argmin(axis=1)

    def _wcss(self, X, centroids):
        asgn = self._assign(X, centroids)
        total = 0.0
        for k in range(len(centroids)):
            mask = asgn == k
            if not mask.any():
                continue
            X_n = X[mask] / (np.linalg.norm(X[mask], axis=1, keepdims=True) + 1e-10)
            c_n = centroids[k] / (np.linalg.norm(centroids[k]) + 1e-10)
            total += (1.0 - X_n @ c_n).sum()
        return total

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
    """Gravitational Search Algorithm (GSA) clustering (cosine distance)."""
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
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        C_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        return (1.0 - X_n @ C_n.T).argmin(axis=1)

    def _fitness(self, X, centroids):
        asgn = self._assign(X, centroids)
        total = 0.0
        for k in range(len(centroids)):
            mask = asgn == k
            if not mask.any():
                continue
            X_n = X[mask] / (np.linalg.norm(X[mask], axis=1, keepdims=True) + 1e-10)
            c_n = centroids[k] / (np.linalg.norm(centroids[k]) + 1e-10)
            total += (1.0 - X_n @ c_n).sum()
        return total

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
    """Random centroid initialisation + cosine nearest-neighbour assignment (no iteration)."""
    def __init__(self, n_clusters: int = 5, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed       = seed
        self.centroids_ = None
        self.labels_    = None

    def _assign(self, X, centroids):
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        C_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        return (1.0 - X_n @ C_n.T).argmin(axis=1)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        idxs = rng.choice(len(X), size=self.n_clusters, replace=False)
        self.centroids_ = X[idxs]
        self.labels_    = self._assign(X, self.centroids_)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


# ── K selection ───────────────────────────────────────────────────────────────
def best_k(pca_emb, k_min, k_max, figures_dir: Path):
    cos_emb = normalize(pca_emb, norm="l2")
    scores = {}
    for k in range(k_min, k_max + 1):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(cos_emb)
        scores[k] = silhouette_score(cos_emb, lbl, metric="cosine")
        logger.info(f"    K={k:2d}  sil={scores[k]:.4f}")
    bk = max(scores, key=scores.get)
    logger.info(f"  Best K={bk}  sil={scores[bk]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(list(scores.keys()), list(scores.values()), "o-", color="#457b9d")
    ax.axvline(bk, color="#e63946", ls="--", label=f"K={bk}")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette")
    ax.set_title("K selection"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = figures_dir / f"17_k_selection_{RUN_TS}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    logger.info(f"  Saved {fname}")
    return bk


# ── Fixed-K experiment ────────────────────────────────────────────────────────
def _fixed_k_grid(pca_emb, proj_coords, proj_name, fname, ks):
    """2×2 grid of KMeans at fixed K values, coloured by cluster only."""
    assert len(ks) == 4, "Need exactly 4 K values for 2×2 grid"
    cos_emb = normalize(pca_emb, norm="l2")
    rows = {}
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Fixed-K experiment — {proj_name} (test split)",
                 fontsize=13, fontweight="bold")
    for ax, k in zip(axes.flat, ks):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(cos_emb)
        rows[f"K={k}"] = dict(
            Silhouette = silhouette_score(cos_emb, lbl, metric="cosine"),
            DB         = davies_bouldin_score(cos_emb, lbl),
        )
        for i, cid in enumerate(sorted(set(lbl))):
            m = lbl == cid
            ax.scatter(proj_coords[m, 0], proj_coords[m, 1],
                       c=COLORS[i % len(COLORS)], alpha=0.7, s=8,
                       label=f"Cluster {cid}")
        sil = rows[f"K={k}"]["Silhouette"]
        ax.set_title(f"K={k}  (sil={sil:.3f})", fontweight="bold", fontsize=10)
        ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
        ax.legend(fontsize=7, markerscale=1.5, loc="best", framealpha=0.6)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")
    return rows


def _print_quality(rows):
    logger.info(f"  {'K':<8} {'Silhouette':>12} {'Davies-Bouldin':>16}  Quality")
    logger.info("  " + "-" * 46)
    for k, r in rows.items():
        sil, db = r["Silhouette"], r["DB"]
        q = "Excellent" if sil > 0.6 else "Good" if sil > 0.4 else "Fair" if sil > 0.2 else "Poor"
        logger.info(f"  {k:<8} {sil:>12.4f} {db:>16.4f}  {q}")


def experiment_fixed_k(pca_emb, ts, um, figures_dir: Path, ks=(3, 5, 7, 11)):
    logger.info(f"\n  Fixed-K experiment  K={list(ks)}")
    fname_ts = figures_dir / f"20_fixed_k_tsne_{RUN_TS}.png"
    rows_ts = _fixed_k_grid(pca_emb, ts, "t-SNE", fname_ts, ks)
    logger.info("\n  Cluster quality — t-SNE projection:")
    _print_quality(rows_ts)

    fname_um = figures_dir / f"20_fixed_k_umap_{RUN_TS}.png"
    rows_um = _fixed_k_grid(pca_emb, um, "UMAP", fname_um, ks)
    logger.info("\n  Cluster quality — UMAP projection:")
    _print_quality(rows_um)

    return pd.DataFrame(rows_ts).T


# ── Clustering ────────────────────────────────────────────────────────────────
def cluster_all(tr_norm, tr_pca, te_norm, te_pca, k,
                xu_norm=None, xu_pca=None):
    # L2-normalise PCA embeddings so cosine distance ≡ Euclidean distance on unit sphere
    tr_cos = normalize(tr_pca, norm="l2")
    te_cos = normalize(te_pca, norm="l2")
    xu_cos = normalize(xu_pca, norm="l2") if xu_pca is not None and len(xu_pca) > 0 else xu_pca

    km      = KMeans(n_clusters=k, random_state=42, n_init=20).fit(tr_cos)
    agg     = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(tr_norm)
    gmm     = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(tr_cos)
    sbscan  = SBScanClustering(n_clusters_hint=k, min_samples=5).fit(tr_cos)
    pos     = PSOClustering(n_clusters=k, seed=42).fit(tr_cos)
    rand_a  = RandomAssignClustering(n_clusters=k, seed=42).fit(tr_cos)
    gsa     = GravitationalSearchClustering(n_clusters=k, seed=42).fit(tr_cos)
    rand_c  = RandomClustering(n_clusters=k, seed=42).fit(tr_cos)

    agg_centroids = np.vstack([tr_norm[agg.labels_ == c].mean(axis=0)
                               for c in range(k)])
    agg_centroids = normalize(agg_centroids, norm="l2")

    def _agg_predict(X_norm):
        X_n = X_norm / (np.linalg.norm(X_norm, axis=1, keepdims=True) + 1e-10)
        return (1.0 - X_n @ agg_centroids.T).argmin(axis=1)

    tr_pred = {
        "kmeans"     : km.labels_,
        "agg"        : agg.labels_,
        "gmm"        : gmm.predict(tr_cos),
        "sbscan"     : sbscan.labels_,
        "pos"        : pos.labels_,
        "rand_assign": rand_a.labels_,
        "gsa"        : gsa.labels_,
        "rand_clust" : rand_c.labels_,
    }
    te_pred = {
        "kmeans"     : km.predict(te_cos),
        "agg"        : _agg_predict(te_norm),
        "gmm"        : gmm.predict(te_cos),
        "sbscan"     : sbscan.predict(te_cos),
        "pos"        : pos.predict(te_cos),
        "rand_assign": RandomAssignClustering(n_clusters=k, seed=99).fit(te_cos).labels_,
        "gsa"        : gsa.predict(te_cos),
        "rand_clust" : RandomClustering(n_clusters=k, seed=99).fit(te_cos).labels_,
    }
    xu_pred = None
    if xu_norm is not None and xu_cos is not None and len(xu_cos) > 0:
        xu_pred = {
            "kmeans"     : km.predict(xu_cos),
            "agg"        : _agg_predict(xu_norm),
            "gmm"        : gmm.predict(xu_cos),
            "sbscan"     : sbscan.predict(xu_cos),
            "pos"        : pos.predict(xu_cos),
            "rand_assign": RandomAssignClustering(n_clusters=k, seed=77).fit(xu_cos).labels_,
            "gsa"        : gsa.predict(xu_cos),
            "rand_clust" : RandomClustering(n_clusters=k, seed=77).fit(xu_cos).labels_,
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
            d = np.linalg.norm(centroids[i] - centroids[j])
            if d < min_inter:
                min_inter = d
    max_intra = max(
        (np.linalg.norm(c - cent, axis=1).mean() * 2
         for c, cent in zip(clusters, centroids) if len(c) > 0),
        default=0.0,
    )
    return np.nan if max_intra < 1e-10 else float(min_inter / max_intra)


def c2s_map(pred, gt):
    return {c: (-1 if c == -1 else int(pd.Series(gt[np.asarray(pred)==c]).mode()[0]))
            for c in set(pred)}


def evaluate(emb, pred, gt, metric="euclidean"):
    pred = np.asarray(pred); ok = pred != -1
    ev, lv, gv = emb[ok], pred[ok], np.asarray(gt)[ok]
    if len(np.unique(lv)) < 2:
        return dict(Silhouette=np.nan, DB=np.nan, CH=np.nan, ARI=np.nan, NMI=np.nan, Dunn=np.nan)
    return dict(
        Silhouette = silhouette_score(ev, lv, metric=metric),
        DB         = davies_bouldin_score(ev, lv),
        CH         = calinski_harabasz_score(ev, lv),
        ARI        = adjusted_rand_score(gv, lv),
        NMI        = normalized_mutual_info_score(gv, lv),
        Dunn       = dunn_index(ev, lv),
    )


def evaluate_unsupervised(emb, pred, pseudo_gt=None, metric="euclidean"):
    """Unsupervised metrics + optional ARI/NMI vs a pseudo ground truth (e.g. KMeans)."""
    pred = np.asarray(pred); ok = pred != -1
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


_METHOD_CFG = {
    "KMeans"      : ("kmeans",      "cosine", True),
    "Agg"         : ("agg",         "cosine", False),
    "GMM"         : ("gmm",         "cosine", True),
    "SBScan"      : ("sbscan",      "cosine", True),
    "POS"         : ("pos",         "cosine", True),
    "RandAssign"  : ("rand_assign", "cosine", True),
    "GSA"         : ("gsa",         "cosine", True),
    "RandClust"   : ("rand_clust",  "cosine", True),
}


def print_metrics(tr_norm, tr_pca, tr_pred, tr_lbl,
                  te_norm, te_pca, te_pred, te_lbl,
                  xu_norm=None, xu_pca=None, xu_pred=None):
    for split, pred, gt in [("TRAIN", tr_pred, tr_lbl), ("TEST", te_pred, te_lbl)]:
        rows = {}
        for name, (key, metric, use_pca) in _METHOD_CFG.items():
            emb_tr = tr_pca if use_pca else tr_norm
            emb_te = te_pca if use_pca else te_norm
            emb    = emb_tr if split == "TRAIN" else emb_te
            rows[name] = evaluate(emb, pred[key], gt, metric)
        logger.info(f"\n── {split} ──────────────────────────────────────")
        logger.info("\n" + pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))

    logger.info("\nGeneralisation gap  (train ARI − test ARI):")
    for name, (key, metric, use_pca) in _METHOD_CFG.items():
        emb_tr = tr_pca if use_pca else tr_norm
        emb_te = te_pca if use_pca else te_norm
        tr_v   = evaluate(emb_tr, tr_pred[key], tr_lbl, metric)["ARI"] or 0
        te_v   = evaluate(emb_te, te_pred[key], te_lbl, metric)["ARI"] or 0
        gap    = (tr_v or 0) - (te_v or 0)
        flag   = "good" if gap < 0.10 else "overfit" if gap > 0.20 else "acceptable"
        logger.info(f"  {name:12s}  train={tr_v:.4f}  test={te_v:.4f}  gap={gap:+.4f}  [{flag}]")

    if xu_pred is not None and xu_norm is not None and xu_pca is not None:
        km_ref = xu_pred["kmeans"]
        rows = {}
        for name, (key, metric, use_pca) in _METHOD_CFG.items():
            emb = xu_pca if use_pca else xu_norm
            rows[name] = evaluate_unsupervised(emb, xu_pred[key],
                                               pseudo_gt=km_ref, metric=metric)
        logger.info("\n── UNLABELED (unsupervised + ARI/NMI vs KMeans pseudo-GT) ──────────")
        logger.info("\n" + pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))


def save_unlabeled_predictions(xu_pred, cfg):
    """Save per-window cluster assignments for unlabeled data to CSV."""
    if xu_pred is None:
        return
    n  = len(xu_pred["kmeans"])
    df = pd.DataFrame({"window_idx": np.arange(n)})
    for key in xu_pred:
        df[f"cluster_{key}"] = xu_pred[key]
    out = cfg["output_csv"]
    df.to_csv(out, index=False)
    logger.info(f"  Unlabeled predictions saved: {Path(out).resolve()}  ({n} windows)")


# ── Visualisation ─────────────────────────────────────────────────────────────
COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261",
          "#8338ec","#06d6a0","#fb8500","#3a86ff","#ff006e",
          "#c77dff","#80b918"]

def _scatter(ax, xy, lbls, legend_fn, title, xlabel):
    for i, lbl in enumerate(sorted(set(lbls))):
        m = np.asarray(lbls) == lbl
        ax.scatter(xy[m, 0], xy[m, 1],
                   c="#bbb" if lbl == -1 else COLORS[i % len(COLORS)],
                   alpha=0.6, s=8, label=legend_fn(lbl))
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{xlabel} 1"); ax.set_ylabel(f"{xlabel} 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6)
    ax.grid(alpha=0.2)


def project(pca_emb, tag):
    perp = min(30, len(pca_emb) - 1)
    logger.info(f"  t-SNE [{tag}]...")
    ts = TSNE(n_components=2, random_state=42,
              perplexity=perp, max_iter=1000).fit_transform(pca_emb)
    return ts


def project_umap(pca_emb, tag):
    logger.info(f"  UMAP [{tag}]...")
    um = UMAP(n_components=2, random_state=42,
              n_neighbors=15, min_dist=0.1).fit_transform(pca_emb)
    return um


_ALL_METHODS = [
    ("kmeans",      "K-Means"),
    ("agg",         "Agglomerative"),
    ("gmm",         "GMM"),
    ("sbscan",      "SBScan"),
    ("pos",         "PSO"),
    ("rand_assign", "RandomAssign"),
    ("gsa",         "GravitationalSearch"),
    ("rand_clust",  "RandomClustering"),
]


def _plot_proj_grid(coords, proj_name, pred, c2s, gt, fname):
    """3×3 grid: 8 clustering methods + Ground Truth."""
    ncols, n_panels = 3, len(_ALL_METHODS) + 1
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 6))
    fig.suptitle(f"{proj_name} — Test split (super-classes)",
                 fontsize=13, fontweight="bold")
    axes_flat = iter(axes.flat)
    for key, title in _ALL_METHODS:
        ax     = next(axes_flat)
        leg_fn = (lambda lbl, k=key:
                  f"{super_name(c2s[k].get(lbl, '?'))} [c{lbl}]")
        _scatter(ax, coords, pred[key], leg_fn,
                 f"{proj_name} — {title}", proj_name)
    # Ground truth panel
    ax = next(axes_flat)
    for i, sid in enumerate(sorted(np.unique(gt))):
        m = np.asarray(gt) == sid
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=COLORS[i % len(COLORS)], alpha=0.6, s=8,
                   label=super_name(sid))
    ax.set_title(f"{proj_name} — Ground Truth", fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best",
              framealpha=0.6, title="Super-class")
    ax.grid(alpha=0.2)
    for ax in axes_flat:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


def plot_individual(coords, proj_name, prefix, pred, c2s, gt, figures_dir: Path):
    """Save one figure per clustering method plus one for ground truth."""
    for key, title in _ALL_METHODS:
        fig, ax = plt.subplots(figsize=(8, 6))
        leg_fn  = (lambda lbl, k=key:
                   f"{super_name(c2s[k].get(lbl, '?'))} [c{lbl}]")
        _scatter(ax, coords, pred[key], leg_fn, f"{proj_name} — {title}", proj_name)
        plt.tight_layout()
        fname = figures_dir / f"{prefix}_{title}_{RUN_TS}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {fname}")
    # Ground truth
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, sid in enumerate(sorted(np.unique(gt))):
        m = np.asarray(gt) == sid
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=COLORS[i % len(COLORS)], alpha=0.6, s=8,
                   label=super_name(sid))
    ax.set_title(f"{proj_name} — Ground Truth", fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best",
              framealpha=0.6, title="Super-class")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fname = figures_dir / f"{prefix}_GroundTruth_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


def plot_tsne_individual(ts, pred, c2s, gt, figures_dir: Path):
    plot_individual(ts, "t-SNE", "tsne", pred, c2s, gt, figures_dir)


def plot_grid(ts, pred, c2s, gt, figures_dir: Path):
    fname = figures_dir / f"18_tsne_test_clusters_{RUN_TS}.png"
    _plot_proj_grid(ts, "t-SNE", pred, c2s, gt, fname=fname)
    plot_tsne_individual(ts, pred, c2s, gt, figures_dir)


# ── Unlabeled visualisation (no ground truth) ─────────────────────────────────
def _plot_proj_grid_unsupervised(coords, proj_name, pred, fname):
    """3×3 grid of clustering results — no ground-truth panel (unlabeled)."""
    ncols, n_panels = 3, len(_ALL_METHODS)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 6))
    fig.suptitle(f"{proj_name} — Unlabeled windows (unsupervised)",
                 fontsize=13, fontweight="bold")
    axes_flat = iter(axes.flat)
    for key, title in _ALL_METHODS:
        ax = next(axes_flat)
        for i, cid in enumerate(sorted(set(pred[key]))):
            m = pred[key] == cid
            ax.scatter(coords[m, 0], coords[m, 1],
                       c=COLORS[i % len(COLORS)], alpha=0.6, s=8,
                       label=f"Cluster {cid}")
        ax.set_title(f"{proj_name} — {title}", fontsize=9, fontweight="bold")
        ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
        ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6)
        ax.grid(alpha=0.2)
    for ax in axes_flat:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


def plot_unlabeled_individual(coords, proj_name, prefix, pred, figures_dir: Path):
    """Save one figure per clustering method for unlabeled data."""
    for key, title in _ALL_METHODS:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cid in enumerate(sorted(set(pred[key]))):
            m = pred[key] == cid
            ax.scatter(coords[m, 0], coords[m, 1],
                       c=COLORS[i % len(COLORS)], alpha=0.6, s=8,
                       label=f"Cluster {cid}")
        ax.set_title(f"{proj_name} — {title} (unlabeled)", fontsize=9, fontweight="bold")
        ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
        ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fname = figures_dir / f"{prefix}_unlabeled_{title}_{RUN_TS}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_stem = Path(__file__).stem
    figures_dir = CONFIG["figures_base"] / f"{script_stem}_{RUN_TS}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=Path("logs"))

    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Run timestamp: {RUN_TS}")
    logger.info(f"Figures dir  : {figures_dir.resolve()}")

    # ── [0] Surface names ──────────────────────────────────────────────────────
    logger.info("\n[0] Surface names")
    load_surface_names(CONFIG["surface_types_csv"])

    # ── [1] Load data ──────────────────────────────────────────────────────────
    logger.info("\n[1] Load windowed data (labeled + unlabeled)")
    tr_X, tr_y, te_X, te_y, X_u = load_windowed_data(CONFIG)

    logger.info("\n[2] Super-class distribution (train):")
    for sc, n in sorted(zip(*np.unique(tr_y, return_counts=True))):
        logger.info(f"    {super_name(sc):30s}: {n} windows")

    # ── [3] Load TimesFM model ─────────────────────────────────────────────────
    logger.info(f"\n[3] Load TimesFM model: {CONFIG['timesfm_model_id']}")
    model = TimesFmModel.from_pretrained(CONFIG["timesfm_model_id"]).to(device)
    model.eval()
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── [4] Magnitude → TimesFM embeddings → StandardScaler → PCA (labeled) ──
    logger.info("\n[4] Embeddings + PCA (labeled train / test)")
    bs = CONFIG["timesfm_batch_size"]
    tr_mag = compute_magnitude(tr_X)   # (N_tr, T)
    te_mag = compute_magnitude(te_X)   # (N_te, T)
    tr_emb_raw = extract_timesfm_embeddings(model, tr_mag, device, bs)
    te_emb_raw = extract_timesfm_embeddings(model, te_mag, device, bs)
    scaler = StandardScaler().fit(tr_emb_raw)
    tr_emb = scaler.transform(tr_emb_raw)
    te_emb = scaler.transform(te_emb_raw)
    tr_norm, tr_pca, te_norm, te_pca, pca_obj = pca_reduce(
        tr_emb, te_emb, CONFIG["pca_variance"])

    # ── [4b] Embed unlabeled ───────────────────────────────────────────────────
    xu_norm = xu_pca = None
    if len(X_u) > 0:
        logger.info(f"\n[4b] Embed unlabeled windows ({len(X_u)})")
        xu_mag     = compute_magnitude(X_u)
        xu_emb_raw = extract_timesfm_embeddings(model, xu_mag, device, bs)
        xu_emb     = scaler.transform(xu_emb_raw)
        xu_norm    = normalize(xu_emb, norm="l2")
        xu_pca     = pca_obj.transform(xu_norm)
        logger.info(f"  Unlabeled embeddings: {xu_norm.shape}  PCA: {xu_pca.shape}")

    # ── [5] Clustering ─────────────────────────────────────────────────────────
    logger.info("\n[5] Clustering")
    n_surf = N_SUPER_CLASSES
    logger.info(f"  Super-classes={n_surf}  — sweeping K {max(2,n_surf-2)}..{n_surf+2}")
    k = best_k(tr_pca, max(2, n_surf - 2), n_surf + 2, figures_dir)
    tr_pred, te_pred, xu_pred, _ = cluster_all(
        tr_norm, tr_pca, te_norm, te_pca, k,
        xu_norm=xu_norm, xu_pca=xu_pca)

    tr_c2s = {m: c2s_map(tr_pred[m], tr_y) for m in tr_pred}
    te_c2s = {m: c2s_map(te_pred[m], te_y) for m in te_pred}

    logger.info("\n  K-Means cluster → surface (train):")
    for cid, sid in sorted(tr_c2s["kmeans"].items()):
        logger.info(f"    cluster {cid:2d} → {super_name(sid)}")
    logger.info("\n  SBScan cluster → surface (train):")
    for cid, sid in sorted(tr_c2s["sbscan"].items()):
        logger.info(f"    cluster {cid:2d} → {super_name(sid)}")

    # ── [6] Evaluation metrics (labeled) ──────────────────────────────────────
    logger.info("\n[6] Evaluation metrics (labeled)")
    print_metrics(tr_norm, tr_pca, tr_pred, tr_y,
                  te_norm, te_pca, te_pred, te_y,
                  xu_norm=xu_norm, xu_pca=xu_pca, xu_pred=xu_pred)

    # ── [6b] Save unlabeled predictions ───────────────────────────────────────
    logger.info("\n[6b] Save unlabeled predictions")
    save_unlabeled_predictions(xu_pred, CONFIG)

    # ── [7] Visualise labeled (t-SNE + UMAP) ──────────────────────────────────
    logger.info("\n[7] t-SNE + UMAP visualisation (labeled test split)")
    ts = project(te_pca, "Test")
    um = project_umap(te_pca, "Test")
    plot_grid(ts, te_pred, te_c2s, te_y, figures_dir)
    plot_individual(um, "UMAP", "umap", te_pred, te_c2s, te_y, figures_dir)

    # Fixed-K experiment on test split
    experiment_fixed_k(te_pca, ts, um, figures_dir, ks=(3, 5, 7, 11))

    # ── [9] Unlabeled visualisation ────────────────────────────────────────────
    if xu_pred is not None and xu_pca is not None:
        logger.info(f"\n[9] Unlabeled visualisation ({len(X_u)} windows)")
        ts_u = project(xu_pca, "Unlabeled")
        um_u = project_umap(xu_pca, "Unlabeled")

        fname_ts_u = figures_dir / f"tsne_unlabeled_clusters_{RUN_TS}.png"
        _plot_proj_grid_unsupervised(ts_u, "t-SNE", xu_pred, fname_ts_u)
        plot_unlabeled_individual(ts_u, "t-SNE", "tsne", xu_pred, figures_dir)

        fname_um_u = figures_dir / f"umap_unlabeled_clusters_{RUN_TS}.png"
        _plot_proj_grid_unsupervised(um_u, "UMAP", xu_pred, fname_um_u)
        plot_unlabeled_individual(um_u, "UMAP", "umap", xu_pred, figures_dir)

        # Fixed-K on unlabeled
        logger.info("\n  Fixed-K experiment on unlabeled data")
        fname_fk_ts = figures_dir / f"fixed_k_tsne_unlabeled_{RUN_TS}.png"
        rows_fk = _fixed_k_grid(xu_pca, ts_u, "t-SNE", fname_fk_ts, (3, 5, 7, 11))
        _print_quality(rows_fk)

        fname_fk_um = figures_dir / f"fixed_k_umap_unlabeled_{RUN_TS}.png"
        rows_fk_um = _fixed_k_grid(xu_pca, um_u, "UMAP", fname_fk_um, (3, 5, 7, 11))
        _print_quality(rows_fk_um)

    else:
        logger.info("\n[9] No unlabeled data — skipping unlabeled visualisation")

    logger.info(f"\nDone!  All plots saved to: {figures_dir.resolve()}")

if __name__ == "__main__":
    main()

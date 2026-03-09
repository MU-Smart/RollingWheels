"""
Accelerometer Features: Multi-Method Clustering + t-SNE & UMAP Visualization
------------------------------------------------------------------------------
Clustering methods: KMeans, Agglomerative, GMM, SBScan, PSO,
                    RandomAssign, GravitationalSearch, RandomClustering

Outputs (all under figures/{timestamp}_raw_feature_result/):
  • One scatter grid (t-SNE + UMAP) per clustering method
  • Metrics comparison bar chart across all methods
  • metrics_raw_feature_result.csv

Log: logs/run_{timestamp}_raw_feature_result.log

Usage:
    python raw_feature_clustering.py
    python raw_feature_clustering.py --data path/to/features.csv
    python raw_feature_clustering.py --labels path/to/labels.csv  # enables ARI/NMI
"""

import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

try:
    import umap
except ImportError:
    raise ImportError("Install umap-learn: pip install umap-learn")

import sklearn
_tsne_iter_key = (
    "max_iter" if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5)
    else "n_iter"
)

# ── Timestamp & run identity ──────────────────────────────────────────────────
RUN_TS      = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX  = "raw_feature_result"
RUN_NAME    = f"{RUN_TS}_{RUN_SUFFIX}"

FIGURES_DIR = Path("figures") / RUN_NAME
LOGS_DIR    = Path("logs")

# ── Config ────────────────────────────────────────────────────────────────────
N_CLUSTERS   = [3, 5, 7, 11]
TSNE_PARAMS  = {"n_components": 2, "perplexity": 30, "random_state": 42, _tsne_iter_key: 1000}
UMAP_PARAMS  = dict(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
PALETTE      = "tab10"

LABELED_FEATURES_CSV = "../../Datasets/ExtractedFeatures/labeled_accelerometer_features.csv"

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"run_{RUN_NAME}.log"
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(RUN_SUFFIX)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Run: {RUN_NAME}")
    logger.info(f"Log: {log_path.resolve()}")
    logger.info(f"Figures: {FIGURES_DIR.resolve()}")
    return logger

logger = logging.getLogger(RUN_SUFFIX)


# ── Metrics helper ────────────────────────────────────────────────────────────
def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Dunn Index = min inter-cluster centroid dist / max intra-cluster diameter.
    Higher is better."""
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


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_feature_df(path: str, extra_drop: list | None = None) -> pd.DataFrame:
    df   = pd.read_csv(path)
    drop = [c for c in df.columns if df[c].dtype == object]
    if extra_drop:
        drop += [c for c in extra_drop if c in df.columns]
    if drop:
        logger.info(f"  {Path(path).name}: dropping columns {drop}")
    return df.drop(columns=drop)


def load_and_scale(path: str) -> np.ndarray:
    """Load unlabeled + labeled (surface_id ignored) feature CSVs, concatenate, Z-normalise."""
    df_u = _load_feature_df(path)
    logger.info(f"  Unlabeled features : {df_u.shape}")

    try:
        df_l = _load_feature_df(LABELED_FEATURES_CSV, extra_drop=["surface_id"])
        logger.info(f"  Labeled features   : {df_l.shape}  (surface_id dropped)")
        common = df_u.columns.intersection(df_l.columns)
        df = pd.concat([df_u[common], df_l[common]], ignore_index=True)
        logger.info(f"  Combined           : {df.shape}")
    except FileNotFoundError:
        logger.warning("  Labeled features CSV not found — using unlabeled only.")
        df = df_u

    X = np.nan_to_num(df.values).astype(np.float64)
    X_scaled = StandardScaler().fit_transform(X)
    logger.info(f"  Final scaled shape : {X_scaled.shape}")
    return X_scaled


def load_labels(path: str) -> np.ndarray:
    df  = pd.read_csv(path)
    col = "label" if "label" in df.columns else df.columns[0]
    return df[col].values


# ══════════════════════════════════════════════════════════════════════════════
# Clustering algorithms
# ══════════════════════════════════════════════════════════════════════════════

class SBScanClustering:
    """DBSCAN with automatic epsilon via k-NN distance elbow."""
    def __init__(self, n_clusters_hint: int = 5, min_samples: int = 5):
        self.n_clusters_hint = n_clusters_hint
        self.min_samples     = min_samples
        self.labels_         = None
        self._knn_clf        = None

    def fit(self, X):
        nbrs     = NearestNeighbors(n_neighbors=self.min_samples).fit(X)
        dists, _ = nbrs.kneighbors(X)
        k_dists  = np.sort(dists[:, -1])
        n        = len(k_dists)
        xs       = np.linspace(0.0, 1.0, n)
        rng_d    = k_dists[-1] - k_dists[0]
        ys       = (k_dists - k_dists[0]) / (rng_d + 1e-10)
        eps      = float(k_dists[int(np.argmax(np.abs(ys - xs)))])
        if eps < 1e-6:
            eps = float(np.median(k_dists))
        logger.info(f"      SBScan auto-eps={eps:.4f}")
        db           = DBSCAN(eps=eps, min_samples=self.min_samples).fit(X)
        self.labels_ = db.labels_
        mask    = self.labels_ != -1
        n_valid = int(mask.sum())
        n_cls   = len(np.unique(self.labels_[mask])) if n_valid > 0 else 0
        logger.info(f"      SBScan: {n_cls} clusters, {n_valid}/{n} non-noise")
        if n_valid > 1 and n_cls > 1:
            self._knn_clf = KNeighborsClassifier(
                n_neighbors=min(5, n_valid)).fit(X[mask], self.labels_[mask])
        return self

    def predict(self, X):
        if self._knn_clf is None:
            return np.zeros(len(X), dtype=int)
        return self._knn_clf.predict(X)


class PSOClustering:
    """Particle Swarm Optimisation clustering (minimises WCSS)."""
    def __init__(self, n_clusters: int = 5, n_particles: int = 15,
                 max_iter: int = 50, seed: int = 42):
        self.n_clusters  = n_clusters
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.seed        = seed
        self.centroids_  = None
        self.labels_     = None

    def _assign(self, X, c):
        return np.linalg.norm(X[:, None] - c[None], axis=-1).argmin(axis=1)

    def _wcss(self, X, c):
        a = self._assign(X, c)
        return sum(((X[a == k] - c[k]) ** 2).sum()
                   for k in range(len(c)) if (a == k).any())

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        D, K, N = X.shape[1], self.n_clusters, self.n_particles
        particles  = X[rng.choice(len(X), N * K, replace=True)].reshape(N, K, D)
        velocities = rng.uniform(-0.1, 0.1, (N, K, D))
        pbest      = particles.copy()
        pbest_fit  = np.array([self._wcss(X, p) for p in particles])
        gbest      = pbest[pbest_fit.argmin()].copy()
        gbest_fit  = pbest_fit.min()
        w, c1, c2  = 0.5, 1.5, 1.5
        for _ in range(self.max_iter):
            r1 = rng.uniform(0, 1, (N, K, D))
            r2 = rng.uniform(0, 1, (N, K, D))
            velocities = (w * velocities + c1 * r1 * (pbest - particles)
                          + c2 * r2 * (gbest[None] - particles))
            particles += velocities
            fits = np.array([self._wcss(X, p) for p in particles])
            imp  = fits < pbest_fit
            pbest[imp] = particles[imp]; pbest_fit[imp] = fits[imp]
            if fits.min() < gbest_fit:
                gbest = particles[fits.argmin()].copy(); gbest_fit = fits.min()
        self.centroids_ = gbest
        self.labels_    = self._assign(X, gbest)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


class GravitationalSearchClustering:
    """Gravitational Search Algorithm clustering."""
    def __init__(self, n_clusters: int = 5, n_agents: int = 15,
                 max_iter: int = 50, G0: float = 100.0, alpha: float = 20.0,
                 seed: int = 42):
        self.n_clusters = n_clusters; self.n_agents = n_agents
        self.max_iter = max_iter; self.G0 = G0; self.alpha = alpha
        self.seed = seed; self.centroids_ = None; self.labels_ = None

    def _assign(self, X, c):
        return np.linalg.norm(X[:, None] - c[None], axis=-1).argmin(axis=1)

    def _fitness(self, X, c):
        a = self._assign(X, c)
        return sum(((X[a == k] - c[k]) ** 2).sum()
                   for k in range(len(c)) if (a == k).any())

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        D, K, N = X.shape[1], self.n_clusters, self.n_agents
        agents     = X[rng.choice(len(X), N * K, replace=True)].reshape(N, K, D)
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
                best_fit = fits.min(); best_agent = agents[fits.argmin()].copy()
            k_best  = max(2, int(N * (1 - t / self.max_iter)))
            top_idx = np.argsort(fits)[:k_best]
            acc = np.zeros((N, K, D))
            for i in range(N):
                for j in top_idx:
                    if i == j: continue
                    r       = np.linalg.norm(agents[j] - agents[i]) + 1e-10
                    acc[i] += G * masses[j] * (agents[j] - agents[i]) / r
            velocities = rng.uniform(0, 1, (N, K, D)) * velocities + acc
            agents    += velocities
        self.centroids_ = best_agent
        self.labels_    = self._assign(X, best_agent)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


class RandomAssignClustering:
    """Randomly assigns cluster IDs — baseline sanity check."""
    def __init__(self, n_clusters: int = 5, seed: int = 42):
        self.n_clusters = n_clusters; self.seed = seed; self.labels_ = None

    def fit(self, X):
        self.labels_ = np.random.default_rng(self.seed).integers(0, self.n_clusters, len(X))
        return self

    def predict(self, X):
        return np.random.default_rng(self.seed + 1).integers(0, self.n_clusters, len(X))


class RandomClustering:
    """Random centroid initialisation + nearest-neighbour assignment (no iteration)."""
    def __init__(self, n_clusters: int = 5, seed: int = 42):
        self.n_clusters = n_clusters; self.seed = seed
        self.centroids_ = None; self.labels_ = None

    def _assign(self, X, c):
        return np.linalg.norm(X[:, None] - c[None], axis=-1).argmin(axis=1)

    def fit(self, X):
        idxs = np.random.default_rng(self.seed).choice(len(X), self.n_clusters, replace=False)
        self.centroids_ = X[idxs]
        self.labels_    = self._assign(X, self.centroids_)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


# ── All methods registry ───────────────────────────────────────────────────────
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


def run_all_clustering(X: np.ndarray, n_clusters_list: list) -> dict:
    """
    Run all 8 clustering methods for each k.
    Returns: all_labels[method_key][k] = label_array
    """
    X_norm = normalize(X, norm="l2")   # Agglomerative uses cosine → needs L2-normed input
    all_labels: dict[str, dict] = {key: {} for key, _ in _ALL_METHODS}

    for k in n_clusters_list:
        logger.info(f"\n  ── k={k} ──────────────────────────────────────────")

        # KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        all_labels["kmeans"][k] = km.labels_
        logger.info(f"    KMeans        inertia={km.inertia_:.2f}")

        # Agglomerative (cosine on L2-normed)
        agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(X_norm)
        all_labels["agg"][k] = agg.labels_
        logger.info(f"    Agglomerative done")

        # GMM
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3).fit(X)
        all_labels["gmm"][k] = gmm.predict(X)
        logger.info(f"    GMM           lower_bound={gmm.lower_bound_:.4f}")

        # SBScan
        sbscan = SBScanClustering(n_clusters_hint=k, min_samples=5).fit(X)
        all_labels["sbscan"][k] = sbscan.labels_

        # PSO
        pos = PSOClustering(n_clusters=k, seed=42).fit(X)
        all_labels["pos"][k] = pos.labels_
        logger.info(f"    PSO           done")

        # RandomAssign
        ra = RandomAssignClustering(n_clusters=k, seed=42).fit(X)
        all_labels["rand_assign"][k] = ra.labels_
        logger.info(f"    RandomAssign  done")

        # GSA
        gsa = GravitationalSearchClustering(n_clusters=k, seed=42).fit(X)
        all_labels["gsa"][k] = gsa.labels_
        logger.info(f"    GSA           done")

        # RandomClustering
        rc = RandomClustering(n_clusters=k, seed=42).fit(X)
        all_labels["rand_clust"][k] = rc.labels_
        logger.info(f"    RandomClust   done")

    return all_labels


# ── Dimensionality reduction ───────────────────────────────────────────────────
def compute_tsne(X: np.ndarray) -> np.ndarray:
    logger.info("  t-SNE fitting …")
    emb = TSNE(**TSNE_PARAMS).fit_transform(X)
    logger.info("  t-SNE done.")
    return emb


def compute_umap(X: np.ndarray) -> np.ndarray:
    logger.info("  UMAP fitting …")
    emb = umap.UMAP(**UMAP_PARAMS).fit_transform(X)
    logger.info("  UMAP done.")
    return emb


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(X: np.ndarray, all_labels: dict, true_labels=None) -> pd.DataFrame:
    """
    Compute Silhouette, DB, CH, Dunn, ARI, NMI for every (method, k) pair.
    Returns a DataFrame indexed by (method, k).
    """
    rows = []
    for method_key, method_name in _ALL_METHODS:
        for k in sorted(all_labels[method_key].keys()):
            lbl  = np.asarray(all_labels[method_key][k])
            valid = lbl[lbl != -1]
            if len(np.unique(valid)) < 2:
                logger.warning(f"  {method_name} k={k}: fewer than 2 clusters — skipping metrics")
                continue
            ok   = lbl != -1
            Xv, lv = X[ok], lbl[ok]
            row = {
                "Method"           : method_name,
                "k"                : k,
                "Silhouette"       : round(silhouette_score(Xv, lv), 4),
                "Davies-Bouldin"   : round(davies_bouldin_score(Xv, lv), 4),
                "Calinski-Harabasz": round(calinski_harabasz_score(Xv, lv), 2),
                "Dunn"             : round(dunn_index(Xv, lv), 4),
                "ARI"              : (round(adjusted_rand_score(true_labels[ok], lv), 4)
                                      if true_labels is not None else None),
                "NMI"              : (round(normalized_mutual_info_score(true_labels[ok], lv), 4)
                                      if true_labels is not None else None),
            }
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["Method", "k"])

    logger.info("\n" + "=" * 70)
    logger.info("  CLUSTERING METRICS")
    logger.info("  Silhouette ↑  Davies-Bouldin ↓  Calinski-Harabasz ↑  Dunn ↑")
    if true_labels is not None:
        logger.info("  ARI ↑  NMI ↑")
    logger.info("=" * 70)
    logger.info("\n" + df.to_string(float_format=lambda x: f"{x:.4f}"))
    return df


def save_metrics(metrics_df: pd.DataFrame):
    path = FIGURES_DIR / f"metrics_{RUN_SUFFIX}.csv"
    metrics_df.to_csv(path)
    logger.info(f"  Metrics saved: {path.resolve()}")


# ── Cluster membership summary ─────────────────────────────────────────────────
def print_cluster_membership(all_labels: dict):
    logger.info("\n" + "=" * 60)
    logger.info("  CLUSTER MEMBERSHIP SUMMARY")
    logger.info("=" * 60)
    for method_key, method_name in _ALL_METHODS:
        for k in sorted(all_labels[method_key].keys()):
            lbl    = all_labels[method_key][k]
            counts = pd.Series(lbl).value_counts().sort_index()
            total  = len(lbl)
            logger.info(f"\n  {method_name}  k={k}")
            logger.info(f"  {'Cluster':<12} {'Count':>8} {'%':>8}")
            logger.info(f"  {'-'*30}")
            for cid, cnt in counts.items():
                tag = " (noise)" if cid == -1 else ""
                logger.info(f"  C{cid:<11} {cnt:>8}  ({100*cnt/total:5.1f}%){tag}")


# ── Metric bar chart ───────────────────────────────────────────────────────────
def plot_metrics(metrics_df: pd.DataFrame):
    """Bar chart comparing all methods for each metric, one subplot per k."""
    has_sup = metrics_df["ARI"].notna().any() if "ARI" in metrics_df.columns else False
    metric_cols = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Dunn"]
    if has_sup:
        metric_cols += ["ARI", "NMI"]

    k_vals   = sorted(metrics_df.index.get_level_values("k").unique())
    methods  = [name for _, name in _ALL_METHODS]
    n_k      = len(k_vals)
    n_m      = len(metric_cols)

    fig, axes = plt.subplots(n_m, n_k, figsize=(4 * n_k, 3.5 * n_m), squeeze=False)
    fig.suptitle("Clustering Metrics — All Methods", fontsize=13, fontweight="bold")

    for row, metric in enumerate(metric_cols):
        for col, k in enumerate(k_vals):
            ax = axes[row][col]
            vals = []
            for method_name in methods:
                try:
                    vals.append(metrics_df.loc[(method_name, k), metric])
                except KeyError:
                    vals.append(np.nan)
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(methods)))
            bars = ax.bar(range(len(methods)), vals, color=colors)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{metric} | k={k}", fontsize=9, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    fname = FIGURES_DIR / f"metrics_comparison_{RUN_SUFFIX}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


# ── Scatter grid — one file per clustering method ────────────────────────────
def scatter_grid_for_method(method_key: str, method_name: str,
                             labels_by_k: dict, emb_tsne: np.ndarray,
                             emb_umap: np.ndarray, metrics_df: pd.DataFrame):
    """Save a (2 projections × N_CLUSTERS) scatter grid for one clustering method."""
    projections = [("t-SNE", emb_tsne), ("UMAP", emb_umap)]
    n_proj = len(projections)
    n_k    = len(N_CLUSTERS)

    fig = plt.figure(figsize=(5 * n_k, 4.5 * n_proj))
    gs  = gridspec.GridSpec(n_proj, n_k, figure=fig, hspace=0.45, wspace=0.25)
    fig.suptitle(f"{method_name} Clustering — t-SNE & UMAP",
                 fontsize=12, fontweight="bold")

    for row, (proj_name, emb) in enumerate(projections):
        for col, k in enumerate(N_CLUSTERS):
            ax  = fig.add_subplot(gs[row, col])
            lbl = np.asarray(labels_by_k.get(k, np.zeros(len(emb), dtype=int)))
            unique_lbls = sorted(set(lbl))
            cmap = plt.get_cmap(PALETTE, max(len(unique_lbls), 1))
            for i, cid in enumerate(unique_lbls):
                mask  = lbl == cid
                count = mask.sum()
                color = "#aaaaaa" if cid == -1 else cmap(i)
                label = f"noise (n={count})" if cid == -1 else f"C{cid} (n={count})"
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           s=8, alpha=0.7, color=color, label=label, linewidths=0)

            # Metric annotations
            try:
                sil  = metrics_df.loc[(method_name, k), "Silhouette"]
                db   = metrics_df.loc[(method_name, k), "Davies-Bouldin"]
                dunn = metrics_df.loc[(method_name, k), "Dunn"]
                ann  = f"Sil={sil:.3f}  DB={db:.3f}  Dunn={dunn:.3f}"
            except KeyError:
                ann = ""
            ax.set_title(f"{proj_name} | k={k}\n{ann}", fontsize=8, pad=4)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Dim 1", fontsize=8); ax.set_ylabel("Dim 2", fontsize=8)
            ax.legend(fontsize=5, markerscale=2, loc="best",
                      title="Cluster", title_fontsize=6, framealpha=0.6)

    plt.tight_layout()
    fname = FIGURES_DIR / f"{method_key}_{RUN_SUFFIX}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="../../Datasets/ExtractedFeatures/unlabeled_accelerometer_features.csv")
    parser.add_argument("--labels", default=None,
                        help="Optional CSV with ground-truth labels (enables ARI/NMI)")
    args = parser.parse_args()

    true_labels = None
    if args.labels:
        true_labels = load_labels(args.labels)
        logger.info(f"Ground-truth labels loaded: {len(true_labels)} samples")

    # 1. Load & scale
    logger.info("\n[1] Load & scale features")
    X = load_and_scale(args.data)

    # 2. All clustering methods × all k
    logger.info("\n[2] Clustering (8 methods × 4 k-values)")
    all_labels = run_all_clustering(X, N_CLUSTERS)

    # 3. Membership summary
    logger.info("\n[3] Cluster membership")
    print_cluster_membership(all_labels)

    # 4. Metrics
    logger.info("\n[4] Computing metrics (Silhouette, DB, CH, Dunn, ARI, NMI)")
    metrics_df = compute_metrics(X, all_labels, true_labels)
    save_metrics(metrics_df)

    # 5. Metrics bar chart
    logger.info("\n[5] Metrics comparison plot")
    plot_metrics(metrics_df)

    # 6. Dimensionality reduction
    logger.info("\n[6] Dimensionality reduction")
    emb_tsne = compute_tsne(X)
    emb_umap = compute_umap(X)

    # 7. Scatter grids — one per clustering method
    logger.info("\n[7] Scatter grids (one figure per method)")
    for method_key, method_name in _ALL_METHODS:
        scatter_grid_for_method(
            method_key, method_name,
            all_labels[method_key], emb_tsne, emb_umap, metrics_df,
        )

    logger.info(f"\nDone!  Outputs in: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()

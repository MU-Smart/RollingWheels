"""
Accelerometer Features: KMeans Clustering + t-SNE & UMAP Visualization
-----------------------------------------------------------------------
Usage:
    python cluster_visualize.py --data ../Datasets/ExtractedFeatures/accelerometer_features.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    raise ImportError("Install umap-learn: pip install umap-learn")

# ── Config ──────────────────────────────────────────────────────────────────
N_CLUSTERS   = [3, 5, 7, 11]
import sklearn
_tsne_iter_key = "max_iter" if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5) else "n_iter"
TSNE_PARAMS  = {"n_components": 2, "perplexity": 30, "random_state": 42, _tsne_iter_key: 1000}
UMAP_PARAMS  = dict(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
PALETTE      = "tab10"

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_and_scale(path: str):
    df = pd.read_csv(path)
    # Drop non-numeric or label columns if present
    drop_cols = [c for c in df.columns if df[c].dtype == object]
    if drop_cols:
        print(f"[info] Dropping non-numeric columns: {drop_cols}")
    X = df.drop(columns=drop_cols).values
    X = np.nan_to_num(X)          # replace any NaN/Inf
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[info] Data shape: {X_scaled.shape}")
    return X_scaled


def run_clustering(X, n_clusters_list):
    labels = {}
    for n in n_clusters_list:
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels[n] = km.fit_predict(X)
        print(f"[kmeans] n={n}  inertia={km.inertia_:.2f}")
    return labels


def compute_tsne(X):
    print("[tsne] Fitting …")
    emb = TSNE(**TSNE_PARAMS).fit_transform(X)
    print("[tsne] Done.")
    return emb


def compute_umap(X):
    print("[umap] Fitting …")
    reducer = umap.UMAP(**UMAP_PARAMS)
    emb = reducer.fit_transform(X)
    print("[umap] Done.")
    return emb


def scatter_grid(embeddings_dict, labels_dict, title_prefix, filename):
    """
    embeddings_dict : {"t-SNE": emb_tsne, "UMAP": emb_umap}
    labels_dict     : {3: arr, 5: arr, 7: arr, 11: arr}
    """
    methods = list(embeddings_dict.keys())
    n_methods = len(methods)
    n_k = len(N_CLUSTERS)

    fig = plt.figure(figsize=(5 * n_k, 4.5 * n_methods))
    fig.patch.set_facecolor("#0f1117")

    gs = gridspec.GridSpec(n_methods, n_k, figure=fig,
                           hspace=0.35, wspace=0.25)

    for row, method in enumerate(methods):
        emb = embeddings_dict[method]
        for col, n in enumerate(N_CLUSTERS):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#1a1d27")

            lbl = labels_dict[n]
            cmap = plt.get_cmap(PALETTE, n)

            for k in range(n):
                mask = lbl == k
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           s=8, alpha=0.7, color=cmap(k),
                           label=f"C{k}", linewidths=0)

            ax.set_title(f"{method}  |  k = {n}",
                         color="white", fontsize=11, pad=6)
            ax.tick_params(colors="#555", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.set_xlabel("Dim 1", color="#888", fontsize=8)
            ax.set_ylabel("Dim 2", color="#888", fontsize=8)

    fig.suptitle(title_prefix, color="white", fontsize=15,
                 fontweight="bold", y=1.01)

    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[saved] {filename}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../Datasets/ExtractedFeatures/accelerometer_features.csv",
                        help="Path to accelerometer_features.csv")
    args = parser.parse_args()

    # 1. Load & scale
    X = load_and_scale(args.data)

    # 2. Cluster
    labels = run_clustering(X, N_CLUSTERS)

    # 3. Dimensionality reduction
    emb_tsne = compute_tsne(X)
    emb_umap = compute_umap(X)

    embeddings = {"t-SNE": emb_tsne, "UMAP": emb_umap}

    # 4. Combined grid plot (both methods, all k values)
    scatter_grid(embeddings, labels,
                 title_prefix="Accelerometer Features — KMeans Clustering (k = 3, 5, 7, 11)",
                 filename="clustering_tsne_umap_combined.png")

    # 5. Separate t-SNE plot
    scatter_grid({"t-SNE": emb_tsne}, labels,
                 title_prefix="t-SNE Projection — KMeans Clustering",
                 filename="clustering_tsne.png")

    # 6. Separate UMAP plot
    scatter_grid({"UMAP": emb_umap}, labels,
                 title_prefix="UMAP Projection — KMeans Clustering",
                 filename="clustering_umap.png")

    print("\n✅  All plots saved.")


if __name__ == "__main__":
    main()
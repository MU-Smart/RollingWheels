"""
Accelerometer Features: KMeans Clustering + t-SNE & UMAP Visualization
-----------------------------------------------------------------------
Usage:
    python raw_feature_clustering.py --data ../Datasets/ExtractedFeatures/accelerometer_features.csv
    python raw_feature_clustering.py --data ... --labels ../Datasets/labels.csv   # enables ARI / MI
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

# ── Config ───────────────────────────────────────────────────────────────────
N_CLUSTERS = [3, 5, 7, 11]
import sklearn
_tsne_iter_key = "max_iter" if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5) else "n_iter"
TSNE_PARAMS = {"n_components": 2, "perplexity": 30, "random_state": 42, _tsne_iter_key: 1000}
UMAP_PARAMS = dict(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
PALETTE     = "tab10"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_and_scale(path: str):
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if df[c].dtype == object]
    if drop_cols:
        print(f"[info] Dropping non-numeric columns: {drop_cols}")
    X = df.drop(columns=drop_cols).values
    X = np.nan_to_num(X)
    X_scaled = StandardScaler().fit_transform(X)
    print(f"[info] Data shape: {X_scaled.shape}")
    return X_scaled


def load_labels(path: str):
    """Load ground-truth labels from a single-column CSV (no header assumed, or column named 'label')."""
    df = pd.read_csv(path)
    col = "label" if "label" in df.columns else df.columns[0]
    return df[col].values


# ── Clustering ────────────────────────────────────────────────────────────────
def run_clustering(X, n_clusters_list):
    labels = {}
    for n in n_clusters_list:
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels[n] = km.fit_predict(X)
        print(f"[kmeans] k={n}  inertia={km.inertia_:.2f}")
    return labels


# ── Dimensionality reduction ──────────────────────────────────────────────────
def compute_tsne(X):
    print("[tsne] Fitting …")
    emb = TSNE(**TSNE_PARAMS).fit_transform(X)
    print("[tsne] Done.")
    return emb


def compute_umap(X):
    print("[umap] Fitting …")
    emb = umap.UMAP(**UMAP_PARAMS).fit_transform(X)
    print("[umap] Done.")
    return emb


# ── Cluster membership summary ────────────────────────────────────────────────
def print_cluster_membership(labels_dict):
    print("\n" + "=" * 60)
    print("  CLUSTER MEMBERSHIP SUMMARY")
    print("=" * 60)
    for n, lbl in sorted(labels_dict.items()):
        counts = pd.Series(lbl).value_counts().sort_index()
        total  = len(lbl)
        print(f"\n  k = {n}")
        print(f"  {'Cluster':<10} {'Count':>8} {'%':>8}")
        print(f"  {'-'*28}")
        for cluster_id, count in counts.items():
            print(f"  C{cluster_id:<9} {count:>8}  ({100*count/total:5.1f}%)")


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(X, labels_dict, true_labels=None):
    rows = []
    for n, lbl in sorted(labels_dict.items()):
        row = {
            "k":                        n,
            "Silhouette":               round(silhouette_score(X, lbl), 4),
            "Davies-Bouldin":           round(davies_bouldin_score(X, lbl), 4),
            "Calinski-Harabasz":        round(calinski_harabasz_score(X, lbl), 2),
            "ARI":                      round(adjusted_rand_score(true_labels, lbl), 4)          if true_labels is not None else None,
            "NMI":                      round(normalized_mutual_info_score(true_labels, lbl), 4) if true_labels is not None else None,
        }
        rows.append(row)

    metrics_df = pd.DataFrame(rows).set_index("k")
    print("\n" + "=" * 60)
    print("  CLUSTERING METRICS  (↑ = higher is better, ↓ = lower is better)")
    print("=" * 60)
    print(f"\n  Silhouette Score       ↑  (best: 1.0,  worst: -1.0)")
    print(f"  Davies-Bouldin Index   ↓  (best: 0)")
    print(f"  Calinski-Harabasz      ↑  (higher = more compact/separated)")
    if true_labels is not None:
        print(f"  ARI                    ↑  (best: 1.0,  random: 0)")
        print(f"  NMI                    ↑  (best: 1.0,  worst: 0)")
    else:
        print(f"  ARI / NMI              —  (no ground-truth labels provided)")
    print()
    print(metrics_df.to_string())
    return metrics_df


def plot_metrics(metrics_df):
    has_supervised = metrics_df["ARI"].notna().any()
    metric_specs = [
        ("Silhouette",         "↑ higher is better",  "#4c72b0"),
        ("Davies-Bouldin",     "↓ lower is better",   "#dd8452"),
        ("Calinski-Harabasz",  "↑ higher is better",  "#55a868"),
    ]
    if has_supervised:
        metric_specs += [
            ("ARI", "↑ higher is better", "#c44e52"),
            ("NMI", "↑ higher is better", "#8172b2"),
        ]

    ncols = 3
    nrows = int(np.ceil(len(metric_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    k_vals = metrics_df.index.tolist()
    for i, (metric, note, color) in enumerate(metric_specs):
        ax = axes[i]
        vals = metrics_df[metric].values
        ax.plot(k_vals, vals, marker="o", color=color, linewidth=2, markersize=8)
        for x, y in zip(k_vals, vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9)
        ax.set_title(f"{metric}\n({note})", fontsize=11)
        ax.set_xlabel("k (number of clusters)")
        ax.set_xticks(k_vals)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Clustering Metrics vs k", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
    plt.close()


# ── Scatter grid ──────────────────────────────────────────────────────────────
def scatter_grid(embeddings_dict, labels_dict, metrics_df, title_prefix, filename):
    methods  = list(embeddings_dict.keys())
    n_k      = len(N_CLUSTERS)

    fig = plt.figure(figsize=(5 * n_k, 4.5 * len(methods)))
    gs  = gridspec.GridSpec(len(methods), n_k, figure=fig, hspace=0.45, wspace=0.25)

    for row, method in enumerate(methods):
        emb = embeddings_dict[method]
        for col, n in enumerate(N_CLUSTERS):
            ax  = fig.add_subplot(gs[row, col])
            lbl = labels_dict[n]
            cmap = plt.get_cmap(PALETTE, n)

            for k in range(n):
                mask  = lbl == k
                count = mask.sum()
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           s=8, alpha=0.7, color=cmap(k),
                           label=f"C{k} (n={count})", linewidths=0)

            # Metric annotations
            sil = metrics_df.loc[n, "Silhouette"]
            db  = metrics_df.loc[n, "Davies-Bouldin"]
            ax.set_title(f"{method}  |  k={n}\nSil={sil:.3f}  DB={db:.3f}",
                         fontsize=9, pad=5)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Dim 1", fontsize=8)
            ax.set_ylabel("Dim 2", fontsize=8)

            # Legend with cluster sizes
            ax.legend(fontsize=6, markerscale=2, loc="best",
                      title="Cluster", title_fontsize=7, framealpha=0.6)

    fig.suptitle(title_prefix, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="../Datasets/ExtractedFeatures/accelerometer_features.csv")
    parser.add_argument("--labels", default=None,
                        help="Optional CSV with ground-truth labels (enables ARI / NMI)")
    args = parser.parse_args()

    true_labels = load_labels(args.labels) if args.labels else None
    if true_labels is not None:
        print(f"[info] Loaded ground-truth labels: {len(true_labels)} samples")

    # 1. Load & scale
    X = load_and_scale(args.data)

    # 2. Cluster
    labels = run_clustering(X, N_CLUSTERS)

    # 3. Cluster membership
    print_cluster_membership(labels)

    # 4. Metrics
    metrics_df = compute_metrics(X, labels, true_labels)

    # 5. Metrics plot
    plot_metrics(metrics_df)

    # 6. Dimensionality reduction
    emb_tsne = compute_tsne(X)
    emb_umap = compute_umap(X)
    embeddings = {"t-SNE": emb_tsne, "UMAP": emb_umap}

    # 7. Combined grid (t-SNE + UMAP)
    scatter_grid(embeddings, labels, metrics_df,
                 title_prefix="Accelerometer Features — KMeans Clustering (k = 3, 5, 7, 11)",
                 filename="clustering_tsne_umap_combined.png")

    # 8. Separate plots
    scatter_grid({"t-SNE": emb_tsne}, labels, metrics_df,
                 title_prefix="t-SNE Projection — KMeans Clustering",
                 filename="clustering_tsne.png")
    scatter_grid({"UMAP": emb_umap}, labels, metrics_df,
                 title_prefix="UMAP Projection — KMeans Clustering",
                 filename="clustering_umap.png")

    print("\n✅  Done.")


if __name__ == "__main__":
    main()

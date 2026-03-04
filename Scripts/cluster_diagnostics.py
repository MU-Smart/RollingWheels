"""
Cluster Diagnostics — Surface Vibration Embeddings
===================================================
Run AFTER surface_clustering.py has trained the model and produced embeddings.
Pass your embeddings, ground-truth labels, and predicted cluster labels into
run_all_diagnostics() to get a full visual report.

What each plot tells you
────────────────────────
1. t-SNE side-by-side        → are GT blobs == cluster blobs?
2. Confusion matrix          → which surfaces get mixed up?
3. Cosine similarity heatmap → are same-surface embeddings close?
4. Intra vs inter distances  → quantify separation (the key test)
5. Per-surface silhouette    → which individual surfaces are weakest?
6. Cluster purity bars       → how pure is each predicted cluster?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.preprocessing import normalize

# ── Surface name map (keep in sync with surface_clustering.py) ───────────────
SURFACE_NAMES = {
    1: 'Asphalt', 2: 'Concrete', 3: 'Cobblestone', 4: 'Gravel',
    5: 'Grass',   6: 'Dirt',     7: 'Sand',         8: 'Tiles',
}
def sname(sid): return SURFACE_NAMES.get(int(sid), f'Surface {sid}')

PALETTE = [
    '#e63946','#457b9d','#2a9d8f','#e9c46a','#f4a261',
    '#8338ec','#06d6a0','#fb8500','#3a86ff','#ff006e','#ccc','#a8dadc'
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. t-SNE side-by-side:  Ground Truth  |  Predicted Clusters
# ─────────────────────────────────────────────────────────────────────────────
def plot_tsne_comparison(embeddings, gt_labels, pred_labels,
                         cluster_to_surface, method_name='K-Means'):
    print("  Computing t-SNE …")
    coords = TSNE(n_components=2, random_state=42,
                  perplexity=30, max_iter=1000).fit_transform(embeddings)

    unique_surfaces = sorted(set(gt_labels))
    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(unique_surfaces)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f't-SNE: Ground Truth vs {method_name} Clusters', fontsize=14)

    # ── Left: ground truth coloured by true surface ───────────────────────────
    for surf in unique_surfaces:
        mask = np.asarray(gt_labels) == surf
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        c=color_map[surf], alpha=0.55, s=10, label=sname(surf))
    axes[0].set_title('Ground Truth', fontweight='bold')
    axes[0].legend(fontsize=7, markerscale=2, title='Surface', loc='best')
    axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
    axes[0].grid(alpha=0.2)

    # ── Right: predicted clusters, coloured by dominant surface ──────────────
    # Use the SAME colour as ground truth so matching is instant visually
    pred_arr = np.asarray(pred_labels)
    for cluster_id, surf_id in sorted(cluster_to_surface.items()):
        if cluster_id == -1:
            color, label = '#aaaaaa', 'Noise'
        else:
            color = color_map.get(surf_id, '#cccccc')
            label = f'{sname(surf_id)} [c{cluster_id}]'
        mask = pred_arr == cluster_id
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        c=color, alpha=0.55, s=10, label=label)
    axes[1].set_title(f'{method_name} Predictions\n(colour = dominant surface)',
                      fontweight='bold')
    axes[1].legend(fontsize=7, markerscale=2, title='Cluster→Surface', loc='best')
    axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig('diag_tsne_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved diag_tsne_comparison.png")
    return coords   # reuse in other plots


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion matrix  (rows = true surface, cols = predicted cluster)
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion(gt_labels, pred_labels, cluster_to_surface, method_name='K-Means'):
    gt   = np.asarray(gt_labels)
    pred = np.asarray(pred_labels)
    valid = pred != -1          # exclude DBSCAN noise

    unique_gt   = sorted(set(gt[valid]))
    unique_pred = sorted(set(pred[valid]))

    cm_raw = confusion_matrix(gt[valid], pred[valid],
                               labels=unique_gt if len(unique_gt) < len(unique_pred)
                               else unique_pred)

    # Row labels = surface names, col labels = cluster → surface
    row_labels = [sname(s) for s in unique_gt]
    col_labels = [f'c{c}\n{sname(cluster_to_surface.get(c,"?"))}' for c in unique_pred]

    # Normalise rows to show % of each surface that went where
    cm_pct = cm_raw.astype(float)
    row_sums = cm_pct.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_pct, row_sums, where=row_sums > 0) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(unique_pred)*1.1),
                                    max(6, len(unique_gt)*0.9)))
    sns.heatmap(cm_pct, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=col_labels, yticklabels=row_labels,
                linewidths=0.4, linecolor='#ddd', ax=ax,
                cbar_kws={'label': '% of true surface → cluster'})
    ax.set_title(f'Confusion Matrix — {method_name}\n'
                 f'(rows = true surface, values = % assigned to each cluster)',
                 fontsize=11)
    ax.set_xlabel('Predicted cluster'); ax.set_ylabel('True surface')
    plt.tight_layout()
    plt.savefig('diag_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved diag_confusion.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cosine similarity heatmap  (mean pairwise sim between every surface pair)
#    KEY TEST: diagonal >> off-diagonal means good separation
# ─────────────────────────────────────────────────────────────────────────────
def plot_cosine_heatmap(embeddings, gt_labels):
    emb_norm     = normalize(embeddings, norm='l2', axis=1)
    unique_surfs = sorted(set(gt_labels))
    n            = len(unique_surfs)
    sim_matrix   = np.zeros((n, n))

    for i, si in enumerate(unique_surfs):
        for j, sj in enumerate(unique_surfs):
            zi = emb_norm[np.asarray(gt_labels) == si]
            zj = emb_norm[np.asarray(gt_labels) == sj]
            # Mean cosine similarity between the two groups
            sim_matrix[i, j] = (zi @ zj.T).mean()

    labels = [sname(s) for s in unique_surfs]
    fig, ax = plt.subplots(figsize=(max(7, n), max(6, n - 1)))
    sns.heatmap(sim_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=labels, yticklabels=labels,
                vmin=-1, vmax=1, center=0,
                linewidths=0.4, linecolor='#ccc', ax=ax,
                cbar_kws={'label': 'Mean cosine similarity'})
    ax.set_title('Mean Cosine Similarity Between Surfaces\n'
                 'Diagonal ≈ 1.0 = tight clusters  |  Off-diagonal ≈ 0 or negative = well separated',
                 fontsize=10)
    plt.xticks(rotation=35, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('diag_cosine_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved diag_cosine_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Intra- vs inter-cluster distance boxplot
#    THE most intuitive check: intra boxes should sit far BELOW inter boxes
# ─────────────────────────────────────────────────────────────────────────────
def plot_intra_inter_distances(embeddings, gt_labels, sample_size=3000):
    emb_norm = normalize(embeddings, norm='l2', axis=1)
    gt       = np.asarray(gt_labels)
    rng      = np.random.default_rng(42)

    # Sub-sample for speed
    idx = rng.choice(len(emb_norm), min(sample_size, len(emb_norm)), replace=False)
    emb_s, gt_s = emb_norm[idx], gt[idx]

    # Cosine distance = 1 - cosine_similarity
    sim   = emb_s @ emb_s.T                      # (N, N)
    dist  = 1 - sim

    same_mask = gt_s[:, None] == gt_s[None, :]   # (N, N) bool
    upper     = np.triu(np.ones_like(same_mask, dtype=bool), k=1)

    intra_dists = dist[same_mask & upper]
    inter_dists = dist[~same_mask & upper]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([intra_dists, inter_dists],
                    labels=['Intra-surface\n(same)', 'Inter-surface\n(different)'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#2a9d8f')
    bp['boxes'][1].set_facecolor('#e63946')

    ax.set_ylabel('Cosine distance  (lower = more similar)')
    ax.set_title('Intra- vs Inter-Surface Cosine Distance\n'
                 'Good clustering: green box well below red box, minimal overlap')
    ax.grid(axis='y', alpha=0.3)

    # Annotate medians
    for i, dists in enumerate([intra_dists, inter_dists], 1):
        ax.text(i, np.median(dists) + 0.01, f'med={np.median(dists):.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    overlap = (intra_dists > inter_dists.min()).mean() * 100
    ax.set_title(ax.get_title() + f'\nIntra-max overlap with inter: {overlap:.1f}%')

    plt.tight_layout()
    plt.savefig('diag_intra_inter.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved diag_intra_inter.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-surface silhouette scores  (which surfaces are hardest to separate?)
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_surface_silhouette(embeddings, gt_labels):
    gt        = np.asarray(gt_labels)
    sil_vals  = silhouette_samples(embeddings, gt, metric='cosine')
    unique_s  = sorted(set(gt))
    medians   = {s: np.median(sil_vals[gt == s]) for s in unique_s}
    sorted_s  = sorted(unique_s, key=lambda s: medians[s], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    y_lower = 0
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(sorted_s))]

    for i, surf in enumerate(sorted_s):
        vals = np.sort(sil_vals[gt == surf])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         alpha=0.8, color=colors[i], label=sname(surf))
        ax.text(-0.05, (y_lower + y_upper) / 2, sname(surf),
                ha='right', va='center', fontsize=8)
        y_lower = y_upper + 10   # gap between surfaces

    ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
    ax.axvline(x=np.mean(sil_vals), color='red', linewidth=1.5,
               linestyle='--', label=f'Mean = {np.mean(sil_vals):.3f}')
    ax.set_xlabel('Silhouette coefficient')
    ax.set_title('Per-Surface Silhouette Scores\n'
                 'Wider rightward bars = tighter, better-separated cluster')
    ax.set_yticks([])
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('diag_silhouette_per_surface.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved diag_silhouette_per_surface.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cluster purity bars  (what % of each predicted cluster is the right surface?)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_purity(gt_labels, pred_labels, cluster_to_surface, method_name='K-Means'):
    gt, pred = np.asarray(gt_labels), np.asarray(pred_labels)
    clusters = sorted(c for c in set(pred) if c != -1)
    purities, dominant_surfs, counts = [], [], []

    for c in clusters:
        mask  = pred == c
        total = mask.sum()
        dom   = cluster_to_surface[c]
        purity = (gt[mask] == dom).mean() * 100
        purities.append(purity); dominant_surfs.append(dom); counts.append(total)

    x      = np.arange(len(clusters))
    colors = [PALETTE[unique_surfs.index(s) % len(PALETTE)]
              if (unique_surfs := sorted(set(gt))) and s in unique_surfs else '#ccc'
              for s in dominant_surfs]

    fig, ax = plt.subplots(figsize=(max(8, len(clusters) * 0.9), 5))
    bars = ax.bar(x, purities, color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(100, color='green', linestyle='--', alpha=0.5, label='Perfect purity')
    ax.axhline(np.mean(purities), color='red', linestyle='--', alpha=0.7,
               label=f'Mean purity = {np.mean(purities):.1f}%')

    ax.set_xticks(x)
    ax.set_xticklabels([f'c{c}\n{sname(cluster_to_surface[c])}\n(n={counts[i]})'
                        for i, c in enumerate(clusters)], fontsize=8)
    ax.set_ylabel('Purity (%)')
    ax.set_ylim(0, 110)
    ax.set_title(f'Cluster Purity — {method_name}\n'
                 f'(% of samples in each cluster that match the dominant surface)')
    ax.legend()

    for bar, p in zip(bars, purities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{p:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('diag_cluster_purity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved diag_cluster_purity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Master function — call this with your embeddings and labels
# ─────────────────────────────────────────────────────────────────────────────
def run_all_diagnostics(embeddings, gt_labels, pred_labels,
                        cluster_to_surface, method_name='K-Means'):
    """
    Parameters
    ──────────
    embeddings         : np.ndarray (N, D)  — L2-normalised bottleneck embeddings
    gt_labels          : np.ndarray (N,)    — true surface IDs (integers)
    pred_labels        : np.ndarray (N,)    — cluster assignments from one method
    cluster_to_surface : dict               — {cluster_id: dominant_surface_id}
                         (returned by map_clusters_to_surfaces() in surface_clustering.py)
    method_name        : str                — label for plot titles
    """
    gt   = np.asarray(gt_labels)
    pred = np.asarray(pred_labels)

    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)
    print(f"\n{'─'*55}")
    print(f"  {method_name} diagnostics")
    print(f"  ARI = {ari:.4f}   NMI = {nmi:.4f}")
    print(f"  (1.0 = perfect alignment with ground truth)")
    print(f"{'─'*55}")

    print("\n[1/6] t-SNE comparison ...")
    plot_tsne_comparison(embeddings, gt, pred, cluster_to_surface, method_name)

    print("\n[2/6] Confusion matrix ...")
    plot_confusion(gt, pred, cluster_to_surface, method_name)

    print("\n[3/6] Cosine similarity heatmap ...")
    plot_cosine_heatmap(embeddings, gt)

    print("\n[4/6] Intra vs inter distances ...")
    plot_intra_inter_distances(embeddings, gt)

    print("\n[5/6] Per-surface silhouette ...")
    plot_per_surface_silhouette(embeddings, gt)

    print("\n[6/6] Cluster purity ...")
    plot_cluster_purity(gt, pred, cluster_to_surface, method_name)

    print("\nAll diagnostic plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# How to use from surface_clustering.py
# ─────────────────────────────────────────────────────────────────────────────
# After your main() has run, add at the bottom:
#
#   from cluster_diagnostics import run_all_diagnostics
#
#   run_all_diagnostics(
#       embeddings         = embeddings,             # from get_embeddings()
#       gt_labels          = acc_labels,             # true surface IDs
#       pred_labels        = pred_labels['kmeans'],  # or 'agg' / 'dbscan'
#       cluster_to_surface = cluster_surface_maps['kmeans'],
#       method_name        = 'K-Means',
#   )
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Quick smoke-test with synthetic data
    np.random.seed(42)
    N, D, K = 600, 32, 4
    emb = np.vstack([
        normalize(np.random.randn(N // K, D) + np.random.randn(1, D) * 3, norm='l2')
        for _ in range(K)
    ])
    gt  = np.repeat(np.arange(1, K + 1), N // K)
    # Simulate near-perfect clustering
    pred = gt.copy()
    pred[0] = 2   # introduce one mistake so confusion matrix is non-trivial
    c2s = {i: i + 1 for i in range(K)}

    run_all_diagnostics(emb, gt, pred, c2s, method_name='K-Means (test)')

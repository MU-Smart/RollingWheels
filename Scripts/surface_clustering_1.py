"""
Road Surface Classification — Vibration Embedding & Clustering
==============================================================
Run this script directly. It will:
  1. Load & window the raw CSV data
  2. Normalise
  3. Train the SupCon autoencoder (fixed contrastive loss)
  4. Extract embeddings
  5. Post-process (L2 norm + PCA)
  6. Cluster (K-Means, Agglomerative, DBSCAN)
  7. Evaluate and print comparison table
  8. Visualise with t-SNE (matplotlib 2-D + plotly 3-D)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
import glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)

import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# 2. Config  ← edit paths / hyper-parameters here
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    'data_dir'              : Path('../Datasets/Processed_Data/Labeled_Data_Without_GPS'),
    'window_size'           : 1024,
    'overlap'               : 0.5,
    'pca_variance_threshold': 0.95,

    # Model
    'embedding_dim' : 64,

    # Training
    'epochs'         : 150,
    'batch_size'     : 256,    # more negatives per step is critical
    'lr'             : 3e-4,
    'recon_w'        : 1.0,
    'contrast_w'     : 2.0,    # raise to 3-4 if clusters still overlap
    'temperature'    : 0.07,   # lower = sharper (0.05–0.1 range)
    'hard_neg_ratio' : 0.5,    # fraction of hardest negatives to mine per anchor
    'warmup_epochs'  : 15,     # reconstruction-only warm-up before contrast kicks in
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Data loading & windowing
# ─────────────────────────────────────────────────────────────────────────────
def extract_surface_type_id(path):
    match = re.search(r'SurfaceTypeID_(\d+)', path)
    return int(match.group(1)) if match else None


def load_files(config):
    data_dir   = config['data_dir']
    file_paths = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    files_df = pd.DataFrame({
        "full_path": file_paths,
        "filename" : [os.path.basename(p) for p in file_paths],
    })
    files_df["surface_id"] = files_df["full_path"].apply(extract_surface_type_id)
    files_df["device"]     = files_df["filename"].apply(lambda x: x.split('_')[3])
    files_df = files_df[files_df["device"] == "SamsungGalaxyJ7"].reset_index(drop=True)

    print(files_df["surface_id"].value_counts())
    return files_df


def extract_windows(files_df, config):
    WINDOW_SIZE = config['window_size']
    STEP_SIZE   = int(WINDOW_SIZE * (1 - config['overlap']))

    acc_windows,  acc_labels  = [], []
    gyro_windows, gyro_labels = [], []

    for _, row in tqdm(files_df.iterrows(), total=len(files_df), desc="Windowing"):
        file_path  = row['full_path']
        surface_id = row['surface_id']

        data_df = pd.read_csv(file_path)

        # Pad if shorter than one window
        if len(data_df) < WINDOW_SIZE:
            pad_size = WINDOW_SIZE - len(data_df)
            pad_df   = pd.concat([data_df] * (pad_size // len(data_df) + 1)).iloc[:pad_size]
            data_df  = pd.concat([data_df, pad_df], ignore_index=True)

        # Pad last incomplete window
        remainder = len(data_df) % STEP_SIZE
        if remainder != 0:
            pad_size = WINDOW_SIZE - remainder
            pad_df   = data_df.iloc[-pad_size:].copy()
            data_df  = pd.concat([data_df, pad_df], ignore_index=True)

        for start in range(0, len(data_df) - WINDOW_SIZE + 1, STEP_SIZE):
            window = data_df.iloc[start:start + WINDOW_SIZE]
            xyz    = window[['valueX', 'valueY', 'valueZ']].values  # (W, 3)

            if 'accelerometer' in file_path.lower():
                acc_windows.append(xyz)
                acc_labels.append(surface_id)
            elif 'gyroscope' in file_path.lower():
                gyro_windows.append(xyz)
                gyro_labels.append(surface_id)

    print(f"Accelerometer windows: {len(acc_windows)} | Gyroscope windows: {len(gyro_windows)}")
    return (np.array(acc_windows, dtype=np.float32),  np.array(acc_labels),
            np.array(gyro_windows, dtype=np.float32), np.array(gyro_labels))


def normalise_windows(windows):
    mean = windows.mean(axis=(0, 1), keepdims=True)
    std  = windows.std(axis=(0, 1),  keepdims=True)
    return (windows - mean) / (std + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model
# ─────────────────────────────────────────────────────────────────────────────
class ConvAutoencoder(nn.Module):
    def __init__(self, window_size, embedding_dim=64):
        super().__init__()
        self.window_size  = window_size
        self.reduced_size = window_size // 8

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32),  nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.flatten  = nn.Flatten()
        self.fc_embed = nn.Linear(128 * self.reduced_size, embedding_dim)

        # NOTE: No separate projection head.
        # We apply contrast loss DIRECTLY on the L2-normalised bottleneck so
        # gradients flow straight into the encoder without attenuation.

        self.decoder_fc = nn.Linear(embedding_dim, 128 * self.reduced_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, 3, 5, stride=2, padding=2, output_padding=1),
        )

    def encode(self, x):
        """L2-normalised bottleneck — used for both training loss and clustering."""
        x = x.permute(0, 2, 1)
        h = self.encoder(x)
        z = self.fc_embed(self.flatten(h))
        return F.normalize(z, p=2, dim=-1)   # unit sphere → cosine distance

    def decode(self, z):
        h   = self.decoder_fc(z).view(z.size(0), 128, self.reduced_size)
        out = self.decoder(h)[:, :, :self.window_size]
        return out.permute(0, 2, 1)

    def forward(self, x):
        z   = self.encode(x)
        rec = self.decode(z)
        return rec, z


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hard-negative SupCon loss
#
# Why the vanilla SupCon stalls at log(B) ≈ 2.77:
#   Once embeddings are roughly uniform on the sphere the loss reaches a
#   flat plateau — every negative contributes equally and gradients cancel.
#   Hard-negative mining focuses on the *most confusing* negatives so the
#   gradient signal stays alive throughout training.
# ─────────────────────────────────────────────────────────────────────────────
class HardNegSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, hard_neg_ratio=0.5):
        """
        temperature     : lower → sharper decision boundary (0.05–0.15)
        hard_neg_ratio  : fraction of hardest negatives to keep per anchor (0.3–0.7)
        """
        super().__init__()
        self.temperature     = temperature
        self.hard_neg_ratio  = hard_neg_ratio

    def forward(self, z, labels):
        """
        z      : (B, D)  already L2-normalised
        labels : (B,)    integer class ids
        """
        B      = z.size(0)
        device = z.device

        # ── Similarity & masks ────────────────────────────────────────────────
        sim       = z @ z.T                                              # (B,B) in [-1,1]
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        labels_2d = labels.unsqueeze(1)
        pos_mask  = (labels_2d == labels_2d.T) & ~mask_self             # same class, not self
        neg_mask  = ~pos_mask & ~mask_self                               # different class

        # ── Hard negative selection ───────────────────────────────────────────
        # Keep only the top-k HIGHEST similarity negatives (hardest to push apart)
        k = max(1, int(neg_mask.sum(dim=1).float().mean().item() * self.hard_neg_ratio))
        neg_sim = sim.masked_fill(~neg_mask, -1e9)
        topk_neg_vals, _ = neg_sim.topk(k, dim=1)                       # (B, k)
        hard_neg_threshold = topk_neg_vals[:, -1:].detach()             # (B, 1)
        hard_neg_mask = neg_mask & (sim >= hard_neg_threshold)          # (B, B)

        # ── InfoNCE over positives vs hard negatives ──────────────────────────
        active_mask = pos_mask | hard_neg_mask                          # denominator set
        sim_scaled  = sim / self.temperature

        # Numerical stability: subtract per-row max
        sim_scaled  = sim_scaled - sim_scaled.masked_fill(~active_mask & ~pos_mask, -1e9) \
                                              .max(dim=1, keepdim=True).values.detach()

        exp_sim   = torch.exp(sim_scaled).masked_fill(~active_mask, 0.0)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        n_pos = pos_mask.sum(dim=1).float().clamp(min=1)
        loss  = -((sim_scaled - log_denom) * pos_mask.float()).sum(dim=1) / n_pos

        # Only average over anchors that have at least one positive
        has_pos = pos_mask.any(dim=1)
        return loss[has_pos].mean() if has_pos.any() else loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training  (two-phase warm-up then joint)
# ─────────────────────────────────────────────────────────────────────────────
def train(model, windows_norm, labels, config, device):
    X = torch.tensor(windows_norm, dtype=torch.float32)
    Y = torch.tensor(labels,       dtype=torch.long)

    loader = DataLoader(TensorDataset(X, Y),
                        batch_size=config['batch_size'], shuffle=True, drop_last=True)

    criterion = HardNegSupConLoss(
        temperature    = config['temperature'],
        hard_neg_ratio = config.get('hard_neg_ratio', 0.5),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Phase boundary: first N epochs reconstruction-only to seed good representations,
    # then add contrastive loss so it has something meaningful to separate.
    warmup_epochs = config.get('warmup_epochs', 10)
    model.to(device)

    for epoch in range(config['epochs']):
        model.train()
        tot = rec_sum = con_sum = 0.0
        phase = "warm-up" if epoch < warmup_epochs else "joint  "

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            rec, z = model(xb)
            l_recon = F.mse_loss(rec, xb)

            if epoch < warmup_epochs:
                loss       = l_recon
                l_contrast = torch.tensor(0.0)
            else:
                l_contrast = criterion(z, yb)
                loss       = config['recon_w'] * l_recon + config['contrast_w'] * l_contrast

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot     += loss.item()
            rec_sum += l_recon.item()
            con_sum += l_contrast.item()

        scheduler.step()
        n = len(loader)
        print(f"Epoch {epoch+1:3d}/{config['epochs']} [{phase}] | "
              f"Total {tot/n:.4f} | Recon {rec_sum/n:.4f} | Contrast {con_sum/n:.4f}")

    return model


@torch.no_grad()
def get_embeddings(model, windows_norm, device, batch_size=256, l2_normalise=True):
    """Returns embeddings from the bottleneck."""
    model.eval()
    X, out = torch.tensor(windows_norm, dtype=torch.float32), []
    for i in range(0, len(X), batch_size):
        z = model.encode(X[i:i+batch_size].to(device))
        # encode() already L2-normalises; flag kept for API compatibility
        if not l2_normalise:
            pass  # raw already returned by encode when flag=False would need separate path
        out.append(z.cpu())
    return torch.cat(out).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Post-processing: L2 norm + PCA
# ─────────────────────────────────────────────────────────────────────────────
def postprocess(embeddings, variance_threshold=0.95):
    X_norm = normalize(embeddings, norm='l2', axis=1)
    pca    = PCA(n_components=variance_threshold, svd_solver='full')
    X_pca  = pca.fit_transform(X_norm)
    print(f"  Dims: {embeddings.shape[1]} → {X_pca.shape[1]} "
          f"(explained variance: {pca.explained_variance_ratio_.sum():.4f})")
    return X_norm, X_pca, pca


# ─────────────────────────────────────────────────────────────────────────────
# 8a. Auto-select optimal K via silhouette score sweep
# ─────────────────────────────────────────────────────────────────────────────
def auto_select_k(emb_pca, k_min=2, k_max=12, plot=True):
    """
    Sweeps K-Means over [k_min, k_max] and picks the K with the highest
    average silhouette score. Saves elbow curve to auto_k_selection.png.
    """
    print(f"  Sweeping K from {k_min} to {k_max} ...")
    scores = {}
    for k in range(k_min, k_max + 1):
        km  = KMeans(n_clusters=k, random_state=42, n_init=20)
        lbl = km.fit_predict(emb_pca)
        scores[k] = silhouette_score(emb_pca, lbl)
        print(f"    K={k:2d}  silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"  Best K = {best_k}  (silhouette={scores[best_k]:.4f})")

    if plot:
        ks, sil = list(scores.keys()), list(scores.values())
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ks, sil, marker='o', linewidth=2, color='#457b9d')
        ax.axvline(best_k, color='#e63946', linestyle='--', label=f'Best K={best_k}')
        ax.set_xlabel('Number of clusters K')
        ax.set_ylabel('Silhouette score')
        ax.set_title('Auto K selection — silhouette sweep')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('auto_k_selection.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("  Saved auto_k_selection.png")

    return best_k, scores


# ─────────────────────────────────────────────────────────────────────────────
# 8b. Cluster -> surface label mapping
# ─────────────────────────────────────────────────────────────────────────────
SURFACE_NAMES = {
    # Map your actual SurfaceTypeID integers to readable names.
    # Unknown IDs fall back to "Surface <id>".
    1: 'Asphalt',
    2: 'Concrete',
    3: 'Cobblestone',
    4: 'Gravel',
    5: 'Grass',
    6: 'Dirt',
    7: 'Sand',
    8: 'Tiles',
}

def surface_name(sid):
    return SURFACE_NAMES.get(int(sid), f'Surface {sid}')

def map_clusters_to_surfaces(pred_labels, gt_labels):
    """For each cluster, majority-vote the dominant ground-truth surface."""
    pred, gt = np.asarray(pred_labels), np.asarray(gt_labels)
    cluster_to_surface = {}
    for c in sorted(set(pred)):
        if c == -1:
            cluster_to_surface[c] = -1
            continue
        dominant = pd.Series(gt[pred == c]).mode()[0]
        cluster_to_surface[c] = dominant
    return cluster_to_surface

def cluster_legend_label(cluster_id, cluster_to_surface):
    if cluster_id == -1:
        return 'Noise'
    sid  = cluster_to_surface.get(cluster_id, '?')
    return f'{surface_name(sid)} [c{cluster_id}]'


# ─────────────────────────────────────────────────────────────────────────────
# 8c. Run all clusterers with the auto-selected K
# ─────────────────────────────────────────────────────────────────────────────
def run_clustering(emb_norm, emb_pca, gt_labels, k_min=2, k_max=12):
    """Auto-selects K, runs KMeans / Agglomerative / DBSCAN, maps clusters to surfaces."""
    best_k, _ = auto_select_k(emb_pca, k_min=k_min, k_max=k_max)

    pred_labels, cluster_surface_maps = {}, {}

    # K-Means
    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    pred_labels['kmeans'] = km.fit_predict(emb_pca)
    cluster_surface_maps['kmeans'] = map_clusters_to_surfaces(pred_labels['kmeans'], gt_labels)

    # Agglomerative (cosine)
    agg = AgglomerativeClustering(n_clusters=best_k, metric='cosine', linkage='average')
    pred_labels['agg'] = agg.fit_predict(emb_norm)
    cluster_surface_maps['agg'] = map_clusters_to_surfaces(pred_labels['agg'], gt_labels)

    # DBSCAN with auto eps (90th-pct of 10-NN distances)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10, metric='euclidean').fit(emb_pca)
    dists, _ = nn.kneighbors(emb_pca)
    eps_auto = float(np.percentile(dists[:, -1], 90))
    db = DBSCAN(eps=eps_auto, min_samples=10, metric='euclidean')
    pred_labels['dbscan'] = db.fit_predict(emb_pca)
    cluster_surface_maps['dbscan'] = map_clusters_to_surfaces(pred_labels['dbscan'], gt_labels)
    n_db    = len(set(pred_labels['dbscan'])) - (1 if -1 in pred_labels['dbscan'] else 0)
    n_noise = list(pred_labels['dbscan']).count(-1)
    print(f"  DBSCAN (eps={eps_auto:.3f}): {n_db} clusters, {n_noise} noise points")

    print(f"\n  Cluster to Surface mapping (K-Means):")
    for c, sid in sorted(cluster_surface_maps['kmeans'].items()):
        print(f"    Cluster {c:2d} -> {surface_name(sid)} (id={sid})")

    return pred_labels, cluster_surface_maps, best_k


# ─────────────────────────────────────────────────────────────────────────────
# 9. Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(embeddings, pred_labels, gt_labels=None, metric='euclidean'):
    pred_labels = np.asarray(pred_labels)
    valid       = pred_labels != -1
    lv, ev      = pred_labels[valid], embeddings[valid]

    if len(np.unique(lv)) < 2:
        return {'Silhouette': np.nan, 'Davies-Bouldin': np.nan,
                'Calinski-Harabasz': np.nan, 'ARI': np.nan, 'NMI': np.nan,
                'Quality': 'Only 1 cluster'}

    out = {
        'Silhouette'        : silhouette_score(ev, lv, metric=metric),
        'Davies-Bouldin'    : davies_bouldin_score(ev, lv),
        'Calinski-Harabasz' : calinski_harabasz_score(ev, lv),
    }
    if gt_labels is not None:
        gv = np.asarray(gt_labels)[valid]
        out['ARI'] = adjusted_rand_score(gv, lv)
        out['NMI'] = normalized_mutual_info_score(gv, lv)

    s = out['Silhouette']
    out['Quality'] = ('Excellent' if s > 0.6 else 'Good' if s > 0.4
                      else 'Fair' if s > 0.2 else 'Poor')
    return out


def print_comparison(emb_norm, emb_pca, pred_labels, gt_labels):
    rows = {
        'KMeans'       : evaluate(emb_pca,  pred_labels['kmeans'], gt_labels, 'euclidean'),
        'Agglomerative': evaluate(emb_norm, pred_labels['agg'],    gt_labels, 'cosine'),
        'DBSCAN'       : evaluate(emb_pca,  pred_labels['dbscan'],  gt_labels, 'euclidean'),
    }
    df = pd.DataFrame(rows).T
    print("\n── Clustering Comparison ──────────────────────────────────────")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. Visualisation  (legends show surface names, not raw cluster IDs)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = ['#e63946','#457b9d','#2a9d8f','#e9c46a','#f4a261',
          '#8338ec','#06d6a0','#fb8500','#3a86ff','#ff006e']


def _scatter_with_surface_legend(ax, coords_2d, pred_lbls, c2s, title):
    """2-D scatter where each legend entry shows the dominant surface name."""
    unique = sorted(set(pred_lbls))
    for i, lbl in enumerate(unique):
        mask  = pred_lbls == lbl
        color = '#aaaaaa' if lbl == -1 else COLORS[i % len(COLORS)]
        label = cluster_legend_label(lbl, c2s)
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=color, alpha=0.65, s=10, label=label)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=7, markerscale=1.5, framealpha=0.7,
              loc='best', title='Cluster → Surface')
    ax.grid(alpha=0.2)


def plot_tsne_2d(tsne_coords, pred_labels, cluster_surface_maps, gt_labels,
                 title_prefix="Accelerometer"):
    """2x2 panel: KMeans / Agglomerative / DBSCAN / Ground Truth, with surface legends."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f't-SNE 2D — {title_prefix}', fontsize=14)

    # Ground truth panel uses surface IDs as-is
    gt_c2s = {int(sid): int(sid) for sid in np.unique(gt_labels)}

    panels = [
        (axes[0,0], pred_labels['kmeans'], cluster_surface_maps['kmeans'], 'K-Means'),
        (axes[0,1], pred_labels['agg'],    cluster_surface_maps['agg'],    'Agglomerative'),
        (axes[1,0], pred_labels['dbscan'], cluster_surface_maps['dbscan'], 'DBSCAN'),
        (axes[1,1], gt_labels,             gt_c2s,                         'Ground Truth'),
    ]
    for ax, lbls, c2s, title in panels:
        _scatter_with_surface_legend(ax, tsne_coords, np.asarray(lbls), c2s, title)

    plt.tight_layout()
    plt.savefig('tsne_2d.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved tsne_2d.png")


def make_3d_scatter(coords, pred_lbls, c2s, title):
    """3-D Plotly scatter with surface-labelled legend entries."""
    unique_labels = sorted(set(pred_lbls))
    traces = []
    for i, lbl in enumerate(unique_labels):
        mask  = np.asarray(pred_lbls) == lbl
        color = '#aaaaaa' if lbl == -1 else COLORS[i % len(COLORS)]
        name  = cluster_legend_label(lbl, c2s)
        traces.append(go.Scatter3d(
            x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
            mode='markers', name=name,
            marker=dict(size=4, color=color, opacity=0.75),
        ))
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        scene=dict(
            xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3",
            bgcolor='rgb(15,15,25)',
            xaxis=dict(backgroundcolor='rgb(15,15,25)', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(backgroundcolor='rgb(15,15,25)', gridcolor='rgba(255,255,255,0.1)'),
            zaxis=dict(backgroundcolor='rgb(15,15,25)', gridcolor='rgba(255,255,255,0.1)'),
        ),
        paper_bgcolor='rgb(15,15,25)', font=dict(color='white'),
        legend=dict(bgcolor='rgba(255,255,255,0.07)', borderwidth=1, title='Cluster → Surface'),
        margin=dict(l=0, r=0, t=40, b=0), height=580,
    )
    return fig


def plot_tsne_3d(tsne_coords, pred_labels, cluster_surface_maps, gt_labels,
                 title_prefix="Accelerometer"):
    gt_c2s = {int(sid): int(sid) for sid in np.unique(gt_labels)}
    configs = [
        (pred_labels['kmeans'], cluster_surface_maps['kmeans'], f"K-Means — {title_prefix}"),
        (pred_labels['agg'],    cluster_surface_maps['agg'],    f"Agglomerative — {title_prefix}"),
        (pred_labels['dbscan'], cluster_surface_maps['dbscan'], f"DBSCAN — {title_prefix}"),
        (gt_labels,             gt_c2s,                         f"Ground Truth — {title_prefix}"),
    ]
    for lbls, c2s, title in configs:
        make_3d_scatter(tsne_coords, lbls, c2s, title).show()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/6] Loading files...")
    files_df = load_files(CONFIG)

    print("\n[2/6] Windowing...")
    acc_windows, acc_labels, gyro_windows, gyro_labels = extract_windows(files_df, CONFIG)

    print("\n[3/6] Normalising...")
    acc_norm = normalise_windows(acc_windows)
    print(f"  Classes: {np.unique(acc_labels)}  ({len(np.unique(acc_labels))} total)")

    # ── Train autoencoder ─────────────────────────────────────────────────────
    print("\n[4/6] Training autoencoder...")
    model = ConvAutoencoder(window_size=CONFIG['window_size'],
                            embedding_dim=CONFIG['embedding_dim']).to(device)
    model = train(model, acc_norm, acc_labels, CONFIG, device)

    # ── Extract embeddings ────────────────────────────────────────────────────
    print("\n[5/6] Extracting & post-processing embeddings...")
    embeddings = get_embeddings(model, acc_norm, device)
    print(f"  Raw embeddings: {embeddings.shape}")

    acc_emb_norm, acc_emb_pca, _ = postprocess(embeddings, CONFIG['pca_variance_threshold'])

    # ── Cluster (K auto-selected by silhouette sweep) ─────────────────────────
    print("\n[6/6] Clustering...")
    n_unique = len(np.unique(acc_labels))
    pred_labels, cluster_surface_maps, best_k = run_clustering(
        acc_emb_norm, acc_emb_pca, acc_labels,
        k_min=max(2, n_unique - 2),   # search around the true number of classes
        k_max=n_unique + 3,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print_comparison(acc_emb_norm, acc_emb_pca, pred_labels, acc_labels)

    # ── Visualise ──────────────────────────────────────────────────────────────
    print("\nRunning t-SNE (this may take a moment)...")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)

    acc_tsne_2d = tsne_2d.fit_transform(acc_emb_pca)
    acc_tsne_3d = tsne_3d.fit_transform(acc_emb_pca)

    plot_tsne_2d(acc_tsne_2d, pred_labels, cluster_surface_maps, acc_labels,
                 title_prefix="Accelerometer")
    plot_tsne_3d(acc_tsne_3d, pred_labels, cluster_surface_maps, acc_labels,
                 title_prefix="Accelerometer")

    print("\nDone!")


if __name__ == "__main__":
    main()

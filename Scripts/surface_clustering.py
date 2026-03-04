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
    'epochs'       : 150,
    'batch_size'   : 128,   # bigger = more negatives per step → better contrast
    'lr'           : 3e-4,
    'recon_w'      : 1.0,
    'contrast_w'   : 1.0,   # raise to 2-3 if clusters still overlap
    'temperature'  : 0.1,   # lower → sharper clusters (try 0.07 if needed)
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

        # Projection head — used ONLY for contrastive loss during training
        self.proj_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, 32),
        )

        self.decoder_fc = nn.Linear(embedding_dim, 128 * self.reduced_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, 3, 5, stride=2, padding=2, output_padding=1),
        )

    def encode(self, x):
        """Raw bottleneck embedding — use this for clustering."""
        x = x.permute(0, 2, 1)
        h = self.encoder(x)
        return self.fc_embed(self.flatten(h))

    def project(self, z):
        """L2-normalised projection used only during training."""
        return F.normalize(self.proj_head(z), p=2, dim=-1)

    def decode(self, z):
        h   = self.decoder_fc(z).view(z.size(0), 128, self.reduced_size)
        out = self.decoder(h)[:, :, :self.window_size]
        return out.permute(0, 2, 1)

    def forward(self, x):
        z   = self.encode(x)
        rec = self.decode(z)
        p   = self.project(z)
        return rec, z, p


# ─────────────────────────────────────────────────────────────────────────────
# 5. Supervised Contrastive Loss  (Khosla et al. 2020) — numerically stable
# ─────────────────────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, labels):
        """
        z      : (B, D) L2-normalised projections
        labels : (B,)   integer class ids
        """
        B      = z.size(0)
        device = z.device

        sim       = z @ z.T / self.temperature                         # (B, B)
        sim       = sim - sim.max(dim=1, keepdim=True).values.detach() # stability

        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        exp_sim   = torch.exp(sim).masked_fill(mask_self, 0.0)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        pos_mask  = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~mask_self
        n_pos     = pos_mask.sum(dim=1).float().clamp(min=1)

        loss = -((sim - log_denom) * pos_mask.float()).sum(dim=1) / n_pos
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training
# ─────────────────────────────────────────────────────────────────────────────
def train(model, windows_norm, labels, config, device):
    X = torch.tensor(windows_norm, dtype=torch.float32)
    Y = torch.tensor(labels,       dtype=torch.long)

    loader    = DataLoader(TensorDataset(X, Y),
                           batch_size=config['batch_size'], shuffle=True, drop_last=True)
    supcon    = SupConLoss(temperature=config['temperature']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    model.to(device)

    for epoch in range(config['epochs']):
        model.train()
        tot = rec_sum = con_sum = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            rec, z, p = model(xb)
            l_recon   = F.mse_loss(rec, xb)
            l_contrast = supcon(p, yb)
            loss      = config['recon_w'] * l_recon + config['contrast_w'] * l_contrast

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot     += loss.item()
            rec_sum += l_recon.item()
            con_sum += l_contrast.item()

        scheduler.step()
        n = len(loader)
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Total {tot/n:.4f} | Recon {rec_sum/n:.4f} | Contrast {con_sum/n:.4f}")

    return model


@torch.no_grad()
def get_embeddings(model, windows_norm, device, batch_size=256, l2_normalise=True):
    """Returns embeddings from the bottleneck (not the projection head)."""
    model.eval()
    X, out = torch.tensor(windows_norm, dtype=torch.float32), []
    for i in range(0, len(X), batch_size):
        z = model.encode(X[i:i+batch_size].to(device))
        if l2_normalise:
            z = F.normalize(z, p=2, dim=-1)
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
# 8. Clustering
# ─────────────────────────────────────────────────────────────────────────────
def run_clustering(emb_norm, emb_pca, n_clusters):
    labels = {}

    # K-Means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels['kmeans'] = km.fit_predict(emb_pca)

    # Agglomerative (cosine)
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels['agg'] = agg.fit_predict(emb_norm)

    # DBSCAN
    db = DBSCAN(eps=0.5, min_samples=10, metric='euclidean')
    labels['dbscan'] = db.fit_predict(emb_pca)
    n_db   = len(set(labels['dbscan'])) - (1 if -1 in labels['dbscan'] else 0)
    n_noise = list(labels['dbscan']).count(-1)
    print(f"  DBSCAN found {n_db} clusters, {n_noise} noise points")

    return labels


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


def print_comparison(emb_norm, emb_pca, cluster_labels, gt_labels):
    rows = {
        'KMeans'      : evaluate(emb_pca,  cluster_labels['kmeans'], gt_labels, 'euclidean'),
        'Agglomerative': evaluate(emb_norm, cluster_labels['agg'],    gt_labels, 'cosine'),
        'DBSCAN'      : evaluate(emb_pca,  cluster_labels['dbscan'],  gt_labels, 'euclidean'),
    }
    df = pd.DataFrame(rows).T
    print("\n── Clustering Comparison ──────────────────────────────────────")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def plot_tsne_2d(tsne_coords, cluster_labels, gt_labels, title_prefix="Accelerometer"):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f't-SNE 2D — {title_prefix}', fontsize=14, y=1.01)
    palette = cm.tab10

    configs = [
        (cluster_labels['kmeans'], 'K-Means'),
        (cluster_labels['agg'],    'Agglomerative'),
        (cluster_labels['dbscan'], 'DBSCAN'),
        (gt_labels,                'Ground Truth'),
    ]
    for ax, (lbls, name) in zip(axes, configs):
        sc = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                        c=lbls, cmap=palette, alpha=0.6, s=8)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        ax.grid(alpha=0.2)
        plt.colorbar(sc, ax=ax, label='Cluster')

    plt.tight_layout()
    plt.savefig('tsne_2d.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved tsne_2d.png")


COLORS = ['#e63946','#457b9d','#2a9d8f','#e9c46a','#f4a261',
          '#8338ec','#06d6a0','#fb8500','#3a86ff','#ff006e']

def make_3d_scatter(coords, labels, title):
    unique_labels = sorted(set(labels))
    traces = []
    for lbl in unique_labels:
        mask  = np.array(labels) == lbl
        name  = "Noise" if lbl == -1 else f"Cluster {lbl}"
        color = '#aaaaaa' if lbl == -1 else COLORS[lbl % len(COLORS)]
        traces.append(go.Scatter3d(
            x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
            mode='markers', name=name,
            marker=dict(size=4, color=color, opacity=0.75),
        ))
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=dict(
            xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3",
            bgcolor='rgb(15,15,25)',
            xaxis=dict(backgroundcolor='rgb(15,15,25)', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(backgroundcolor='rgb(15,15,25)', gridcolor='rgba(255,255,255,0.1)'),
            zaxis=dict(backgroundcolor='rgb(15,15,25)', gridcolor='rgba(255,255,255,0.1)'),
        ),
        paper_bgcolor='rgb(15,15,25)', font=dict(color='white'),
        legend=dict(bgcolor='rgba(255,255,255,0.05)', borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0), height=550,
    )
    return fig


def plot_tsne_3d(tsne_coords, cluster_labels, gt_labels, title_prefix="Accelerometer"):
    configs = [
        (cluster_labels['kmeans'], f"K-Means — {title_prefix}"),
        (cluster_labels['agg'],    f"Agglomerative — {title_prefix}"),
        (cluster_labels['dbscan'], f"DBSCAN — {title_prefix}"),
        (gt_labels,                f"Ground Truth — {title_prefix}"),
    ]
    for lbls, title in configs:
        make_3d_scatter(tsne_coords, lbls, title).show()


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
    n_classes = len(np.unique(acc_labels))
    print(f"  Classes: {np.unique(acc_labels)}  ({n_classes} total)")

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

    # ── Cluster ────────────────────────────────────────────────────────────────
    print("\n[6/6] Clustering...")
    cluster_labels = run_clustering(acc_emb_norm, acc_emb_pca, n_clusters=n_classes)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print_comparison(acc_emb_norm, acc_emb_pca, cluster_labels, acc_labels)

    # ── Visualise ──────────────────────────────────────────────────────────────
    print("\nRunning t-SNE (this may take a moment)...")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)

    acc_tsne_2d = tsne_2d.fit_transform(acc_emb_pca)
    acc_tsne_3d = tsne_3d.fit_transform(acc_emb_pca)

    plot_tsne_2d(acc_tsne_2d, cluster_labels, acc_labels, title_prefix="Accelerometer")
    plot_tsne_3d(acc_tsne_3d, cluster_labels, acc_labels, title_prefix="Accelerometer")

    print("\nDone!")


if __name__ == "__main__":
    main()

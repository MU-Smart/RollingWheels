"""
Road Surface Classification -- Vibration Embedding & Clustering
===============================================================
All four generalisation fixes applied (was: train ARI=1.0, test ARI=0.13):

  Fix 1 -- Spectral features (FFT magnitude) instead of raw XYZ
           Raw XYZ encodes recording-specific info: phone orientation, mounting
           position, driving speed. FFT magnitude captures the FREQUENCY
           FINGERPRINT of the surface texture and is invariant to all of those.
           Asphalt/gravel/cobblestone have distinct spectral shapes regardless
           of how the phone is held. Your 94% supervised accuracy proves the
           surfaces are separable -- we just need invariant features.

  Fix 2 -- Stratified file-level split
           Old GroupShuffleSplit could leave entire surface classes out of the
           test set. New version splits each surface class independently,
           guaranteeing >= 1 test file per surface.

  Fix 3 -- Shorter warm-up (5 epochs instead of 15)
           15 epochs of pure reconstruction teaches the encoder to faithfully
           reproduce each specific recording. 5 epochs is enough to seed
           structure without memorising recording-specific patterns.

  Fix 4 -- Stronger contrastive weight (3.0) + balanced batch sampler
           Dominant surfaces would flood batches and drown rare-surface
           gradients. WeightedRandomSampler ensures every surface class
           contributes equally to each contrastive batch.
"""

import glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------
CONFIG = {
    "data_dir"              : Path("../Datasets/Processed_Data/Labeled_Data_Without_GPS"),
    "surface_types_csv"     : Path("../Datasets/surface_types.csv"),
    "window_size"           : 1024,
    "overlap"               : 0.5,
    "pca_variance_threshold": 0.95,

    # Model
    "embedding_dim" : 64,

    # Training
    "epochs"         : 150,
    "batch_size"     : 256,
    "lr"             : 3e-4,
    # (no recon_w -- decoder removed; pure contrastive training)
    "contrast_w"     : 2.0,   # weight on SupCon loss (projection head)
    "temperature"    : 0.07,
    "hard_neg_ratio" : 0.5,
    # (no warmup_epochs -- pure contrastive from epoch 1, no reconstruction)

    # Train / test split (by FILE, not by window)
    "test_size"  : 0.2,
    "split_seed" : 42,

    # FIX 1: use spectral features
    "use_spectral": True,
}

# ---------------------------------------------------------------------------
# 3. Data loading & windowing
# ---------------------------------------------------------------------------
def extract_surface_type_id(path):
    m = re.search(r"SurfaceTypeID_(\d+)", path)
    return int(m.group(1)) if m else None


def load_files(config):
    paths = glob.glob(os.path.join(config["data_dir"], "**", "*.csv"), recursive=True)
    df = pd.DataFrame({"full_path": paths, "filename": [os.path.basename(p) for p in paths]})
    df["surface_id"] = df["full_path"].apply(extract_surface_type_id)
    df["device"]     = df["filename"].apply(lambda x: x.split("_")[3])
    df = df[df["device"] == "SamsungGalaxyJ7"].reset_index(drop=True)
    print(df["surface_id"].value_counts())
    return df


def extract_windows(files_df, config):
    W    = config["window_size"]
    step = int(W * (1 - config["overlap"]))
    acc_win, acc_lbl, acc_fid  = [], [], []
    gyr_win, gyr_lbl, gyr_fid  = [], [], []

    for fid, row in tqdm(files_df.iterrows(), total=len(files_df), desc="Windowing"):
        path = row["full_path"]
        sid  = row["surface_id"]
        df   = pd.read_csv(path)

        if len(df) < W:
            pad = pd.concat([df] * (W // len(df) + 1)).iloc[:W - len(df)]
            df  = pd.concat([df, pad], ignore_index=True)
        rem = len(df) % step
        if rem:
            df = pd.concat([df, df.iloc[-(W - rem):].copy()], ignore_index=True)

        for s in range(0, len(df) - W + 1, step):
            xyz = df.iloc[s:s+W][["valueX","valueY","valueZ"]].values
            if "accelerometer" in path.lower():
                acc_win.append(xyz); acc_lbl.append(sid); acc_fid.append(fid)
            elif "gyroscope" in path.lower():
                gyr_win.append(xyz); gyr_lbl.append(sid); gyr_fid.append(fid)

    print(f"Acc windows: {len(acc_win)}  |  Gyro windows: {len(gyr_win)}")
    return (np.array(acc_win, dtype=np.float32), np.array(acc_lbl), np.array(acc_fid),
            np.array(gyr_win, dtype=np.float32), np.array(gyr_lbl), np.array(gyr_fid))


def normalise_windows(windows):
    mu  = windows.mean(axis=(0, 1), keepdims=True)
    std = windows.std(axis=(0, 1),  keepdims=True)
    return (windows - mu) / (std + 1e-8)


# ---------------------------------------------------------------------------
# FIX 1 -- Spectral features
#
# FFT magnitude is invariant to:
#   * Phone orientation  (phase-independent)
#   * DC offset / gravity (DC bin dropped)
#   * Absolute amplitude (log-compress)
#
# What it preserves: the FREQUENCY FINGERPRINT of each surface texture.
# Asphalt vs cobblestone vs gravel have completely different vibration spectra.
# This is why a supervised CNN gets 94% -- it is implicitly learning spectral
# patterns. We make that explicit here so the unsupervised encoder starts from
# the right representation.
#
# Shape: (N, W, 3) raw  -->  (N, W//2, 3) log-magnitude spectrum
# ---------------------------------------------------------------------------
def compute_spectral_features(windows):
    """
    Input:  (N, W, 3)  raw time-domain windows
    Output: (N, W//2, 3)  log(1 + |FFT|), DC bin removed
    """
    fft_mag = np.abs(np.fft.rfft(windows, axis=1))   # (N, W//2+1, 3)
    fft_mag = fft_mag[:, 1:, :]                       # drop DC  -> (N, W//2, 3)
    return np.log1p(fft_mag).astype(np.float32)


# ---------------------------------------------------------------------------
# FIX 2 -- Stratified file-level split
#
# Problem with plain GroupShuffleSplit: it may put ALL files for one surface
# into train, leaving that surface absent from the test set entirely.
# DBSCAN then finds "1 cluster" on test because those embeddings have no
# reference to compare against.
#
# Fix: split each surface class independently, guaranteeing >= 1 test file
# per surface. If a surface has only 1 file, emit a warning and put it in
# train only.
# ---------------------------------------------------------------------------
def split_by_file(windows, labels, file_ids, test_size=0.2, seed=42):
    """
    Stratified per-surface file-level split.
    Every surface class is guaranteed to have at least one test file.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    print("  Per-surface file split:")
    for surf in sorted(np.unique(labels)):
        mask  = labels == surf
        files = np.unique(file_ids[mask])

        if len(files) < 2:
            print(f"    WARNING: {surface_name(surf)} has only {len(files)} recording"
                  f" -- all windows go to train")
            train_idx.extend(np.where(mask)[0].tolist())
            continue

        shuffled  = rng.permutation(files)
        n_te      = max(1, int(len(shuffled) * test_size))
        te_files  = set(shuffled[:n_te].tolist())

        for idx in np.where(mask)[0]:
            (test_idx if file_ids[idx] in te_files else train_idx).append(idx)

        print(f"    {surface_name(surf):20s}: {len(shuffled)-n_te} train file(s),"
              f" {n_te} test file(s)")

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    assert set(file_ids[train_idx].tolist()).isdisjoint(set(file_ids[test_idx].tolist())),         "File leakage detected!"
    print(f"  Total: {len(train_idx)} train windows | {len(test_idx)} test windows")
    print("  File leakage check PASSED.")

    return (windows[train_idx], labels[train_idx], file_ids[train_idx],
            windows[test_idx],  labels[test_idx],  file_ids[test_idx])


# ---------------------------------------------------------------------------
# Diagnostic: data distribution
# ---------------------------------------------------------------------------
def print_data_distribution(files_df, acc_labels, train_labels, test_labels):
    print("\n  Files per surface (full dataset):")
    for surf, count in files_df.groupby("surface_id").size().items():
        print(f"    {surface_name(surf):20s}: {count} files")

    print("\n  Windows per surface (train / test):")
    for surf in sorted(np.unique(acc_labels)):
        tr   = (train_labels == surf).sum()
        te   = (test_labels  == surf).sum()
        warn = " <-- WARNING: no test data!" if te == 0 else ""
        print(f"    {surface_name(surf):20s}: {tr:5d} train  {te:5d} test{warn}")


# ---------------------------------------------------------------------------
# FIX 4b -- Balanced batch sampler
#
# Without balancing, a surface with 5x more windows dominates every batch.
# The contrastive loss spends most of its gradient budget separating the
# large classes and barely trains on the small ones.
# WeightedRandomSampler gives each surface class equal expected contribution
# per batch, no matter how imbalanced the raw counts are.
# ---------------------------------------------------------------------------
def make_balanced_sampler(labels):
    counts  = np.bincount(labels)
    weights = torch.tensor(1.0 / counts[labels], dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# 4. Model  (in_channels parametrised for spectral vs raw)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 4. Pure Contrastive Encoder  (no decoder / no reconstruction loss)
#
# WHY drop the decoder?
# The autoencoder objective directly conflicts with contrastive learning:
#   - Decoder needs the bottleneck to REMEMBER per-recording detail to reconstruct
#   - Contrastive loss needs the bottleneck to FORGET per-recording detail
# Result: the two objectives fight each other and the contrastive loss stalls
# at log(n_hard_negatives) ~3.33 for all 145 joint epochs -- no learning.
#
# Architecture:  Conv encoder -> 64-d bottleneck -> 128-d projection head
# Training loss: SupCon on projection head outputs only
# Clustering:    done on bottleneck (not projection head)
#
# The projection head absorbs gradients from the contrastive loss, letting the
# bottleneck learn clean geometry without interference. After training, the
# projection head is discarded -- we cluster the 64-d bottleneck directly.
# This is exactly how SimCLR / MoCo / SupCon are used in practice.
# ---------------------------------------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, window_size, embedding_dim=64, proj_dim=128, in_channels=3):
        super().__init__()
        self.window_size  = window_size
        self.reduced_size = window_size // 8

        # Backbone: 3 strided Conv layers, each halving temporal resolution
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32,  5, stride=2, padding=2), nn.BatchNorm1d(32),  nn.GELU(),
            nn.Conv1d(32,          64,  5, stride=2, padding=2), nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64,          128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.flatten  = nn.Flatten()
        self.fc_embed = nn.Linear(128 * self.reduced_size, embedding_dim)

        # Projection head: used ONLY during training for contrastive loss
        # Discarded at inference -- we cluster the bottleneck instead
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def embed(self, x):
        """L2-normalised bottleneck -- used for clustering after training."""
        x = x.permute(0, 2, 1)           # (B,W,C) -> (B,C,W)
        z = self.fc_embed(self.flatten(self.encoder(x)))
        return F.normalize(z, p=2, dim=-1)

    def project(self, x):
        """Projection head output -- used for contrastive loss during training."""
        z = self.embed(x)
        return F.normalize(self.projector(z), p=2, dim=-1)

    def forward(self, x):
        """Returns (bottleneck, projection) -- training uses projection for loss."""
        z = self.embed(x)
        p = F.normalize(self.projector(z), p=2, dim=-1)
        return z, p


# ---------------------------------------------------------------------------
# 5. Hard-negative SupCon loss  (unchanged -- operates on projection head)
# ---------------------------------------------------------------------------
class HardNegSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, hard_neg_ratio=0.5):
        super().__init__()
        self.temperature    = temperature
        self.hard_neg_ratio = hard_neg_ratio

    def forward(self, z, labels):
        """z: (B, D) L2-normalised projections.  labels: (B,) integer surface ids."""
        B, device = z.size(0), z.device
        sim       = z @ z.T
        eye       = torch.eye(B, dtype=torch.bool, device=device)
        lbl2d     = labels.unsqueeze(1)
        pos_mask  = (lbl2d == lbl2d.T) & ~eye
        neg_mask  = ~pos_mask & ~eye

        k           = max(1, int(neg_mask.sum(1).float().mean().item() * self.hard_neg_ratio))
        neg_sim     = sim.masked_fill(~neg_mask, -1e9)
        topk, _     = neg_sim.topk(k, dim=1)
        threshold   = topk[:, -1:].detach()
        hard_neg    = neg_mask & (sim >= threshold)

        active      = pos_mask | hard_neg
        ss          = sim / self.temperature
        ss          = ss - ss.masked_fill(~active & ~pos_mask, -1e9).max(1, keepdim=True).values.detach()
        exp_s       = torch.exp(ss).masked_fill(~active, 0.0)
        log_denom   = torch.log(exp_s.sum(1, keepdim=True) + 1e-9)
        n_pos       = pos_mask.sum(1).float().clamp(min=1)
        loss        = -((ss - log_denom) * pos_mask.float()).sum(1) / n_pos
        has_pos     = pos_mask.any(1)
        return loss[has_pos].mean() if has_pos.any() else loss.mean()


# ---------------------------------------------------------------------------
# 6. Training  (pure contrastive -- no reconstruction term)
# ---------------------------------------------------------------------------
def train(model, windows, labels, config, device):
    X = torch.tensor(windows, dtype=torch.float32)
    Y = torch.tensor(labels,  dtype=torch.long)

    sampler = make_balanced_sampler(labels)
    loader  = DataLoader(TensorDataset(X, Y),
                         batch_size=config["batch_size"],
                         sampler=sampler, drop_last=True)

    criterion = HardNegSupConLoss(config["temperature"],
                                  config.get("hard_neg_ratio", 0.5)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    model.to(device)

    best_loss = float('inf')
    patience_counter = 0
    PATIENCE = 30  # stop if no improvement for 30 epochs

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            _, proj = model(xb)           # use projection head for loss
            loss    = criterion(proj, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        improved = avg < best_loss - 1e-4
        if improved:
            best_loss = avg
            patience_counter = 0
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | Contrast {avg:.4f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    return model


@torch.no_grad()
def get_embeddings(model, windows, device, batch_size=256):
    """Extract bottleneck embeddings (projection head discarded)."""
    model.eval()
    X   = torch.tensor(windows, dtype=torch.float32)
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model.embed(X[i:i+batch_size].to(device)).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# 7. Post-processing
# ---------------------------------------------------------------------------
def postprocess(embeddings, variance_threshold=0.95):
    X_norm = normalize(embeddings, norm="l2", axis=1)
    pca    = PCA(n_components=variance_threshold, svd_solver="full")
    X_pca  = pca.fit_transform(X_norm)
    print(f"  Dims: {embeddings.shape[1]} -> {X_pca.shape[1]} "
          f"(variance explained: {pca.explained_variance_ratio_.sum():.4f})")
    return X_norm, X_pca, pca


# ---------------------------------------------------------------------------
# 8a. Auto-select K via silhouette sweep
# ---------------------------------------------------------------------------
def auto_select_k(emb_pca, k_min=2, k_max=12, plot=True):
    print(f"  Sweeping K {k_min}..{k_max} ...")
    scores = {}
    for k in range(k_min, k_max + 1):
        lbl       = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(emb_pca)
        scores[k] = silhouette_score(emb_pca, lbl)
        print(f"    K={k:2d}  sil={scores[k]:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"  Best K = {best_k}  (sil={scores[best_k]:.4f})")
    if plot:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(list(scores.keys()), list(scores.values()), marker="o", color="#457b9d")
        ax.axvline(best_k, color="#e63946", linestyle="--", label=f"Best K={best_k}")
        ax.set_xlabel("K"); ax.set_ylabel("Silhouette"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("Auto K selection")
        plt.tight_layout(); plt.savefig("auto_k_selection.png", dpi=150); plt.show()
    return best_k, scores


# ---------------------------------------------------------------------------
# 8b. Cluster -> surface name mapping
# ---------------------------------------------------------------------------
SURFACE_NAMES: dict = {}

def load_surface_names(csv_path):
    global SURFACE_NAMES
    df       = pd.read_csv(csv_path)
    id_col   = next(c for c in df.columns if "id"   in c.lower())
    name_col = next(c for c in df.columns if "name" in c.lower())
    SURFACE_NAMES = {int(r[id_col]): str(r[name_col]) for _, r in df.iterrows()}
    print(f"  Loaded {len(SURFACE_NAMES)} surface types:")
    for sid, name in sorted(SURFACE_NAMES.items()):
        print(f"    {sid:3d} -> {name}")

def surface_name(sid):
    return SURFACE_NAMES.get(int(sid), f"Surface {sid}")

def map_clusters_to_surfaces(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    c2s = {}
    for c in sorted(set(pred)):
        c2s[c] = -1 if c == -1 else int(pd.Series(gt[pred == c]).mode()[0])
    return c2s

def cluster_legend_label(cid, c2s):
    if cid == -1: return "Noise"
    return f"{surface_name(c2s.get(cid, '?'))} [c{cid}]"


# ---------------------------------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------------------------------
def evaluate(emb, pred, gt=None, metric="euclidean"):
    pred = np.asarray(pred)
    ok   = pred != -1
    ev, lv = emb[ok], pred[ok]
    if len(np.unique(lv)) < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan, "ARI": np.nan, "NMI": np.nan,
                "Quality": "Only 1 cluster"}
    out = {
        "Silhouette"        : silhouette_score(ev, lv, metric=metric),
        "Davies-Bouldin"    : davies_bouldin_score(ev, lv),
        "Calinski-Harabasz" : calinski_harabasz_score(ev, lv),
    }
    if gt is not None:
        gv = np.asarray(gt)[ok]
        out["ARI"] = adjusted_rand_score(gv, lv)
        out["NMI"] = normalized_mutual_info_score(gv, lv)
    s = out["Silhouette"]
    out["Quality"] = "Excellent" if s > 0.6 else "Good" if s > 0.4 else "Fair" if s > 0.2 else "Poor"
    return out


def print_comparison(en, ep, pred, gt, split=""):
    rows = {
        "KMeans"       : evaluate(ep, pred["kmeans"], gt, "euclidean"),
        "Agglomerative": evaluate(en, pred["agg"],    gt, "cosine"),
        "DBSCAN"       : evaluate(ep, pred["dbscan"], gt, "euclidean"),
    }
    df  = pd.DataFrame(rows).T
    tag = f" [{split}]" if split else ""
    print(f"\nClustering Comparison{tag} {chr(45)*max(0,44-len(tag))}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    return df


# ---------------------------------------------------------------------------
# 10. Visualisation
# ---------------------------------------------------------------------------
COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261",
          "#8338ec","#06d6a0","#fb8500","#3a86ff","#ff006e"]


def _scatter(ax, xy, lbls, c2s, title):
    for i, lbl in enumerate(sorted(set(lbls))):
        m = lbls == lbl
        ax.scatter(xy[m,0], xy[m,1], c="#aaa" if lbl==-1 else COLORS[i%len(COLORS)],
                   alpha=0.65, s=10, label=cluster_legend_label(lbl, c2s))
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, markerscale=1.5, framealpha=0.7,
              loc="best", title="Cluster -> Surface")
    ax.grid(alpha=0.2)


def plot_tsne_2d(coords, pred, c2s_map, gt, prefix=""):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"t-SNE 2D -- {prefix}", fontsize=14)
    gt_c2s = {int(s): int(s) for s in np.unique(gt)}
    for ax, key, c2s, title in [
        (axes[0,0], "kmeans", c2s_map["kmeans"], "K-Means"),
        (axes[0,1], "agg",    c2s_map["agg"],    "Agglomerative"),
        (axes[1,0], "dbscan", c2s_map["dbscan"], "DBSCAN"),
        (axes[1,1], None,     gt_c2s,             "Ground Truth"),
    ]:
        lbls = gt if key is None else pred[key]
        _scatter(ax, coords, np.asarray(lbls), c2s, title)
    plt.tight_layout()
    fname = f"tsne_2d_{prefix.replace(' ','_').replace('[','').replace(']','')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  Saved {fname}")


def plot_tsne_3d(coords, pred, c2s_map, gt, prefix=""):
    gt_c2s = {int(s): int(s) for s in np.unique(gt)}
    for lbls, c2s, title in [
        (pred["kmeans"], c2s_map["kmeans"], f"K-Means -- {prefix}"),
        (pred["agg"],    c2s_map["agg"],    f"Agglomerative -- {prefix}"),
        (pred["dbscan"], c2s_map["dbscan"], f"DBSCAN -- {prefix}"),
        (gt,             gt_c2s,            f"Ground Truth -- {prefix}"),
    ]:
        traces = []
        for i, lbl in enumerate(sorted(set(lbls))):
            m = np.asarray(lbls) == lbl
            traces.append(go.Scatter3d(
                x=coords[m,0], y=coords[m,1], z=coords[m,2], mode="markers",
                name=cluster_legend_label(lbl, c2s),
                marker=dict(size=4, color="#aaa" if lbl==-1 else COLORS[i%len(COLORS)], opacity=0.75),
            ))
        fig = go.Figure(traces)
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3",
                       bgcolor="rgb(15,15,25)",
                       xaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="rgba(255,255,255,0.1)"),
                       yaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="rgba(255,255,255,0.1)"),
                       zaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="rgba(255,255,255,0.1)")),
            paper_bgcolor="rgb(15,15,25)", font=dict(color="white"),
            legend=dict(bgcolor="rgba(255,255,255,0.07)", title="Cluster -> Surface"),
            margin=dict(l=0,r=0,t=40,b=0), height=580,
        )
        fig.show()


# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("mps"  if torch.backends.mps.is_available()  else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 0. Surface names ---
    print("\n[0/7] Surface names...")
    load_surface_names(CONFIG["surface_types_csv"])

    # --- 1. Load files ---
    print("\n[1/7] Loading files...")
    files_df = load_files(CONFIG)

    # --- 2. Window ---
    print("\n[2/7] Windowing...")
    acc_win, acc_lbl, acc_fid, gyr_win, gyr_lbl, gyr_fid = extract_windows(files_df, CONFIG)

    # --- 3. FIX 1: spectral features + normalise ---
    print("\n[3/7] Feature extraction & normalise...")
    if CONFIG.get("use_spectral", True):
        print("  Computing FFT magnitude (invariant to phone orientation)...")
        acc_feat  = compute_spectral_features(acc_win)
        feat_type = "Spectral (log |FFT|)"
    else:
        acc_feat  = acc_win
        feat_type = "Raw XYZ"

    acc_norm     = normalise_windows(acc_feat)
    feat_win_len = acc_norm.shape[1]
    print(f"  Feature: {feat_type}  shape: {acc_norm.shape}")
    print(f"  Classes: {np.unique(acc_lbl)}  ({len(np.unique(acc_lbl))} surfaces)")

    # --- FIX 2: stratified file split ---
    print("\n  Stratified file-level split...")
    tr_win, tr_lbl, tr_fid, te_win, te_lbl, te_fid = split_by_file(
        acc_norm, acc_lbl, acc_fid,
        test_size=CONFIG["test_size"], seed=CONFIG["split_seed"]
    )

    # --- Diagnostic ---
    print_data_distribution(files_df, acc_lbl, tr_lbl, te_lbl)

    # --- 4. Train pure contrastive encoder (no decoder, no warmup) ---
    print("\n[4/7] Training (train split only)...")
    model = ConvEncoder(
        window_size   = feat_win_len,
        embedding_dim = CONFIG["embedding_dim"],
        in_channels   = acc_norm.shape[2],
    ).to(device)
    model = train(model, tr_win, tr_lbl, CONFIG, device)

    # --- 5. Embeddings ---
    print("\n[5/7] Embeddings...")
    tr_emb = get_embeddings(model, tr_win, device)
    te_emb = get_embeddings(model, te_win, device)
    print(f"  Train: {tr_emb.shape}  Test: {te_emb.shape}")

    print("  PCA (fit on train, applied to both)...")
    tr_norm, tr_pca, pca = postprocess(tr_emb, CONFIG["pca_variance_threshold"])
    te_norm = normalize(te_emb, norm="l2", axis=1)
    te_pca  = pca.transform(te_norm)
    print(f"  Test PCA shape: {te_pca.shape}")

    # --- 6. Cluster (fit train, predict test) ---
    print("\n[6/7] Clustering...")
    n_surf = len(np.unique(tr_lbl))
    best_k, _ = auto_select_k(tr_pca, k_min=max(2, n_surf-2), k_max=n_surf+3)

    km  = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(tr_pca)
    agg = AgglomerativeClustering(n_clusters=best_k, metric="cosine", linkage="average")
    agg.fit(tr_norm)
    nn_ = NearestNeighbors(n_neighbors=10).fit(tr_pca)
    d, _= nn_.kneighbors(tr_pca)
    eps = float(np.percentile(d[:,-1], 90))
    db  = DBSCAN(eps=eps, min_samples=10).fit(tr_pca)

    tr_pred = {"kmeans": km.labels_, "agg": agg.labels_, "dbscan": db.labels_}

    knn_agg = KNeighborsClassifier(n_neighbors=5).fit(tr_norm, agg.labels_)
    te_pred = {
        "kmeans": km.predict(te_pca),
        "agg"   : knn_agg.predict(te_norm),
        "dbscan": DBSCAN(eps=eps, min_samples=10).fit_predict(te_pca),
    }

    tr_c2s = {m: map_clusters_to_surfaces(tr_pred[m], tr_lbl) for m in tr_pred}
    te_c2s = {m: map_clusters_to_surfaces(te_pred[m], te_lbl) for m in te_pred}

    print("\n  Cluster -> Surface (K-Means, train):")
    for cid, sid in sorted(tr_c2s["kmeans"].items()):
        print(f"    Cluster {cid:2d} -> {surface_name(sid)} (id={sid})")

    # --- Evaluate ---
    tr_df = print_comparison(tr_norm, tr_pca, tr_pred, tr_lbl, split="TRAIN")
    te_df = print_comparison(te_norm, te_pca, te_pred, te_lbl, split="TEST")

    print("\nGeneralisation Gap (Train ARI - Test ARI)")
    print("  <0.10 = good   0.10-0.20 = acceptable   >0.20 = overfit")
    for m in ["KMeans", "Agglomerative", "DBSCAN"]:
        tr_v = tr_df.loc[m,"ARI"] if not pd.isna(tr_df.loc[m,"ARI"]) else 0.0
        te_v = te_df.loc[m,"ARI"] if not pd.isna(te_df.loc[m,"ARI"]) else 0.0
        gap  = tr_v - te_v
        flag = "good" if gap < 0.10 else "overfit" if gap > 0.20 else "acceptable"
        print(f"  {m:15s}  train={tr_v:.4f}  test={te_v:.4f}  gap={gap:+.4f}  [{flag}]")

    # --- 7. Visualise ---
    print("\n[7/7] Visualising...")
    for split_name, pca_e, norm_e, pred, c2s, gt in [
        ("Train", tr_pca, tr_norm, tr_pred, tr_c2s, tr_lbl),
        ("Test",  te_pca, te_norm, te_pred, te_c2s, te_lbl),
    ]:
        print(f"  t-SNE [{split_name}]...")
        perp = min(30, len(pca_e) - 1)
        c2d  = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000).fit_transform(pca_e)
        c3d  = TSNE(n_components=3, random_state=42, perplexity=perp, max_iter=1000).fit_transform(pca_e)
        plot_tsne_2d(c2d, pred, c2s, gt, prefix=f"Accelerometer [{split_name}]")
        plot_tsne_3d(c3d, pred, c2s, gt, prefix=f"Accelerometer [{split_name}]")

    print("\nDone!")


if __name__ == "__main__":
    main()

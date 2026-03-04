"""
Road Surface Classification -- Vibration Embedding & Clustering
===============================================================
Version 7: Multi-device support (SamsungGalaxyJ7 + SamsungGalaxyS7)
           with per-file frequency detection, resampling to a common
           rate, and a fixed 3-second / 50%-overlap window.

Key changes from v6:
  - Added SamsungGalaxyS7 to data loading
  - Each CSV's sampling frequency is estimated from its timestamp column
  - All signals are resampled to TARGET_HZ (default 100 Hz) via linear
    interpolation before windowing
  - Window size = TARGET_HZ * WINDOW_SECONDS (3 s = 300 samples)
  - Overlap fixed at 50%
  - 3-D visualisation removed; only 2-D t-SNE shown (test split only)
  - Train clustering performance table is still printed

Why resample to a common rate?
  Both phones record at different hardware rates (J7 ~50 Hz, S7 ~100 Hz).
  Because each FFT bin maps to  k * fs / W  Hz, mixing windows from
  different rates means the "same" bin index represents a different
  physical frequency -- the spectral block would be comparing apples to
  oranges.  Resampling to a single TARGET_HZ makes every FFT bin
  identical across devices.

Feature blocks (same as v6):
  Block 1 -- Raw XYZ         (W, 3)
  Block 2 -- Spectral FFT    (W//2, 3) zero-padded to (W, 3)
  Block 3 -- Cross-axis corr (W, 3)   X*Y, Y*Z, Z*X
  Concatenated -> (W, 9)
"""

import glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    "data_dir"              : Path("../Datasets/Processed_Data/Labeled_Data_Without_GPS"),
    "surface_types_csv"     : Path("../Datasets/surface_types.csv"),

    # Window: 3 seconds at TARGET_HZ with 50 % overlap
    "target_hz"             : 100,
    "window_seconds"        : 3,
    "overlap"               : 0.5,

    "pca_variance_threshold": 0.95,

    # Devices to include
    "devices"               : ["SamsungGalaxyJ7", "SamsungGalaxyS7"],

    # Model
    "embedding_dim" : 64,

    # Training
    "epochs"         : 150,
    "batch_size"     : 256,
    "lr"             : 3e-4,
    "contrast_w"     : 2.0,
    "temperature"    : 0.07,
    "hard_neg_ratio" : 0.5,

    # Train / test split (by FILE, not by window)
    "test_size"  : 0.2,
    "split_seed" : 42,

    # Combined feature blocks
    "feature_blocks": ["raw", "spectral", "corr"],
}

# Derived window size (set after config is finalised)
CONFIG["window_size"] = CONFIG["target_hz"] * CONFIG["window_seconds"]  # 300

# ---------------------------------------------------------------------------
# Surface name helpers
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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def extract_surface_type_id(path):
    m = re.search(r"SurfaceTypeID_(\d+)", path)
    return int(m.group(1)) if m else None


def load_files(config):
    paths = glob.glob(os.path.join(config["data_dir"], "**", "*.csv"), recursive=True)
    df = pd.DataFrame({"full_path": paths, "filename": [os.path.basename(p) for p in paths]})
    df["surface_id"] = df["full_path"].apply(extract_surface_type_id)
    df["device"]     = df["filename"].apply(lambda x: x.split("_")[3])
    df = df[df["device"].isin(config["devices"])].reset_index(drop=True)

    print("  Files per device:")
    for dev, cnt in df["device"].value_counts().items():
        print(f"    {dev}: {cnt} files")
    print("  Files per surface:")
    print(df["surface_id"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Per-file frequency estimation & resampling
# ---------------------------------------------------------------------------
def estimate_sampling_hz(df: pd.DataFrame) -> float:
    """
    Estimate sampling frequency from the timestamp column.
    Looks for columns named 'timestamp', 'time', or similar (case-insensitive).
    Falls back to counting rows / 60 s if no time column found.
    """
    time_cols = [c for c in df.columns if re.search(r"time|timestamp|t\b", c, re.I)]
    if time_cols:
        ts = pd.to_numeric(df[time_cols[0]], errors="coerce").dropna().values
        if len(ts) > 1:
            diffs = np.diff(ts)
            median_dt = np.median(diffs[diffs > 0])
            if median_dt > 0:
                # timestamps in nanoseconds (Android sensor format)
                if median_dt > 1e6:
                    return 1e9 / median_dt
                # timestamps in seconds
                elif median_dt < 1.0:
                    return 1.0 / median_dt
    # Fallback: assume 50 Hz
    return 50.0


def resample_signal(xyz: np.ndarray, src_hz: float, tgt_hz: float) -> np.ndarray:
    """
    Resample (N, 3) signal from src_hz to tgt_hz using polyphase resampling.
    If src_hz == tgt_hz (within 1 Hz tolerance) returns unchanged.
    """
    if abs(src_hz - tgt_hz) < 1.0:
        return xyz
    g = gcd(int(round(tgt_hz)), int(round(src_hz)))
    up   = int(round(tgt_hz)) // g
    down = int(round(src_hz)) // g
    resampled = resample_poly(xyz, up, down, axis=0)
    return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Windowing  (uses resampled signal)
# ---------------------------------------------------------------------------
def extract_windows(files_df, config):
    W       = config["window_size"]          # samples after resampling
    step    = int(W * (1 - config["overlap"]))
    tgt_hz  = config["target_hz"]

    acc_win, acc_lbl, acc_fid  = [], [], []
    gyr_win, gyr_lbl, gyr_fid  = [], [], []

    freq_log = {}   # fid -> detected Hz (for diagnostics)

    for fid, row in tqdm(files_df.iterrows(), total=len(files_df), desc="Windowing"):
        path = row["full_path"]
        sid  = row["surface_id"]
        df   = pd.read_csv(path)

        src_hz = estimate_sampling_hz(df)
        freq_log[fid] = src_hz

        xyz = df[["valueX", "valueY", "valueZ"]].values.astype(np.float32)
        xyz = resample_signal(xyz, src_hz, tgt_hz)

        # Ensure minimum length
        if len(xyz) < W:
            reps = (W // len(xyz)) + 1
            xyz  = np.tile(xyz, (reps, 1))[:W + step]

        is_acc = "accelerometer" in path.lower()
        is_gyr = "gyroscope"     in path.lower()

        for s in range(0, len(xyz) - W + 1, step):
            win = xyz[s:s + W]
            if is_acc:
                acc_win.append(win); acc_lbl.append(sid); acc_fid.append(fid)
            elif is_gyr:
                gyr_win.append(win); gyr_lbl.append(sid); gyr_fid.append(fid)

    # Frequency diagnostics
    hz_vals = list(freq_log.values())
    print(f"\n  Detected sampling rates: "
          f"min={min(hz_vals):.1f}  max={max(hz_vals):.1f}  "
          f"median={np.median(hz_vals):.1f} Hz  -> resampled to {tgt_hz} Hz")
    print(f"  Acc windows: {len(acc_win)}  |  Gyro windows: {len(gyr_win)}")

    return (np.array(acc_win, dtype=np.float32), np.array(acc_lbl), np.array(acc_fid),
            np.array(gyr_win, dtype=np.float32), np.array(gyr_lbl), np.array(gyr_fid))


# ---------------------------------------------------------------------------
# Feature blocks  (same as v6)
# ---------------------------------------------------------------------------
def compute_spectral_block(windows):
    """(N,W,3) -> log|FFT|, DC dropped, zero-padded to (N,W,3)"""
    W       = windows.shape[1]
    fft_mag = np.abs(np.fft.rfft(windows, axis=1))   # (N, W//2+1, 3)
    fft_mag = fft_mag[:, 1:, :]                       # drop DC
    fft_log = np.log1p(fft_mag)
    pad_len = W - fft_log.shape[1]
    fft_pad = np.pad(fft_log, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
    return fft_pad.astype(np.float32)


def compute_correlation_block(windows):
    """(N,W,3) -> X*Y, Y*Z, Z*X  (N,W,3)"""
    X, Y, Z = windows[..., 0], windows[..., 1], windows[..., 2]
    return np.stack([X * Y, Y * Z, Z * X], axis=-1).astype(np.float32)


def normalise_block(block):
    mu  = block.mean(axis=1, keepdims=True)
    std = block.std(axis=1,  keepdims=True)
    return (block - mu) / (std + 1e-8)


def build_combined_features(windows, feature_blocks):
    """Assemble (N,W,C) from selected blocks.  C = 3 * len(feature_blocks)."""
    blocks, info = [], []
    if "raw" in feature_blocks:
        blocks.append(normalise_block(windows.copy())); info.append("raw XYZ (3ch)")
    if "spectral" in feature_blocks:
        blocks.append(normalise_block(compute_spectral_block(windows))); info.append("spectral FFT (3ch)")
    if "corr" in feature_blocks:
        blocks.append(normalise_block(compute_correlation_block(windows))); info.append("XY/YZ/ZX corr (3ch)")
    combined = np.concatenate(blocks, axis=-1)
    print(f"  Feature blocks : {' + '.join(info)}")
    print(f"  Combined shape : {combined.shape}  (N, W={combined.shape[1]}, C={combined.shape[2]})")
    return combined


# ---------------------------------------------------------------------------
# Stratified file-level split
# ---------------------------------------------------------------------------
def split_by_file(windows, labels, file_ids, test_size=0.2, seed=42):
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

        shuffled = rng.permutation(files)
        n_te     = max(1, int(len(shuffled) * test_size))
        te_files = set(shuffled[:n_te].tolist())

        for idx in np.where(mask)[0]:
            (test_idx if file_ids[idx] in te_files else train_idx).append(idx)

        print(f"    {surface_name(surf):20s}: {len(shuffled)-n_te} train file(s),"
              f" {n_te} test file(s)")

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    assert set(file_ids[train_idx].tolist()).isdisjoint(set(file_ids[test_idx].tolist())), \
        "File leakage detected!"
    print(f"  Total: {len(train_idx)} train windows | {len(test_idx)} test windows")
    print("  File leakage check PASSED.")
    return (windows[train_idx], labels[train_idx], file_ids[train_idx],
            windows[test_idx],  labels[test_idx],  file_ids[test_idx])


# ---------------------------------------------------------------------------
# Diagnostic
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
# Balanced sampler
# ---------------------------------------------------------------------------
def make_balanced_sampler(labels):
    counts  = np.bincount(labels)
    weights = torch.tensor(1.0 / counts[labels], dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, window_size, embedding_dim=64, proj_dim=128, in_channels=9):
        super().__init__()
        self.window_size  = window_size
        self.reduced_size = window_size // 8
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32,  5, stride=2, padding=2), nn.BatchNorm1d(32),  nn.GELU(),
            nn.Conv1d(32,          64,  5, stride=2, padding=2), nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64,          128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.flatten  = nn.Flatten()
        self.fc_embed = nn.Linear(128 * self.reduced_size, embedding_dim)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def embed(self, x):
        x = x.permute(0, 2, 1)
        z = self.fc_embed(self.flatten(self.encoder(x)))
        return F.normalize(z, p=2, dim=-1)

    def project(self, x):
        return F.normalize(self.projector(self.embed(x)), p=2, dim=-1)

    def forward(self, x):
        z = self.embed(x)
        return z, F.normalize(self.projector(z), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Hard-negative SupCon loss
# ---------------------------------------------------------------------------
class HardNegSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, hard_neg_ratio=0.5):
        super().__init__()
        self.temperature    = temperature
        self.hard_neg_ratio = hard_neg_ratio

    def forward(self, z, labels):
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
# Training
# ---------------------------------------------------------------------------
def train(model, windows, labels, config, device):
    X       = torch.tensor(windows, dtype=torch.float32)
    Y       = torch.tensor(labels,  dtype=torch.long)
    sampler = make_balanced_sampler(labels)
    loader  = DataLoader(TensorDataset(X, Y),
                         batch_size=config["batch_size"],
                         sampler=sampler, drop_last=True)
    criterion = HardNegSupConLoss(config["temperature"],
                                  config.get("hard_neg_ratio", 0.5)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_loss, patience_counter, PATIENCE = float('inf'), 0, 30

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            _, proj = model(xb)
            loss    = criterion(proj, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg      = total_loss / len(loader)
        improved = avg < best_loss - 1e-4
        if improved:
            best_loss, patience_counter = avg, 0
        else:
            patience_counter += 1
        marker = " *" if improved else ""
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | Contrast {avg:.4f}{marker}")
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    return model


@torch.no_grad()
def get_embeddings(model, windows, device, batch_size=256):
    model.eval()
    X   = torch.tensor(windows, dtype=torch.float32)
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model.embed(X[i:i+batch_size].to(device)).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def postprocess(embeddings, variance_threshold=0.95):
    X_norm = normalize(embeddings, norm="l2", axis=1)
    pca    = PCA(n_components=variance_threshold, svd_solver="full")
    X_pca  = pca.fit_transform(X_norm)
    print(f"  Dims: {embeddings.shape[1]} -> {X_pca.shape[1]} "
          f"(variance explained: {pca.explained_variance_ratio_.sum():.4f})")
    return X_norm, X_pca, pca


# ---------------------------------------------------------------------------
# Auto-select K
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
# Cluster -> surface mapping
# ---------------------------------------------------------------------------
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
# Evaluation
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
    print(f"\nClustering Comparison{tag} {'-'*max(0,44-len(tag))}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    return df


# ---------------------------------------------------------------------------
# 2-D t-SNE visualisation  (test split only)
# ---------------------------------------------------------------------------
COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261",
          "#8338ec","#06d6a0","#fb8500","#3a86ff","#ff006e"]


def _scatter2d(ax, xy, lbls, c2s, title):
    for i, lbl in enumerate(sorted(set(lbls))):
        m = lbls == lbl
        ax.scatter(xy[m, 0], xy[m, 1],
                   c="#aaa" if lbl == -1 else COLORS[i % len(COLORS)],
                   alpha=0.65, s=10,
                   label=cluster_legend_label(lbl, c2s))
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, markerscale=1.5, framealpha=0.7,
              loc="best", title="Cluster -> Surface")
    ax.grid(alpha=0.2)


def plot_tsne_2d_test(coords, pred, c2s_map, gt, prefix="Test"):
    """2-D t-SNE for the TEST split only: 3 clustering methods + ground truth."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"t-SNE 2D -- {prefix}", fontsize=14, fontweight="bold")
    gt_c2s = {int(s): int(s) for s in np.unique(gt)}
    for ax, key, c2s, title in [
        (axes[0, 0], "kmeans", c2s_map["kmeans"], "K-Means"),
        (axes[0, 1], "agg",    c2s_map["agg"],    "Agglomerative"),
        (axes[1, 0], "dbscan", c2s_map["dbscan"], "DBSCAN"),
        (axes[1, 1], None,     gt_c2s,            "Ground Truth"),
    ]:
        lbls = gt if key is None else np.asarray(pred[key])
        _scatter2d(ax, coords, np.asarray(lbls), c2s, title)
    plt.tight_layout()
    fname = f"tsne_2d_{prefix.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("mps"  if torch.backends.mps.is_available()  else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Window: {CONFIG['window_seconds']}s  @  {CONFIG['target_hz']} Hz"
          f"  = {CONFIG['window_size']} samples  |  overlap = {CONFIG['overlap']*100:.0f}%")

    # --- 0. Surface names ---
    print("\n[0/7] Surface names...")
    load_surface_names(CONFIG["surface_types_csv"])

    # --- 1. Load files (J7 + S7) ---
    print("\n[1/7] Loading files...")
    files_df = load_files(CONFIG)

    # --- 2. Windowing (with per-file resample to TARGET_HZ) ---
    print("\n[2/7] Windowing (resample -> window)...")
    acc_win, acc_lbl, acc_fid, gyr_win, gyr_lbl, gyr_fid = extract_windows(files_df, CONFIG)

    # --- 3. Feature extraction ---
    print("\n[3/7] Feature extraction (raw + spectral FFT + cross-axis correlations)...")
    acc_norm     = build_combined_features(acc_win, CONFIG["feature_blocks"])
    feat_win_len = acc_norm.shape[1]
    in_channels  = acc_norm.shape[2]
    print(f"  Classes: {np.unique(acc_lbl)}  ({len(np.unique(acc_lbl))} surfaces)")

    # --- Stratified file split ---
    print("\n  Stratified file-level split...")
    tr_win, tr_lbl, tr_fid, te_win, te_lbl, te_fid = split_by_file(
        acc_norm, acc_lbl, acc_fid,
        test_size=CONFIG["test_size"], seed=CONFIG["split_seed"]
    )
    print_data_distribution(files_df, acc_lbl, tr_lbl, te_lbl)

    # --- 4. Train ---
    print("\n[4/7] Training (train split only)...")
    model = ConvEncoder(
        window_size   = feat_win_len,
        embedding_dim = CONFIG["embedding_dim"],
        in_channels   = in_channels,
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

    # --- 6. Clustering ---
    print("\n[6/7] Clustering...")
    n_surf = len(np.unique(tr_lbl))
    best_k, _ = auto_select_k(tr_pca, k_min=max(2, n_surf - 2), k_max=n_surf + 3)

    km  = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(tr_pca)
    agg = AgglomerativeClustering(n_clusters=best_k, metric="cosine", linkage="average")
    agg.fit(tr_norm)
    nn_ = NearestNeighbors(n_neighbors=10).fit(tr_pca)
    d, _= nn_.kneighbors(tr_pca)
    eps = float(np.percentile(d[:, -1], 90))
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

    # --- Evaluate (train + test metrics) ---
    tr_df = print_comparison(tr_norm, tr_pca, tr_pred, tr_lbl, split="TRAIN")
    te_df = print_comparison(te_norm, te_pca, te_pred, te_lbl, split="TEST")

    print("\nGeneralisation Gap (Train ARI - Test ARI)")
    print("  <0.10 = good   0.10-0.20 = acceptable   >0.20 = overfit")
    for m in ["KMeans", "Agglomerative", "DBSCAN"]:
        tr_v = tr_df.loc[m, "ARI"] if not pd.isna(tr_df.loc[m, "ARI"]) else 0.0
        te_v = te_df.loc[m, "ARI"] if not pd.isna(te_df.loc[m, "ARI"]) else 0.0
        gap  = tr_v - te_v
        flag = "good" if gap < 0.10 else "overfit" if gap > 0.20 else "acceptable"
        print(f"  {m:15s}  train={tr_v:.4f}  test={te_v:.4f}  gap={gap:+.4f}  [{flag}]")

    # --- 7. Visualise: 2-D t-SNE for TEST only ---
    print("\n[7/7] Visualising (2-D t-SNE, test split only)...")
    perp  = min(30, len(te_pca) - 1)
    te_2d = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000).fit_transform(te_pca)
    plot_tsne_2d_test(te_2d, te_pred, te_c2s, te_lbl,
                      prefix=f"Accelerometer [Test]  —  {CONFIG['window_seconds']}s window @ {CONFIG['target_hz']}Hz")

    print("\nDone!")


if __name__ == "__main__":
    main()

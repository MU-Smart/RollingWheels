"""
Road Surface Classification — VibClustNet Clustering Pipeline

Labeled rows   : surface_id is a known positive integer
Unlabeled rows : surface_id is 0, NaN, or negative

Steps
-----
1. Load windowed pkl  →  Z-normalise  →  stratified 80/20 split
2. Train VibClustNet autoencoder (multi-depth reconstruction loss)
3. Extract embeddings  →  PCA
4. Cluster (KMeans / Agglomerative / GMM) on PCA embeddings
5. Evaluate on test split
6. Visualise: t-SNE  (1×4 grid: 3 methods + ground truth)
7. VibClustNet diagnostic plots (attention, CAIM, reconstruction)
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "surface_types_csv"  : Path("../Datasets/surface_types.csv"),
    "output_csv"         : Path("23_unlabeled_predictions.csv"),
    "unlabeled_id"       : 0,          # surface_id value meaning "no label"

    # merge map type: "A" (6 classes), "B" (8 classes), "C" (9 classes)
    "merge_map_type"     : "B",

    # split
    "test_size"          : 0.2,
    "seed"               : 42,

    # PCA
    "pca_variance"       : 0.95,

    # raw windowed data
    "windowed_csv"       : Path("../Datasets/ExtractedFeatures/labeled_accelerometer_raw_windows.csv"),

    # VibClustNet hyper-parameters
    "vcn_epochs"         : 150,
    "vcn_batch_size"     : 32,
    "vcn_lr"             : 1e-3,
    "vcn_patience"       : 50,
    "vcn_embedding_dim"  : 256,
    "vcn_checkpoint"     : Path("vibclustnet_best.pth"),
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
            print(f"    {sid:3d} -> {name}")
    except Exception as e:
        print(f"  WARNING: {e}")

def sname(sid):
    return SURFACE_NAMES.get(int(sid), f"Surface {sid}")

# ── Surface merging ───────────────────────────────────────────────────────────
# Groups of surface IDs that share similar vibration patterns are collapsed
# into a single super-class before training.  Three schemes are available,
# selected via CONFIG["merge_map_type"]:
#
#   Type A (5 classes) — aggressive merging
#     0: Paving Blocks (Smooth) + Smooth Brick + Linoleum + Indoor Tile
#     1: Concrete Sidewalk
#     2: Rough Brick + Asphalt + Indoor Carpet
#     3: Curb (Up + Down)
#     4: Rect. Tiles + Paving Blocks (Rough)
#
#   Type B (7 classes, indices 0–4, 6–7) — moderate merging
#     0: Paving Blocks (Smooth) + Indoor Tile   1: Concrete Sidewalk
#     2: Brick (Smooth + Rough)                 3: Asphalt + Indoor Carpet
#     4: Indoor Linoleum                        6: Curb (Up + Down)
#     7: Rect. Tiles + Paving Blocks (Rough)    (index 5 unused)
#
#   Type C (9 classes) — fine-grained merging
#     0: Paving Blocks (Smooth)        1: Concrete Sidewalk
#     2: Smooth Brick                  3: Rough Brick
#     4: Asphalt + Indoor Carpet       5: Indoor Linoleum
#     6: Indoor Tile                   7: Curb (Up + Down)
#     8: Rect. Tiles + Paving Blocks (Rough)


MERGE_MAP_TYPE_A = {
    1:  0,   # Paving Blocks (Smooth) (Red)
    2:  1,   # Concrete Sidewalk
    3:  0,   # Smooth Brick (High Street)
    4:  2,   # Rough Brick (High Street)
    5:  2,   # Asphalt / Tar surface    <- merged with Concrete
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
    5:  3,   # Asphalt / Tar surface    <- merged with Concrete
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
    5:  4,   # Asphalt / Tar surface    <- merged with Concrete
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
N_SUPER_CLASSES = len(SUPER_NAMES)   # derived from active merge map; used to set K sweep range

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
        print(f"    {sname(cls):30s}: {len(idx)-n:5d} train  {n:4d} test")
    return (X[tr_i], y[tr_i], X[te_i], y[te_i])

# ── Embeddings ────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed(model, X, device, bs=512):
    model.eval()
    out = [model.embed(torch.tensor(X[i:i+bs]).to(device)).cpu()
           for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()

# ══════════════════════════════════════════════════════════════════════════════
# VibClustNet — raw-window autoencoder path
# ══════════════════════════════════════════════════════════════════════════════

# ── Module A: Multi-Scale Temporal Conv Block ──────────────────────────────────
class MSTCB(nn.Module):
    """5-branch multi-scale conv block → fixed 160-channel output."""
    def __init__(self, in_ch: int, out_ch: int = 160):
        super().__init__()
        b = out_ch // 5   # 32 filters per branch (5 × 32 = 160)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, b, k, padding="same"),
                          nn.BatchNorm1d(b), nn.ReLU())
            for k in [3, 7, 15, 31]
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, b, 1),
            nn.BatchNorm1d(b), nn.ReLU(),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        out = torch.cat([br(x) for br in self.branches] + [self.pool_branch(x)], dim=1)
        return F.relu(out + self.residual(x))


# ── Module B: Frequency-Aware Attention Gate ───────────────────────────────────
class FAAG(nn.Module):
    """Temporal × frequency channel gating."""
    def __init__(self, in_ch: int, T: int):
        super().__init__()
        freq_bins = T // 2 + 1
        self.temporal_gate = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, 7, padding="same", groups=in_ch),
            nn.Sigmoid(),
        )
        self.freq_mlp = nn.Sequential(
            nn.Linear(freq_bins, max(1, freq_bins // 4)),
            nn.ReLU(),
            nn.Linear(max(1, freq_bins // 4), freq_bins),
            nn.Sigmoid(),
        )

    def forward(self, x):
        t_attn = self.temporal_gate(x)                           # (B, C, T)
        mag    = torch.fft.rfft(x, dim=-1).abs()                 # (B, C, freq_bins)
        f_gate = self.freq_mlp(mag).mean(dim=-1, keepdim=True)  # (B, C, 1)
        f_attn = f_gate.expand_as(x)                             # (B, C, T)
        return x * t_attn * f_attn, t_attn, f_attn


# ── Module C: Cross-Axis Interaction Module ────────────────────────────────────
class CAIM(nn.Module):
    """Multi-head self-attention over 3 pooled axis feature vectors."""
    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        nh = num_heads if in_ch % num_heads == 0 else 1
        self.attn = nn.MultiheadAttention(in_ch, nh, batch_first=True)
        self.norm = nn.LayerNorm(in_ch)

    def forward(self, axis_feats):
        # axis_feats: list of 3 (B, C, T)
        pooled   = torch.stack([f.mean(dim=-1) for f in axis_feats], dim=1)  # (B, 3, C)
        attn_out, weights = self.attn(pooled, pooled, pooled)                 # (B, 3, C), (B, 3, 3)
        attn_out = self.norm(attn_out + pooled)
        attended = [axis_feats[i] * attn_out[:, i, :].unsqueeze(-1) for i in range(3)]
        return attended, weights


# ── VibClustNet ────────────────────────────────────────────────────────────────
class VibClustNet(nn.Module):
    """
    Autoencoder for 3-axis vibration windows (channels-first).
    Input / output: (batch, 3, T)
    """
    def __init__(self, T: int, emb_dim: int = 256):
        super().__init__()
        self.T = T
        # Shared per-axis MSTCB — same nn.Module instance for all 3 axes
        self.mstcb1 = MSTCB(1,   160)
        self.mstcb2 = MSTCB(160, 160)
        # Cross-axis + frequency attention
        self.caim   = CAIM(160, num_heads=4)
        self.faag   = FAAG(480, T)        # 3 axes × 160ch concatenated
        # Third MSTCB after attention
        self.mstcb3 = MSTCB(480, 160)
        # Encoder head
        self.enc_head = nn.Linear(160, emb_dim)
        # Decoder (mirror, simplified)
        self.dec_linear   = nn.Linear(emb_dim, 160)
        self.dec_upsample = nn.Upsample(size=T, mode="linear", align_corners=False)
        self.dec_conv1    = nn.Sequential(nn.Conv1d(160, 160, 3, padding="same"), nn.ReLU())
        self.dec_conv2    = nn.Sequential(nn.Conv1d(160,  64, 3, padding="same"), nn.ReLU())
        self.dec_out      = nn.Conv1d(64, 3, 1)

    def _encode(self, x):
        """x must be (B, 3, T). Returns emb (B, emb_dim) + enc_inter dict."""
        axes = [x[:, i:i+1, :] for i in range(3)]              # 3 × (B, 1, T)
        axes = [self.mstcb2(self.mstcb1(a)) for a in axes]     # shared weights; 3 × (B, 160, T)
        attended, caim_w = self.caim(axes)                      # 3 × (B, 160, T), (B, 3, 3)
        cat              = torch.cat(attended, dim=1)           # (B, 480, T)
        fout, t_attn, f_attn = self.faag(cat)                   # (B, 480, T)
        after3           = self.mstcb3(fout)                    # (B, 160, T)
        pooled = after3.mean(dim=-1)                            # (B, 160)
        emb    = self.enc_head(pooled)                          # (B, emb_dim)
        enc_inter = {
            "after_mstcb12": torch.stack([a.mean(-1) for a in axes], dim=1),  # (B, 3, 160)
            "after_mstcb3" : pooled,                                           # (B, 160)
            "caim_weights" : caim_w,                                           # (B, 3, 3)
            "t_attn"       : t_attn,                                           # (B, 240, T)
            "f_attn"       : f_attn,                                           # (B, 240, T)
        }
        return emb, enc_inter

    def _decode(self, emb):
        """emb: (B, emb_dim). Returns recon (B, 3, T) + dec_inter dict."""
        h  = self.dec_linear(emb).unsqueeze(-1)   # (B, 80, 1)
        h  = self.dec_upsample(h)                 # (B, 80, T)
        d1 = self.dec_conv1(h)                    # (B, 80, T)
        d2 = self.dec_conv2(d1)                   # (B, 32, T)
        recon = self.dec_out(d2)                  # (B, 3, T)
        dec_inter = {
            "d_after_upsample": h.mean(dim=-1),    # (B, 80) — mirrors after_mstcb3
            "d_after_conv1"   : d1.mean(dim=-1),   # (B, 80) — mirrors after_mstcb12 mean
        }
        return recon, dec_inter

    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)
        emb, enc_inter = self._encode(x)
        recon, dec_inter = self._decode(emb)
        return recon, emb, enc_inter, dec_inter

    def embed(self, x):
        """Inference-time embedding — normalised, no grad needed from caller."""
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)
        emb, _ = self._encode(x)
        return F.normalize(emb, dim=-1)


# ── Multi-depth reconstruction loss ───────────────────────────────────────────
def multi_rec_loss(x_orig, recon, enc_inter, dec_inter):
    """
    MSE at 3 levels (signal + two intermediate pairs), each normalised by
    its dimensionality, then summed with 0.5 weight on intermediates.
    """
    if x_orig.shape[1] != 3:
        x_orig = x_orig.permute(0, 2, 1)
    T = x_orig.shape[-1]
    # L1: main reconstruction
    l1 = F.mse_loss(recon, x_orig) / (3 * T)
    # L2: encoder bottleneck ↔ decoder upsample output  (both 80-dim, pooled)
    l2 = F.mse_loss(enc_inter["after_mstcb3"], dec_inter["d_after_upsample"]) / 80
    # L3: encoder stage-1 mean (avg 3 axes → 80-dim) ↔ decoder conv1 output
    enc_s1 = enc_inter["after_mstcb12"].mean(dim=1)   # (B, 80)
    l3 = F.mse_loss(enc_s1, dec_inter["d_after_conv1"]) / 80
    return l1 + 0.5 * l2 + 0.5 * l3


# ── Windowed dataset ───────────────────────────────────────────────────────────
class WindowedDataset(torch.utils.data.Dataset):
    """Dataset wrapping a (N, 3, T) float32 numpy array + optional labels."""
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = (torch.tensor(y, dtype=torch.long) if y is not None
                  else torch.full((len(X),), -1, dtype=torch.long))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Load windowed csv ──────────────────────────────────────────────────────────
def load_windowed_data(cfg):
    """
    Loads the CSV file produced by FeatureExtractor notebook.
    Columns: valueX, valueY, valueZ, surface_id, window_id.
    Groups by window_id, Z-normalises per window per channel, remaps labels, stratified splits.
    Returns tr_X, tr_y, te_X, te_y  (float32 (N,3,T)),  X_u (unlabeled).
    """
    csv_path = cfg["windowed_csv"]
    print(f"  Loading: {csv_path}")
    raw = pd.read_csv(csv_path)
    print(f"  Rows: {len(raw)}  Windows: {raw['window_id'].nunique()}")

    # ── Group by window_id → arr (N, 3, T) + labels ───────────────────────────
    windows, labels = [], []
    for _, group in raw.groupby("window_id", sort=True):
        xyz = group[["valueX", "valueY", "valueZ"]].to_numpy(dtype=np.float32).T  # (3, T)
        sid = int(group["surface_id"].iloc[0])

        windows.append(xyz)
        labels.append(sid)
    arr    = np.stack(windows).astype(np.float32)   # (N, 3, T)
    labels = np.array(labels, dtype=int)

    print(f"\n  Windows: {arr.shape}  Labels: {labels.shape}")

    # ── Z-normalise per window per channel ────────────────────────────────────
    mu  = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1,  keepdims=True).clip(1e-8)
    arr = (arr - mu) / std

    # ── Labeled / unlabeled split ─────────────────────────────────────────────
    unl_mask = (labels == cfg["unlabeled_id"]) | (labels < 0)
    X_l, y_l = arr[~unl_mask], labels[~unl_mask]
    X_u       = arr[unl_mask]
    y_l       = remap_labels(y_l)
    print(f"  Labeled: {len(X_l)}  Unlabeled: {len(X_u)}")
    print(f"  Classes: {sorted(np.unique(y_l))}")

    # ── Stratified 80/20 split ────────────────────────────────────────────────
    print("  Stratified split:")
    tr_X, tr_y, te_X, te_y = stratified_split(X_l, y_l, cfg["test_size"], cfg["seed"])
    return tr_X, tr_y, te_X, te_y, X_u


# ── VibClustNet training ───────────────────────────────────────────────────────
def train_vibclustnet(model, train_loader, val_loader, cfg, device,
                      X_labeled=None, y_labeled=None):
    """
    Trains VibClustNet autoencoder with multi_rec_loss.
    - Early stopping on val loss (patience = cfg["vcn_patience"])
    - Gradient clipping max_norm=1.0
    - Every 10 epochs: KMeans NMI probe on labeled subset
    - Saves / reloads best checkpoint
    """
    opt  = torch.optim.Adam(model.parameters(), lr=cfg["vcn_lr"])
    best = float("inf")
    pat  = 0
    ckpt = cfg["vcn_checkpoint"]

    for ep in range(cfg["vcn_epochs"]):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, _, enc_inter, dec_inter = model(xb)
            loss = multi_rec_loss(xb, recon, enc_inter, dec_inter)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                recon, _, enc_inter, dec_inter = model(xb)
                va_loss += multi_rec_loss(xb, recon, enc_inter, dec_inter).item()
        va_loss /= len(val_loader)

        imp  = va_loss < best - 1e-6
        best, pat = (va_loss, 0) if imp else (best, pat + 1)
        if imp:
            torch.save(model.state_dict(), ckpt)
        print(f"  [VCN] {ep+1:3d}/{cfg['vcn_epochs']}  "
              f"train={tr_loss:.6f}  val={va_loss:.6f}" + (" *" if imp else ""))

        # ── NMI probe every 10 epochs ─────────────────────────────────────────
        if (ep + 1) % 10 == 0 and X_labeled is not None and len(X_labeled) >= 2:
            with torch.no_grad():
                embs = np.concatenate([
                    model.embed(torch.tensor(X_labeled[i:i+256]).to(device)).cpu().numpy()
                    for i in range(0, len(X_labeled), 256)
                ])
            k_tmp  = len(np.unique(y_labeled))
            km_tmp = KMeans(n_clusters=k_tmp, random_state=42, n_init=5).fit_predict(embs)
            nmi    = normalized_mutual_info_score(y_labeled, km_tmp)
            print(f"    NMI (labeled, K={k_tmp}): {nmi:.4f}")

        if pat >= cfg["vcn_patience"]:
            print(f"  Early stop at epoch {ep+1}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model



# ── Post-processing ───────────────────────────────────────────────────────────
def pca_reduce(emb_tr, emb_te, variance=0.95):
    n_tr = normalize(emb_tr, norm="l2")
    n_te = normalize(emb_te, norm="l2")
    pca  = PCA(n_components=variance, svd_solver="full").fit(n_tr)
    print(f"  PCA: {emb_tr.shape[1]}d → {pca.n_components_}d "
          f"(var={pca.explained_variance_ratio_.sum():.3f})")
    return n_tr, pca.transform(n_tr), n_te, pca.transform(n_te), pca

# ── Clustering ────────────────────────────────────────────────────────────────
def best_k(pca_emb, k_min, k_max):
    scores = {}
    for k in range(k_min, k_max+1):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(pca_emb)
        scores[k] = silhouette_score(pca_emb, lbl)
        print(f"    K={k:2d}  sil={scores[k]:.4f}")
    bk = max(scores, key=scores.get)
    print(f"  Best K={bk}  sil={scores[bk]:.4f}")
    # plot
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(list(scores.keys()), list(scores.values()), "o-", color="#457b9d")
    ax.axvline(bk, color="#e63946", ls="--", label=f"K={bk}")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette")
    ax.set_title("K selection"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("17_k_selection.png", dpi=150); plt.show()
    return bk


def _fixed_k_grid(pca_emb, proj_coords, proj_name, fname, ks):
    """2×2 grid of KMeans at fixed K values, coloured by cluster only."""
    assert len(ks) == 4, "Need exactly 4 K values for 2×2 grid"
    rows = {}
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Fixed-K experiment — {proj_name} (test split)",
                 fontsize=13, fontweight="bold")
    for ax, k in zip(axes.flat, ks):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(pca_emb)
        rows[f"K={k}"] = dict(
            Silhouette = silhouette_score(pca_emb, lbl),
            DB         = davies_bouldin_score(pca_emb, lbl),
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
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  Saved {fname}")
    return rows

def experiment_fixed_k(pca_emb, ts, ks=(3, 5, 7, 11)):
    print(f"\n  Fixed-K experiment  K={list(ks)}")
    rows_ts = _fixed_k_grid(pca_emb, ts, "t-SNE", "20_fixed_k_tsne.png", ks)

    print("\n  Cluster quality — t-SNE projection:")
    _print_quality(rows_ts)
    return pd.DataFrame(rows_ts).T

def _print_quality(rows):
    print(f"  {'K':<8} {'Silhouette':>12} {'Davies-Bouldin':>16}  Quality")
    print("  " + "-" * 46)
    for k, r in rows.items():
        sil, db = r["Silhouette"], r["DB"]
        q = "Excellent" if sil > 0.6 else "Good" if sil > 0.4 else "Fair" if sil > 0.2 else "Poor"
        print(f"  {k:<8} {sil:>12.4f} {db:>16.4f}  {q}")

def cluster_all(tr_norm, tr_pca, te_norm, te_pca, k):
    km      = KMeans(n_clusters=k, random_state=42, n_init=20).fit(tr_pca)
    agg     = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(tr_norm)
    gmm     = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(tr_pca)

    tr_pred = {"kmeans": km.labels_, "agg": agg.labels_, "gmm": gmm.predict(tr_pca)}
    knn_agg = KNeighborsClassifier(n_neighbors=5).fit(tr_norm, agg.labels_)
    te_pred = {
        "kmeans": km.predict(te_pca),
        "agg"   : knn_agg.predict(te_norm),
        "gmm"   : gmm.predict(te_pca),
    }
    return tr_pred, te_pred, km

# ── Evaluation ────────────────────────────────────────────────────────────────
def c2s_map(pred, gt):
    return {c: (-1 if c == -1 else int(pd.Series(gt[np.asarray(pred)==c]).mode()[0]))
            for c in set(pred)}

def evaluate(emb, pred, gt, metric="euclidean"):
    pred = np.asarray(pred); ok = pred != -1
    ev, lv, gv = emb[ok], pred[ok], np.asarray(gt)[ok]
    if len(np.unique(lv)) < 2:
        return dict(Silhouette=np.nan, DB=np.nan, CH=np.nan, ARI=np.nan, NMI=np.nan)
    return dict(
        Silhouette = silhouette_score(ev, lv, metric=metric),
        DB         = davies_bouldin_score(ev, lv),
        CH         = calinski_harabasz_score(ev, lv),
        ARI        = adjusted_rand_score(gv, lv),
        NMI        = normalized_mutual_info_score(gv, lv),
    )

def print_metrics(tr_norm, tr_pca, tr_pred, tr_lbl,
                  te_norm, te_pca, te_pred, te_lbl):
    for split, en, ep, pred, gt in [
        ("TRAIN", tr_norm, tr_pca, tr_pred, tr_lbl),
        ("TEST",  te_norm, te_pca, te_pred, te_lbl),
    ]:
        rows = {
            "KMeans" : evaluate(ep, pred["kmeans"], gt),
            "Agg"    : evaluate(en, pred["agg"],    gt, "cosine"),
            "GMM"    : evaluate(ep, pred["gmm"],    gt),
        }
        print(f"\n── {split} ──────────────────────────────────────")
        print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nGeneralisation gap  (train ARI − test ARI):")
    for m, ek, em in [("KMeans", "kmeans", "euclidean"),
                      ("Agg",    "agg",    "cosine"),
                      ("GMM",    "gmm",    "euclidean")]:
        tr_v = evaluate(tr_pca if em=="euclidean" else tr_norm, tr_pred[ek], tr_lbl, em)["ARI"] or 0
        te_v = evaluate(te_pca if em=="euclidean" else te_norm, te_pred[ek], te_lbl, em)["ARI"] or 0
        gap  = (tr_v or 0) - (te_v or 0)
        flag = "good" if gap < 0.10 else "overfit" if gap > 0.20 else "acceptable"
        print(f"  {m:8s}  train={tr_v:.4f}  test={te_v:.4f}  gap={gap:+.4f}  [{flag}]")

# ── VibClustNet diagnostic plots ──────────────────────────────────────────────
def plot_vibclustnet_diagnostics(model, te_X, te_pred_kmeans, device):
    """
    Three diagnostic plots (saved to disk, no plt.show() blocking):
      1. Average temporal attention profile per cluster (mean over samples & channels → T)
      2. CAIM cross-axis attention heatmap per cluster (3×3 matrix, X/Y/Z)
      3. Original vs reconstructed signal for 3 random test samples (all 3 axes)
    """
    model.eval()
    all_t_attn, all_caim_w = [], []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(te_X), bs):
            xb = torch.tensor(te_X[i:i+bs]).to(device)
            if xb.shape[1] != 3:
                xb = xb.permute(0, 2, 1)
            _, enc_inter = model._encode(xb)
            all_t_attn.append(enc_inter["t_attn"].cpu().numpy())    # (b, 240, T)
            all_caim_w.append(enc_inter["caim_weights"].cpu().numpy())  # (b, 3, 3)

    all_t_attn = np.concatenate(all_t_attn, axis=0)   # (N, 240, T)
    all_caim_w = np.concatenate(all_caim_w, axis=0)   # (N, 3, 3)
    clusters   = sorted(set(te_pred_kmeans))
    n_cls      = len(clusters)

    # ── 1. Temporal attention per cluster ────────────────────────────────────
    fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 3), squeeze=False)
    fig.suptitle("VibClustNet — Avg Temporal Attention per Cluster", fontweight="bold")
    for ax, cid in zip(axes[0], clusters):
        mask = te_pred_kmeans == cid
        avg  = all_t_attn[mask].mean(axis=(0, 1))   # mean over N and channels → (T,)
        ax.plot(avg, color=COLORS[cid % len(COLORS)], linewidth=1.0)
        ax.set_title(f"Cluster {cid}", fontsize=9)
        ax.set_xlabel("Time step"); ax.set_ylabel("Attention")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("vcn_temporal_attention.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved vcn_temporal_attention.png")

    # ── 2. CAIM cross-axis heatmap per cluster ────────────────────────────────
    axis_labels = ["X", "Y", "Z"]
    fig, axes = plt.subplots(1, n_cls, figsize=(3.5 * n_cls, 3.5), squeeze=False)
    fig.suptitle("VibClustNet — CAIM Cross-Axis Attention per Cluster", fontweight="bold")
    for ax, cid in zip(axes[0], clusters):
        mask  = te_pred_kmeans == cid
        avg_w = all_caim_w[mask].mean(axis=0)           # (3, 3)
        im    = ax.imshow(avg_w, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(axis_labels)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(axis_labels)
        ax.set_title(f"Cluster {cid}", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("vcn_caim_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved vcn_caim_heatmap.png")

    # ── 3. Reconstruction for 3 random test samples ───────────────────────────
    rng  = np.random.default_rng(42)
    idxs = rng.choice(len(te_X), size=min(3, len(te_X)), replace=False)
    fig, axes = plt.subplots(len(idxs), 3, figsize=(15, 4 * len(idxs)), squeeze=False)
    fig.suptitle("VibClustNet — Reconstruction (test samples)", fontweight="bold")
    axis_names = ["X", "Y", "Z"]
    with torch.no_grad():
        for row, idx in enumerate(idxs):
            xb = torch.tensor(te_X[idx:idx+1]).to(device)
            if xb.shape[1] != 3:
                xb = xb.permute(0, 2, 1)
            recon, _, _, _ = model(xb)
            orig = xb[0].cpu().numpy()    # (3, T)
            rec  = recon[0].cpu().numpy() # (3, T)
            for col in range(3):
                ax = axes[row, col]
                ax.plot(orig[col], label="Original",      alpha=0.85, linewidth=0.8)
                ax.plot(rec[col],  label="Reconstructed", alpha=0.85, linewidth=0.8, ls="--")
                ax.set_title(f"Sample {idx} — Axis {axis_names[col]}", fontsize=9)
                ax.set_xlabel("Time"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("vcn_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved vcn_reconstruction.png")


# ── Visualisation ─────────────────────────────────────────────────────────────
COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261",
          "#8338ec","#06d6a0","#fb8500","#3a86ff","#ff006e",
          "#c77dff","#80b918"]

def _scatter(ax, xy, lbls, legend_fn, title, xlabel):
    for i, lbl in enumerate(sorted(set(lbls))):
        m = np.asarray(lbls) == lbl
        ax.scatter(xy[m,0], xy[m,1],
                   c="#bbb" if lbl==-1 else COLORS[i % len(COLORS)],
                   alpha=0.6, s=8, label=legend_fn(lbl))
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{xlabel} 1"); ax.set_ylabel(f"{xlabel} 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6)
    ax.grid(alpha=0.2)

def project(pca_emb, tag):
    perp = min(30, len(pca_emb) - 1)
    print(f"  t-SNE [{tag}]...")
    ts = TSNE(n_components=2, random_state=42,
              perplexity=perp, max_iter=1000).fit_transform(pca_emb)
    return ts

def _plot_proj_grid(coords, proj_name, pred, c2s, gt, fname):
    """1-row × 4-col: K-Means | Agglomerative | Spectral | Ground Truth."""
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.suptitle(f"{proj_name} — Test split (super-classes)",
                 fontsize=13, fontweight="bold")
    methods = [
        ("kmeans", lambda lbl: f"{super_name(c2s['kmeans'].get(lbl,'?'))} [c{lbl}]", "K-Means"),
        ("agg",    lambda lbl: f"{super_name(c2s['agg'].get(lbl,'?'))} [c{lbl}]",    "Agglomerative"),
        ("gmm",    lambda lbl: f"{super_name(c2s['gmm'].get(lbl,'?'))} [c{lbl}]",    "GMM"),
    ]
    for col, (key, leg_fn, title) in enumerate(methods):
        _scatter(axes[col], coords, pred[key], leg_fn,
                 f"{proj_name} — {title}", proj_name)
    # Ground truth
    ax = axes[3]
    for i, sid in enumerate(sorted(np.unique(gt))):
        m = np.asarray(gt) == sid
        ax.scatter(coords[m,0], coords[m,1],
                   c=COLORS[i % len(COLORS)], alpha=0.6, s=8, label=super_name(sid))
    ax.set_title(f"{proj_name} — Ground Truth", fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6, title="Super-class")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  Saved {fname}")

def plot_grid(ts, pred, c2s, gt):
    _plot_proj_grid(ts, "t-SNE", pred, c2s, gt, fname="18_tsne_test_clusters.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("[0] Surface names")
    load_surface_names(CONFIG["surface_types_csv"])

    print("\n[1] Load windowed pkl")
    tr_X, tr_y, te_X, te_y, _ = load_windowed_data(CONFIG)
    T = tr_X.shape[2]          # tr_X is (N, 3, T)

    print("\n[2] Super-class distribution (train):")
    for sc, n in sorted(zip(*np.unique(tr_y, return_counts=True))):
        print(f"    {super_name(sc):30s}: {n} windows")

    print("\n[3] Train VibClustNet autoencoder")
    tr_ds = WindowedDataset(tr_X, tr_y)
    va_ds = WindowedDataset(te_X, te_y)
    tr_loader = DataLoader(tr_ds, batch_size=CONFIG["vcn_batch_size"],
                           shuffle=True,  drop_last=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=CONFIG["vcn_batch_size"],
                           shuffle=False, num_workers=0)
    model = VibClustNet(T=T, emb_dim=CONFIG["vcn_embedding_dim"]).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = train_vibclustnet(model, tr_loader, va_loader, CONFIG, device,
                              X_labeled=tr_X, y_labeled=tr_y)

    print("\n[4] Embeddings + PCA")
    tr_emb = embed(model, tr_X, device)
    te_emb = embed(model, te_X, device)

    tr_norm, tr_pca, te_norm, te_pca, _ = pca_reduce(tr_emb, te_emb, CONFIG["pca_variance"])

    print("\n[7] Clustering")
    n_surf = N_SUPER_CLASSES
    print(f"  Super-classes={n_surf}  — sweeping K {max(2,n_surf-2)}..{n_surf+2}")
    k = best_k(tr_pca, max(2, n_surf-2), n_surf+2)
    tr_pred, te_pred, _ = cluster_all(tr_norm, tr_pca, te_norm, te_pca, k)

    tr_c2s = {m: c2s_map(tr_pred[m], tr_y) for m in tr_pred}
    te_c2s = {m: c2s_map(te_pred[m], te_y) for m in te_pred}

    print("\n  K-Means cluster → surface (train):")
    for cid, sid in sorted(tr_c2s["kmeans"].items()):
        print(f"    cluster {cid:2d} → {super_name(sid)}")

    print("\n[8] Metrics")
    print_metrics(tr_norm, tr_pca, tr_pred, tr_y,
                  te_norm, te_pca, te_pred, te_y)

    print("\n[9] Visualise (t-SNE on test split)")
    ts = project(te_pca, "Test")
    plot_grid(ts, te_pred, te_c2s, te_y)

    # Fixed-K experiment: K=3, 5, 7, 11
    experiment_fixed_k(te_pca, ts, ks=(3, 5, 7, 11))

    print("\n[10] VibClustNet diagnostic plots")
    plot_vibclustnet_diagnostics(model, te_X, te_pred["kmeans"], device)

    print("\nDone!")

if __name__ == "__main__":
    main()

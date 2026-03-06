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
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

# ── Logging setup ─────────────────────────────────────────────────────────────
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(log_dir: Path = Path(".")) -> logging.Logger:
    """Set up logger writing to both console and a timestamped log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{RUN_TS}_22_gpu.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("vcn")
    logger.setLevel(logging.DEBUG)

    # File handler — captures DEBUG and above
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
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
    "windowed_csv"       : Path("../../Datasets/ExtractedFeatures/labeled_accelerometer_raw_windows.csv"),

    # VibClustNet hyper-parameters
    "vcn_epochs"         : 150,
    "vcn_batch_size"     : 32,
    "vcn_lr"             : 1e-3,
    "vcn_patience"       : 50,
    "vcn_embedding_dim"  : 128,
    "vcn_checkpoint"     : Path("vibclustnet_best_22.pth"),
    "vcn_cls_weight"     : 1.0,    # weight for classification auxiliary loss

    # Output directories
    "figures_dir"        : Path("figures"),
    "models_dir"         : Path("models"),
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
        logger.info(f"    {sname(cls):30s}: {len(idx)-n:5d} train  {n:4d} test")
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
    """4-branch multi-scale conv block → fixed output channels."""
    def __init__(self, in_ch: int, out_ch: int = 96):
        super().__init__()
        b = out_ch // 4   # filters per branch (3 conv + 1 pool = 4 × b)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, b, k, padding="same"),
                          nn.BatchNorm1d(b), nn.ReLU())
            for k in [3, 7, 15]
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
    n_classes: if > 0, adds a classification head for auxiliary supervised loss.
    """
    def __init__(self, T: int, emb_dim: int = 256, n_classes: int = 0):
        super().__init__()
        self.T = T
        CH = 96                                      # internal channel width
        # Shared per-axis MSTCB — same nn.Module instance for all 3 axes
        self.mstcb1 = MSTCB(1,      CH)
        self.mstcb2 = MSTCB(CH,     CH)
        # Cross-axis + frequency attention
        self.caim   = CAIM(CH, num_heads=4)
        self.faag   = FAAG(3 * CH, T)               # 3 axes × 96ch = 288
        # Third MSTCB after attention
        self.mstcb3 = MSTCB(3 * CH, CH)
        # Encoder head
        self.enc_head = nn.Linear(CH, emb_dim)
        # Optional classification head (auxiliary supervised loss)
        self.classifier = (
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(emb_dim // 2, n_classes),
            ) if n_classes > 0 else None
        )
        # Decoder (mirror, simplified)
        self.dec_linear   = nn.Linear(emb_dim, CH)
        self.dec_upsample = nn.Upsample(size=T, mode="linear", align_corners=False)
        self.dec_conv1    = nn.Sequential(nn.Conv1d(CH, CH, 3, padding="same"), nn.ReLU())
        self.dec_conv2    = nn.Sequential(nn.Conv1d(CH, CH // 3, 3, padding="same"), nn.ReLU())
        self.dec_out      = nn.Conv1d(CH // 3, 3, 1)

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

    def classify(self, emb):
        """Raw logits from the auxiliary classification head."""
        if self.classifier is None:
            raise RuntimeError("VibClustNet built without n_classes > 0")
        return self.classifier(F.normalize(emb, dim=-1))


# ── Log-cosh reconstruction loss ──────────────────────────────────────────────
def multi_rec_loss(x_orig, recon):
    """
    Log-cosh reconstruction loss: Σᵢ log(cosh(yᵢ − ŷᵢ))  (sum over all n elements).

    Numerically stable via the identity:
        log(cosh(d)) = logaddexp(d, −d) − log(2)
                     = log(eᵈ + e⁻ᵈ) − log(2)

    Behaves like ½d² for small residuals (smooth, like MSE) and like |d| − log2
    for large residuals (robust, like MAE) — avoids the predict-zero collapse
    that MSE alone suffers on Z-normalised data.
    """
    if x_orig.shape[1] != 3:
        x_orig = x_orig.permute(0, 2, 1)
    d    = recon - x_orig
    ln2  = torch.tensor(2.0, device=d.device).log()
    return (torch.logaddexp(d, -d) - ln2).sum()


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
    logger.info(f"  Loading: {csv_path}")
    raw = pd.read_csv(csv_path)
    logger.info(f"  Rows: {len(raw)}  Windows: {raw['window_id'].nunique()}")

    # ── Group by window_id → arr (N, 3, T) + labels ───────────────────────────
    windows, labels = [], []
    for _, group in raw.groupby("window_id", sort=True):
        xyz = group[["valueX", "valueY", "valueZ"]].to_numpy(dtype=np.float32).T  # (3, T)
        sid = int(group["surface_id"].iloc[0])

        windows.append(xyz)
        labels.append(sid)
    arr    = np.stack(windows).astype(np.float32)   # (N, 3, T)
    labels = np.array(labels, dtype=int)

    logger.info(f"\n  Windows: {arr.shape}  Labels: {labels.shape}")

    # ── Z-normalise per window per channel ────────────────────────────────────
    mu  = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1,  keepdims=True).clip(1e-8)
    arr = (arr - mu) / std

    # ── Labeled / unlabeled split ─────────────────────────────────────────────
    unl_mask = (labels == cfg["unlabeled_id"]) | (labels < 0)
    X_l, y_l = arr[~unl_mask], labels[~unl_mask]
    X_u       = arr[unl_mask]
    y_l       = remap_labels(y_l)
    logger.info(f"  Labeled: {len(X_l)}  Unlabeled: {len(X_u)}")
    logger.info(f"  Classes: {sorted(np.unique(y_l))}")

    # ── Stratified 80/20 split ────────────────────────────────────────────────
    logger.info("  Stratified split:")
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
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-5)
    best = float("inf")
    pat  = 0
    ckpt = cfg["vcn_checkpoint"]

    for ep in range(cfg["vcn_epochs"]):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        cls_w = cfg.get("vcn_cls_weight", 0.0)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            recon, emb, _, _ = model(xb)
            loss = multi_rec_loss(xb, recon)
            # Auxiliary classification loss on labeled samples
            if model.classifier is not None and cls_w > 0:
                labeled = yb >= 0
                if labeled.any():
                    logits = model.classify(emb[labeled])
                    loss = loss + cls_w * F.cross_entropy(logits, yb[labeled])
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
                recon, _, _, _ = model(xb)
                va_loss += multi_rec_loss(xb, recon).item()
        va_loss /= len(val_loader)

        sched.step(va_loss)
        imp  = va_loss < best - 1e-6
        best, pat = (va_loss, 0) if imp else (best, pat + 1)
        if imp:
            torch.save(model.state_dict(), ckpt)
        lr_now = opt.param_groups[0]["lr"]
        logger.info(f"  [VCN] {ep+1:3d}/{cfg['vcn_epochs']}  "
                    f"train={tr_loss:.6f}  val={va_loss:.6f}  lr={lr_now:.2e}"
                    + (" *" if imp else ""))

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
            logger.info(f"    NMI (labeled, K={k_tmp}): {nmi:.4f}")

        if pat >= cfg["vcn_patience"]:
            logger.info(f"  Early stop at epoch {ep+1}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def save_model(model, cfg):
    """Save full model and state dict with run timestamp."""
    models_dir = cfg["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    sd_path   = models_dir / f"vibclustnet_{RUN_TS}_state_dict.pth"
    full_path = models_dir / f"vibclustnet_{RUN_TS}_full.pth"

    torch.save(model.state_dict(), sd_path)
    torch.save(model, full_path)

    logger.info(f"  Model state dict saved: {sd_path.resolve()}")
    logger.info(f"  Full model saved      : {full_path.resolve()}")
    return sd_path, full_path


# ── Post-processing ───────────────────────────────────────────────────────────
def pca_reduce(emb_tr, emb_te, variance=0.95):
    n_tr = normalize(emb_tr, norm="l2")
    n_te = normalize(emb_te, norm="l2")
    pca  = PCA(n_components=variance, svd_solver="full").fit(n_tr)
    logger.info(f"  PCA: {emb_tr.shape[1]}d → {pca.n_components_}d "
                f"(var={pca.explained_variance_ratio_.sum():.3f})")
    return n_tr, pca.transform(n_tr), n_te, pca.transform(n_te), pca

# ── Additional clustering algorithms ─────────────────────────────────────────

class SBScanClustering:
    """DBSCAN with automatic epsilon selection via k-NN distance elbow."""
    def __init__(self, n_clusters_hint: int = 5, min_samples: int = 5):
        self.n_clusters_hint = n_clusters_hint
        self.min_samples     = min_samples
        self.labels_         = None
        self.eps_            = None
        self._knn_clf        = None

    def fit(self, X):
        k        = self.min_samples
        nbrs     = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nbrs.kneighbors(X)
        k_dists  = np.sort(dists[:, -1])          # ascending
        n        = len(k_dists)
        # Knee-point: normalise both axes to [0,1], then max |y - x| gives
        # the index where the curve deviates most from the diagonal.
        xs = np.linspace(0.0, 1.0, n)
        rng_d = k_dists[-1] - k_dists[0]
        ys = (k_dists - k_dists[0]) / (rng_d + 1e-10)
        knee_idx  = int(np.argmax(np.abs(ys - xs)))
        self.eps_ = float(k_dists[knee_idx])
        if self.eps_ < 1e-6:                       # degenerate: use median
            self.eps_ = float(np.median(k_dists))
        logger.info(f"    SBScan auto-eps={self.eps_:.4f}  (knee idx={knee_idx}/{n})")
        db = DBSCAN(eps=self.eps_, min_samples=self.min_samples).fit(X)
        self.labels_ = db.labels_
        mask    = self.labels_ != -1
        n_valid = int(mask.sum())
        n_cls   = len(np.unique(self.labels_[mask])) if n_valid > 0 else 0
        logger.info(f"    SBScan: {n_cls} clusters, {n_valid} non-noise / {n} total")
        if n_valid > 1 and n_cls > 1:
            self._knn_clf = KNeighborsClassifier(
                n_neighbors=min(5, n_valid)
            ).fit(X[mask], self.labels_[mask])
        return self

    def predict(self, X):
        if self._knn_clf is None:
            return np.zeros(len(X), dtype=int)
        return self._knn_clf.predict(X)


class PSOClustering:
    """Particle Swarm Optimisation clustering (minimise WCSS)."""
    def __init__(self, n_clusters: int = 5, n_particles: int = 15,
                 max_iter: int = 50, seed: int = 42):
        self.n_clusters  = n_clusters
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.seed        = seed
        self.centroids_  = None
        self.labels_     = None

    def _assign(self, X, centroids):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
        return dists.argmin(axis=1)

    def _wcss(self, X, centroids):
        asgn = self._assign(X, centroids)
        return sum(((X[asgn == k] - centroids[k]) ** 2).sum()
                   for k in range(len(centroids)) if (asgn == k).any())

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
    """Gravitational Search Algorithm (GSA) clustering."""
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
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
        return dists.argmin(axis=1)

    def _fitness(self, X, centroids):
        asgn = self._assign(X, centroids)
        return sum(((X[asgn == k] - centroids[k]) ** 2).sum()
                   for k in range(len(centroids)) if (asgn == k).any())

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
    """Random centroid initialisation + nearest-neighbour assignment (no iteration)."""
    def __init__(self, n_clusters: int = 5, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed       = seed
        self.centroids_ = None
        self.labels_    = None

    def _assign(self, X, centroids):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
        return dists.argmin(axis=1)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        idxs = rng.choice(len(X), size=self.n_clusters, replace=False)
        self.centroids_ = X[idxs]
        self.labels_    = self._assign(X, self.centroids_)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


# ── Clustering ────────────────────────────────────────────────────────────────
def best_k(pca_emb, k_min, k_max, figures_dir: Path):
    scores = {}
    for k in range(k_min, k_max+1):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(pca_emb)
        scores[k] = silhouette_score(pca_emb, lbl)
        logger.info(f"    K={k:2d}  sil={scores[k]:.4f}")
    bk = max(scores, key=scores.get)
    logger.info(f"  Best K={bk}  sil={scores[bk]:.4f}")
    # plot
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
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")
    return rows

def experiment_fixed_k(pca_emb, ts, figures_dir: Path, ks=(3, 5, 7, 11)):
    logger.info(f"\n  Fixed-K experiment  K={list(ks)}")
    fname = figures_dir / f"20_fixed_k_tsne_{RUN_TS}.png"
    rows_ts = _fixed_k_grid(pca_emb, ts, "t-SNE", fname, ks)

    logger.info("\n  Cluster quality — t-SNE projection:")
    _print_quality(rows_ts)
    return pd.DataFrame(rows_ts).T

def _print_quality(rows):
    logger.info(f"  {'K':<8} {'Silhouette':>12} {'Davies-Bouldin':>16}  Quality")
    logger.info("  " + "-" * 46)
    for k, r in rows.items():
        sil, db = r["Silhouette"], r["DB"]
        q = "Excellent" if sil > 0.6 else "Good" if sil > 0.4 else "Fair" if sil > 0.2 else "Poor"
        logger.info(f"  {k:<8} {sil:>12.4f} {db:>16.4f}  {q}")

def cluster_all(tr_norm, tr_pca, te_norm, te_pca, k):
    km      = KMeans(n_clusters=k, random_state=42, n_init=20).fit(tr_pca)
    agg     = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(tr_norm)
    gmm     = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(tr_pca)
    sbscan  = SBScanClustering(n_clusters_hint=k, min_samples=5).fit(tr_pca)
    pos     = PSOClustering(n_clusters=k, seed=42).fit(tr_pca)
    rand_a  = RandomAssignClustering(n_clusters=k, seed=42).fit(tr_pca)
    gsa     = GravitationalSearchClustering(n_clusters=k, seed=42).fit(tr_pca)
    rand_c  = RandomClustering(n_clusters=k, seed=42).fit(tr_pca)

    # Agg test prediction: nearest-centroid in cosine space is much more robust
    # than KNN when Agglomerative produces imbalanced clusters.
    agg_centroids = np.vstack([tr_norm[agg.labels_ == c].mean(axis=0)
                               for c in range(k)])
    agg_centroids = normalize(agg_centroids, norm="l2")
    te_agg_pred   = np.linalg.norm(
        te_norm[:, None, :] - agg_centroids[None, :, :], axis=-1
    ).argmin(axis=1)

    tr_pred = {
        "kmeans"     : km.labels_,
        "agg"        : agg.labels_,
        "gmm"        : gmm.predict(tr_pca),
        "sbscan"     : sbscan.labels_,
        "pos"        : pos.labels_,
        "rand_assign": rand_a.labels_,
        "gsa"        : gsa.labels_,
        "rand_clust" : rand_c.labels_,
    }
    te_pred = {
        "kmeans"     : km.predict(te_pca),
        "agg"        : te_agg_pred,
        "gmm"        : gmm.predict(te_pca),
        "sbscan"     : sbscan.predict(te_pca),
        "pos"        : pos.predict(te_pca),
        "rand_assign": RandomAssignClustering(n_clusters=k, seed=99).fit(te_pca).labels_,
        "gsa"        : gsa.predict(te_pca),
        "rand_clust" : RandomClustering(n_clusters=k, seed=99).fit(te_pca).labels_,
    }
    return tr_pred, te_pred, km

# ── Evaluation ────────────────────────────────────────────────────────────────
def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Dunn Index = min inter-cluster centroid distance / max intra-cluster diameter.
    Higher values indicate better-separated, more compact clusters.
    """
    unique = np.unique(labels[labels != -1])
    if len(unique) < 2:
        return np.nan
    clusters  = [X[labels == lbl] for lbl in unique]
    centroids = np.array([c.mean(axis=0) for c in clusters])
    # Inter-cluster: minimum distance between any pair of centroids
    min_inter = np.inf
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = np.linalg.norm(centroids[i] - centroids[j])
            if d < min_inter:
                min_inter = d
    # Intra-cluster: max mean-diameter (2 × avg distance from centroid)
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

_METHOD_CFG = {
    # display-name : (dict-key, sklearn-metric, use-pca)
    "KMeans"      : ("kmeans",      "euclidean", True),
    "Agg"         : ("agg",         "cosine",    False),
    "GMM"         : ("gmm",         "euclidean", True),
    "SBScan"      : ("sbscan",      "euclidean", True),
    "POS"         : ("pos",         "euclidean", True),
    "RandAssign"  : ("rand_assign", "euclidean", True),
    "GSA"         : ("gsa",         "euclidean", True),
    "RandClust"   : ("rand_clust",  "euclidean", True),
}

def print_metrics(tr_norm, tr_pca, tr_pred, tr_lbl,
                  te_norm, te_pca, te_pred, te_lbl):
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

# ── VibClustNet diagnostic plots ──────────────────────────────────────────────
def plot_vibclustnet_diagnostics(model, te_X, te_pred_kmeans, device, figures_dir: Path):
    """
    Three diagnostic plots (saved to disk):
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
    fname = figures_dir / f"vcn_temporal_attention_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")

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
    fname = figures_dir / f"vcn_caim_heatmap_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")

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
    fname = figures_dir / f"vcn_reconstruction_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


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
    logger.info(f"  t-SNE [{tag}]...")
    ts = TSNE(n_components=2, random_state=42,
              perplexity=perp, max_iter=1000).fit_transform(pca_emb)
    return ts

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
    ncols, n_panels = 3, len(_ALL_METHODS) + 1          # +1 for ground truth
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
    # Hide any leftover axes
    for ax in axes_flat:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


def plot_tsne_individual(ts, pred, c2s, gt, figures_dir: Path):
    """Save one t-SNE figure per clustering method plus one for ground truth."""
    for key, title in _ALL_METHODS:
        fig, ax = plt.subplots(figsize=(8, 6))
        leg_fn  = (lambda lbl, k=key:
                   f"{super_name(c2s[k].get(lbl, '?'))} [c{lbl}]")
        _scatter(ax, ts, pred[key], leg_fn, f"t-SNE — {title}", "t-SNE")
        plt.tight_layout()
        fname = figures_dir / f"tsne_{title}_{RUN_TS}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {fname}")
    # Ground truth
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, sid in enumerate(sorted(np.unique(gt))):
        m = np.asarray(gt) == sid
        ax.scatter(ts[m, 0], ts[m, 1],
                   c=COLORS[i % len(COLORS)], alpha=0.6, s=8,
                   label=super_name(sid))
    ax.set_title("t-SNE — Ground Truth", fontsize=9, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best",
              framealpha=0.6, title="Super-class")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fname = figures_dir / f"tsne_GroundTruth_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


def plot_grid(ts, pred, c2s, gt, figures_dir: Path):
    fname = figures_dir / f"18_tsne_test_clusters_{RUN_TS}.png"
    _plot_proj_grid(ts, "t-SNE", pred, c2s, gt, fname=fname)
    plot_tsne_individual(ts, pred, c2s, gt, figures_dir)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    figures_dir = CONFIG["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=Path("logs"))

    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Run timestamp: {RUN_TS}")
    logger.info(f"Figures dir  : {figures_dir.resolve()}")
    logger.info(f"Models dir   : {CONFIG['models_dir'].resolve()}")

    logger.info("\n[0] Surface names")
    load_surface_names(CONFIG["surface_types_csv"])

    logger.info("\n[1] Load windowed pkl")
    tr_X, tr_y, te_X, te_y, _ = load_windowed_data(CONFIG)
    T = tr_X.shape[2]          # tr_X is (N, 3, T)

    logger.info("\n[2] Super-class distribution (train):")
    for sc, n in sorted(zip(*np.unique(tr_y, return_counts=True))):
        logger.info(f"    {super_name(sc):30s}: {n} windows")

    logger.info("\n[3] Train VibClustNet autoencoder")
    tr_ds = WindowedDataset(tr_X, tr_y)
    va_ds = WindowedDataset(te_X, te_y)
    tr_loader = DataLoader(tr_ds, batch_size=CONFIG["vcn_batch_size"],
                           shuffle=True,  drop_last=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=CONFIG["vcn_batch_size"],
                           shuffle=False, num_workers=0)
    # n_classes = max super-class index + 1 (handles non-contiguous indices like type B)
    n_cls_for_clf = max(SUPER_NAMES.keys()) + 1
    model = VibClustNet(T=T, emb_dim=CONFIG["vcn_embedding_dim"],
                        n_classes=n_cls_for_clf).to(device)
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = train_vibclustnet(model, tr_loader, va_loader, CONFIG, device,
                              X_labeled=tr_X, y_labeled=tr_y)

    logger.info("\n[3b] Save model")
    save_model(model, CONFIG)

    logger.info("\n[4] Embeddings + PCA")
    tr_emb = embed(model, tr_X, device)
    te_emb = embed(model, te_X, device)

    tr_norm, tr_pca, te_norm, te_pca, _ = pca_reduce(tr_emb, te_emb, CONFIG["pca_variance"])

    logger.info("\n[7] Clustering")
    n_surf = N_SUPER_CLASSES
    logger.info(f"  Super-classes={n_surf}  — sweeping K {max(2,n_surf-2)}..{n_surf+2}")
    k = best_k(tr_pca, max(2, n_surf-2), n_surf+2, figures_dir)
    tr_pred, te_pred, _ = cluster_all(tr_norm, tr_pca, te_norm, te_pca, k)

    tr_c2s = {m: c2s_map(tr_pred[m], tr_y) for m in tr_pred}
    te_c2s = {m: c2s_map(te_pred[m], te_y) for m in te_pred}

    logger.info("\n  K-Means cluster → surface (train):")
    for cid, sid in sorted(tr_c2s["kmeans"].items()):
        logger.info(f"    cluster {cid:2d} → {super_name(sid)}")
    logger.info("\n  SBScan cluster → surface (train):")
    for cid, sid in sorted(tr_c2s["sbscan"].items()):
        logger.info(f"    cluster {cid:2d} → {super_name(sid)}")

    logger.info("\n[8] Metrics")
    print_metrics(tr_norm, tr_pca, tr_pred, tr_y,
                  te_norm, te_pca, te_pred, te_y)

    logger.info("\n[9] Visualise (t-SNE on test split)")
    ts = project(te_pca, "Test")
    plot_grid(ts, te_pred, te_c2s, te_y, figures_dir)

    # Fixed-K experiment: K=3, 5, 7, 11
    experiment_fixed_k(te_pca, ts, figures_dir, ks=(3, 5, 7, 11))

    logger.info("\n[10] VibClustNet diagnostic plots")
    plot_vibclustnet_diagnostics(model, te_X, te_pred["kmeans"], device, figures_dir)

    logger.info("\nDone!")

if __name__ == "__main__":
    main()

"""
Road Surface Discovery — VibClustNet Fully Unsupervised Pipeline (Result Only)

Model      : VibClustNet from script 26 (variable n_channels, Upsample decoder, emb_dim=64)
Checkpoint : vibclustnet_best_26.pth  (no training — result-only mode)

Steps
-----
1. Load all windowed data (both CSVs)  →  GLOBAL Z-normalise  →  random 80/20 split
2. Load VibClustNet from checkpoint (script-26 model, trained with vib-aware loss)
3. Extract embeddings for ALL windows  →  PCA
4. Cluster (KMeans / Agglomerative / GMM / SBScan / PSO / RandomAssign / GSA / RandomClustering)
   with COSINE distance throughout  ← script-24 clustering settings
5. Evaluate with unsupervised metrics (Silhouette, DB, CH, Dunn, ARI/NMI vs KMeans pseudo-GT)
6. Visualise: t-SNE + UMAP  (grid + individual plots)
7. Fixed-K experiments K=3, 5, 7, 11  — t-SNE + UMAP
8. VibClustNet diagnostic plots (attention, CAIM, reconstruction)
9. Save cluster predictions to CSV

Multi-sensor support: sensor_cols in CONFIG controls which value columns are
loaded.  Set to e.g. ["valueX","valueY","valueZ","gyroX","gyroY","gyroZ"]
for a 6-channel setup.  The model adjusts its input channels automatically.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

# ── Logging setup ─────────────────────────────────────────────────────────────
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(log_dir: Path = Path(".")) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{RUN_TS}_26_cosine_results.log"
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("vcn")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path.resolve()}")
    return logger


logger = logging.getLogger("vcn")

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "output_csv"             : Path("26_cosine_results_unlabeled_predictions.csv"),

    # raw windowed data — both files are loaded and pooled
    "windowed_csv_1"         : Path("../../Datasets/ExtractedFeatures/labeled_accelerometer_raw_windows.csv"),
    "windowed_csv_2"         : Path("../../Datasets/ExtractedFeatures/unlabeled_accelerometer_raw_windows.csv"),

    # Multi-sensor: list every value column to use.
    # None → auto-detect all columns starting with "value"
    "sensor_cols"            : None,

    # split
    "test_size"              : 0.2,
    "seed"                   : 42,

    # PCA
    "pca_variance"           : 0.95,

    # VibClustNet (script-26 architecture)
    "vcn_embedding_dim"      : 64,
    "vcn_checkpoint"         : Path("vibclustnet_best_26.pth"),

    # Output directories
    "figures_dir"            : Path("figures"),
}

# ── Embeddings ────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed(model, X, device, bs=512):
    model.eval()
    out = [model.embed(torch.tensor(X[i:i+bs]).to(device)).cpu()
           for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# VibClustNet — script-26 architecture (variable n_channels, Upsample decoder)
# ══════════════════════════════════════════════════════════════════════════════

class MSTCB(nn.Module):
    """4-branch multi-scale conv block → fixed output channels."""
    def __init__(self, in_ch: int, out_ch: int = 96):
        super().__init__()
        b = out_ch // 4
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
        t_attn = self.temporal_gate(x)
        mag    = torch.fft.rfft(x, dim=-1).abs()
        f_gate = self.freq_mlp(mag).mean(dim=-1, keepdim=True)
        f_attn = f_gate.expand_as(x)
        return x * t_attn * f_attn, t_attn, f_attn


class CAIM(nn.Module):
    """Multi-head self-attention over N_ch pooled axis feature vectors."""
    def __init__(self, in_ch: int, n_axes: int, num_heads: int = 4):
        super().__init__()
        nh = num_heads if in_ch % num_heads == 0 else 1
        self.n_axes = n_axes
        self.attn   = nn.MultiheadAttention(in_ch, nh, batch_first=True)
        self.norm   = nn.LayerNorm(in_ch)

    def forward(self, axis_feats):
        pooled   = torch.stack([f.mean(dim=-1) for f in axis_feats], dim=1)   # (B, n_axes, CH)
        attn_out, weights = self.attn(pooled, pooled, pooled)
        attn_out = self.norm(attn_out + pooled)
        attended = [axis_feats[i] * attn_out[:, i, :].unsqueeze(-1)
                    for i in range(self.n_axes)]
        return attended, weights


class VibClustNet(nn.Module):
    """
    Autoencoder for multi-channel vibration windows (channels-first).
    Input / output: (batch, n_channels, T)

    Script-26 architecture: Upsample decoder, variable n_channels, emb_dim=64.
    """
    def __init__(self, T: int, n_channels: int = 3, emb_dim: int = 64):
        super().__init__()
        self.T          = T
        self.n_channels = n_channels
        CH = 96
        self.mstcb1   = MSTCB(1,              CH)
        self.mstcb2   = MSTCB(CH,             CH)
        self.caim     = CAIM(CH, n_axes=n_channels, num_heads=4)
        self.faag     = FAAG(n_channels * CH, T)
        self.mstcb3   = MSTCB(n_channels * CH, CH)
        self.enc_head = nn.Linear(CH, emb_dim)

        self.dec_proj = nn.Linear(emb_dim, CH * 4)
        self.rec_head = nn.Sequential(
            nn.Upsample(size=T, mode="linear", align_corners=False),
            nn.Conv1d(CH, CH, 7, padding="same"), nn.ReLU(),
            nn.Conv1d(CH, CH, 5, padding="same"), nn.ReLU(),
            nn.Conv1d(CH, CH // 2, 3, padding="same"), nn.ReLU(),
            nn.Conv1d(CH // 2, n_channels, 1),
        )

    def _ensure_channels_first(self, x):
        if x.shape[1] != self.n_channels:
            x = x.permute(0, 2, 1)
        return x

    def _encode(self, x):
        axes = [x[:, i:i+1, :] for i in range(self.n_channels)]
        axes = [self.mstcb2(self.mstcb1(a)) for a in axes]
        attended, caim_w = self.caim(axes)
        cat              = torch.cat(attended, dim=1)
        fout, t_attn, f_attn = self.faag(cat)
        after3           = self.mstcb3(fout)
        pooled = after3.mean(dim=-1)
        emb    = self.enc_head(pooled)
        enc_inter = {
            "after_mstcb12"  : torch.stack([a.mean(-1) for a in axes], dim=1),
            "after_mstcb3"   : pooled,
            "caim_weights"   : caim_w,
            "t_attn"         : t_attn,
            "f_attn"         : f_attn,
            "after3_temporal": after3,
        }
        return emb, enc_inter

    def _decode(self, emb, T):
        CH = 96
        h = self.dec_proj(emb)
        h = h.view(h.shape[0], CH, 4)
        recon = self.rec_head(h)
        dec_inter = {
            "d_after_upsample": h.mean(dim=-1),
            "d_after_conv1"   : h.mean(dim=-1),
        }
        return recon, dec_inter

    def forward(self, x):
        x = self._ensure_channels_first(x)
        T = x.shape[-1]
        emb, enc_inter = self._encode(x)
        recon, dec_inter = self._decode(emb, T)
        return recon, emb, enc_inter, dec_inter

    def embed(self, x):
        x = self._ensure_channels_first(x)
        emb, _ = self._encode(x)
        return F.normalize(emb, dim=-1)


# ── Data loading ──────────────────────────────────────────────────────────────
def _detect_sensor_cols(df: pd.DataFrame, sensor_cols) -> list:
    if sensor_cols is not None:
        missing = [c for c in sensor_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Sensor columns not found in CSV: {missing}")
        return list(sensor_cols)
    detected = sorted(c for c in df.columns if c.lower().startswith("value"))
    if not detected:
        raise ValueError("No 'value*' columns found. Set sensor_cols in CONFIG.")
    logger.info(f"  Auto-detected sensor columns: {detected}")
    return detected


def _load_windows_raw(csv_path: Path, sensor_cols) -> tuple:
    """Load windowed CSV → raw (N, C, T) array WITHOUT normalisation."""
    logger.info(f"  Loading: {csv_path}")
    raw = pd.read_csv(csv_path)
    logger.info(f"  Rows: {len(raw)}  Windows: {raw['window_id'].nunique()}")
    cols = _detect_sensor_cols(raw, sensor_cols)
    windows = []
    for _, group in raw.groupby("window_id", sort=True):
        win = group[cols].to_numpy(dtype=np.float32).T   # (C, T)
        windows.append(win)
    arr = np.stack(windows).astype(np.float32)            # (N, C, T)
    return arr, cols


def load_all_data(cfg):
    """
    Load both CSVs, pool all windows, apply GLOBAL Z-normalisation,
    then random 80/20 split.  No labels are read or used.

    Returns:
        tr_X       (N_tr, C, T)
        te_X       (N_te, C, T)
        all_X      (N,    C, T)  — full pool in original order
        n_channels  int
    """
    arr1, cols = _load_windows_raw(cfg["windowed_csv_1"], cfg["sensor_cols"])
    arr2, _    = _load_windows_raw(cfg["windowed_csv_2"], cfg["sensor_cols"])
    all_X = np.concatenate([arr1, arr2], axis=0)
    logger.info(f"  Total windows: {len(all_X)}  Channels: {all_X.shape[1]}  "
                f"T: {all_X.shape[2]}  Cols: {cols}")

    g_mean = float(all_X.mean())
    g_std  = float(all_X.std().clip(1e-8))
    all_X  = (all_X - g_mean) / g_std
    logger.info(f"  Global norm  mean={g_mean:.4f}  std={g_std:.4f}")

    rng   = np.random.default_rng(cfg["seed"])
    idx   = rng.permutation(len(all_X))
    n_te  = max(1, int(len(all_X) * cfg["test_size"]))
    te_idx, tr_idx = idx[:n_te], idx[n_te:]
    logger.info(f"  Train: {len(tr_idx)}  Test: {len(te_idx)}")
    return all_X[tr_idx], all_X[te_idx], all_X, all_X.shape[1]


# ── PCA reduction ─────────────────────────────────────────────────────────────
def pca_reduce(tr_emb, other_emb, variance=0.95):
    """Fit PCA on L2-norm of tr_emb, apply to both."""
    tr_norm    = normalize(tr_emb, norm="l2")
    other_norm = normalize(other_emb, norm="l2")
    pca = PCA(n_components=variance, random_state=42)
    tr_pca    = pca.fit_transform(tr_norm)
    other_pca = pca.transform(other_norm)
    logger.info(f"  PCA: {tr_emb.shape[1]} → {tr_pca.shape[1]} dims  "
                f"({100*variance:.0f}% variance)")
    return tr_norm, tr_pca, other_norm, other_pca, pca


# ── Clustering algorithms (cosine distance — script-24 settings) ──────────────
class SBScanClustering:
    """DBSCAN with automatic epsilon selection via k-NN cosine distance elbow."""
    def __init__(self, n_clusters_hint: int = 5, min_samples: int = 5):
        self.n_clusters_hint = n_clusters_hint
        self.min_samples     = min_samples
        self.labels_         = None
        self.eps_            = None
        self._knn_clf        = None

    def fit(self, X):
        k        = self.min_samples
        nbrs     = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
        dists, _ = nbrs.kneighbors(X)
        k_dists  = np.sort(dists[:, -1])
        n        = len(k_dists)
        xs = np.linspace(0.0, 1.0, n)
        rng_d = k_dists[-1] - k_dists[0]
        ys = (k_dists - k_dists[0]) / (rng_d + 1e-10)
        knee_idx  = int(np.argmax(np.abs(ys - xs)))
        self.eps_ = float(k_dists[knee_idx])
        if self.eps_ < 1e-6:
            self.eps_ = float(np.median(k_dists))
        logger.info(f"    SBScan auto-eps={self.eps_:.4f}  (knee idx={knee_idx}/{n})")
        db = DBSCAN(eps=self.eps_, min_samples=self.min_samples, metric="cosine").fit(X)
        self.labels_ = db.labels_
        mask    = self.labels_ != -1
        n_valid = int(mask.sum())
        n_cls   = len(np.unique(self.labels_[mask])) if n_valid > 0 else 0
        logger.info(f"    SBScan: {n_cls} clusters, {n_valid} non-noise / {n} total")
        if n_valid > 1 and n_cls > 1:
            self._knn_clf = KNeighborsClassifier(
                n_neighbors=min(5, n_valid), metric="cosine"
            ).fit(X[mask], self.labels_[mask])
        return self

    def predict(self, X):
        if self._knn_clf is None:
            return np.zeros(len(X), dtype=int)
        return self._knn_clf.predict(X)


class PSOClustering:
    """Particle Swarm Optimisation clustering (minimise within-cluster cosine distance)."""
    def __init__(self, n_clusters: int = 5, n_particles: int = 15,
                 max_iter: int = 50, seed: int = 42):
        self.n_clusters  = n_clusters
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.seed        = seed
        self.centroids_  = None
        self.labels_     = None

    def _assign(self, X, centroids):
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        C_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        return (1.0 - X_n @ C_n.T).argmin(axis=1)

    def _wcss(self, X, centroids):
        asgn = self._assign(X, centroids)
        total = 0.0
        for k in range(len(centroids)):
            mask = asgn == k
            if not mask.any():
                continue
            X_n = X[mask] / (np.linalg.norm(X[mask], axis=1, keepdims=True) + 1e-10)
            c_n = centroids[k] / (np.linalg.norm(centroids[k]) + 1e-10)
            total += (1.0 - X_n @ c_n).sum()
        return total

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
    """Gravitational Search Algorithm (GSA) clustering (cosine distance)."""
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
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        C_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        return (1.0 - X_n @ C_n.T).argmin(axis=1)

    def _fitness(self, X, centroids):
        asgn = self._assign(X, centroids)
        total = 0.0
        for k in range(len(centroids)):
            mask = asgn == k
            if not mask.any():
                continue
            X_n = X[mask] / (np.linalg.norm(X[mask], axis=1, keepdims=True) + 1e-10)
            c_n = centroids[k] / (np.linalg.norm(centroids[k]) + 1e-10)
            total += (1.0 - X_n @ c_n).sum()
        return total

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
    """Random centroid initialisation + cosine nearest-neighbour assignment (no iteration)."""
    def __init__(self, n_clusters: int = 5, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed       = seed
        self.centroids_ = None
        self.labels_    = None

    def _assign(self, X, centroids):
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        C_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        return (1.0 - X_n @ C_n.T).argmin(axis=1)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        idxs = rng.choice(len(X), size=self.n_clusters, replace=False)
        self.centroids_ = X[idxs]
        self.labels_    = self._assign(X, self.centroids_)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


# ── K selection ───────────────────────────────────────────────────────────────
def best_k(pca_emb, k_min, k_max, figures_dir: Path):
    cos_emb = normalize(pca_emb, norm="l2")
    scores = {}
    for k in range(k_min, k_max + 1):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(cos_emb)
        scores[k] = silhouette_score(cos_emb, lbl, metric="cosine")
        logger.info(f"    K={k:2d}  sil={scores[k]:.4f}")
    bk = max(scores, key=scores.get)
    logger.info(f"  Best K={bk}  sil={scores[bk]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(list(scores.keys()), list(scores.values()), "o-", color="#457b9d")
    ax.axvline(bk, color="#e63946", ls="--", label=f"K={bk}")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette")
    ax.set_title("K selection (cosine)"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = figures_dir / f"17_k_selection_{RUN_TS}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    logger.info(f"  Saved {fname}")
    return bk


# ── Fixed-K experiment ────────────────────────────────────────────────────────
def _fixed_k_grid(pca_emb, proj_coords, proj_name, fname, ks):
    """2×2 grid of KMeans at fixed K values, coloured by cluster only."""
    assert len(ks) == 4, "Need exactly 4 K values for 2×2 grid"
    cos_emb = normalize(pca_emb, norm="l2")
    rows = {}
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Fixed-K experiment — {proj_name} (all windows)",
                 fontsize=13, fontweight="bold")
    for ax, k in zip(axes.flat, ks):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(cos_emb)
        rows[f"K={k}"] = dict(
            Silhouette = silhouette_score(cos_emb, lbl, metric="cosine"),
            DB         = davies_bouldin_score(cos_emb, lbl),
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


def _print_quality(rows):
    logger.info(f"  {'K':<8} {'Silhouette':>12} {'Davies-Bouldin':>16}  Quality")
    logger.info("  " + "-" * 46)
    for k, r in rows.items():
        sil, db = r["Silhouette"], r["DB"]
        q = "Excellent" if sil > 0.6 else "Good" if sil > 0.4 else "Fair" if sil > 0.2 else "Poor"
        logger.info(f"  {k:<8} {sil:>12.4f} {db:>16.4f}  {q}")


def experiment_fixed_k(pca_emb, ts, um, figures_dir: Path, ks=(3, 5, 7, 11)):
    logger.info(f"\n  Fixed-K experiment  K={list(ks)}")
    fname_ts = figures_dir / f"20_fixed_k_tsne_{RUN_TS}.png"
    rows_ts = _fixed_k_grid(pca_emb, ts, "t-SNE", fname_ts, ks)
    logger.info("\n  Cluster quality — t-SNE projection:")
    _print_quality(rows_ts)

    fname_um = figures_dir / f"20_fixed_k_umap_{RUN_TS}.png"
    rows_um = _fixed_k_grid(pca_emb, um, "UMAP", fname_um, ks)
    logger.info("\n  Cluster quality — UMAP projection:")
    _print_quality(rows_um)

    return pd.DataFrame(rows_ts).T


# ── Clustering wrapper ────────────────────────────────────────────────────────
def cluster_all(tr_norm, tr_pca, te_norm, te_pca, all_norm, all_pca, k):
    """
    Fit all 8 clusterers on L2-normalised PCA of training split.
    Predict on test split and full dataset.
    Cosine distance throughout (matches script-24 settings).
    """
    tr_cos  = normalize(tr_pca, norm="l2")
    te_cos  = normalize(te_pca, norm="l2")
    all_cos = normalize(all_pca, norm="l2")

    km      = KMeans(n_clusters=k, random_state=42, n_init=20).fit(tr_cos)
    agg     = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(tr_norm)
    gmm     = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(tr_cos)
    sbscan  = SBScanClustering(n_clusters_hint=k, min_samples=5).fit(tr_cos)
    pso     = PSOClustering(n_clusters=k, seed=42).fit(tr_cos)
    rand_a  = RandomAssignClustering(n_clusters=k, seed=42).fit(tr_cos)
    gsa     = GravitationalSearchClustering(n_clusters=k, seed=42).fit(tr_cos)
    rand_c  = RandomClustering(n_clusters=k, seed=42).fit(tr_cos)

    agg_centroids = np.vstack([tr_norm[agg.labels_ == c].mean(axis=0)
                               for c in range(k)])
    agg_centroids = normalize(agg_centroids, norm="l2")

    def _agg_predict(X_norm):
        X_n = X_norm / (np.linalg.norm(X_norm, axis=1, keepdims=True) + 1e-10)
        return (1.0 - X_n @ agg_centroids.T).argmin(axis=1)

    tr_pred = {
        "kmeans"     : km.labels_,
        "agg"        : agg.labels_,
        "gmm"        : gmm.predict(tr_cos),
        "sbscan"     : sbscan.labels_,
        "pos"        : pso.labels_,
        "rand_assign": rand_a.labels_,
        "gsa"        : gsa.labels_,
        "rand_clust" : rand_c.labels_,
    }

    def _predict_on(cos_emb, norm_emb, ra_seed, rc_seed):
        return {
            "kmeans"     : km.predict(cos_emb),
            "agg"        : _agg_predict(norm_emb),
            "gmm"        : gmm.predict(cos_emb),
            "sbscan"     : sbscan.predict(cos_emb),
            "pos"        : pso.predict(cos_emb),
            "rand_assign": RandomAssignClustering(n_clusters=k, seed=ra_seed).fit(cos_emb).labels_,
            "gsa"        : gsa.predict(cos_emb),
            "rand_clust" : RandomClustering(n_clusters=k, seed=rc_seed).fit(cos_emb).labels_,
        }

    te_pred  = _predict_on(te_cos,  te_norm,  99, 99)
    all_pred = _predict_on(all_cos, all_norm, 77, 77)

    return tr_pred, te_pred, all_pred, km


# ── Evaluation ────────────────────────────────────────────────────────────────
def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
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


def evaluate_unsupervised(emb, pred, pseudo_gt=None, metric="cosine"):
    """Unsupervised metrics + optional ARI/NMI vs a pseudo ground truth (e.g. KMeans)."""
    pred = np.asarray(pred); ok = pred != -1
    ev, lv = emb[ok], pred[ok]
    if len(np.unique(lv)) < 2:
        return dict(Silhouette=np.nan, DB=np.nan, CH=np.nan, Dunn=np.nan,
                    ARI_vs_KM=np.nan, NMI_vs_KM=np.nan)
    result = dict(
        Silhouette = silhouette_score(ev, lv, metric=metric),
        DB         = davies_bouldin_score(ev, lv),
        CH         = calinski_harabasz_score(ev, lv),
        Dunn       = dunn_index(ev, lv),
        ARI_vs_KM  = np.nan,
        NMI_vs_KM  = np.nan,
    )
    if pseudo_gt is not None:
        gv = np.asarray(pseudo_gt)[ok]
        result["ARI_vs_KM"] = adjusted_rand_score(gv, lv)
        result["NMI_vs_KM"] = normalized_mutual_info_score(gv, lv)
    return result


_METHOD_CFG = {
    "KMeans"      : ("kmeans",      "cosine", True),
    "Agg"         : ("agg",         "cosine", False),
    "GMM"         : ("gmm",         "cosine", True),
    "SBScan"      : ("sbscan",      "cosine", True),
    "PSO"         : ("pos",         "cosine", True),
    "RandAssign"  : ("rand_assign", "cosine", True),
    "GSA"         : ("gsa",         "cosine", True),
    "RandClust"   : ("rand_clust",  "cosine", True),
}


def print_metrics(all_norm, all_pca, all_pred):
    """Print unsupervised metrics with KMeans pseudo-GT for all methods."""
    km_ref = all_pred["kmeans"]
    rows = {}
    for name, (key, metric, use_pca) in _METHOD_CFG.items():
        emb = all_pca if use_pca else all_norm
        rows[name] = evaluate_unsupervised(emb, all_pred[key],
                                           pseudo_gt=km_ref, metric=metric)
    logger.info("\n── ALL DATA (unsupervised + ARI/NMI vs KMeans pseudo-GT) ──────────")
    logger.info("\n" + pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))


# ── Save predictions ──────────────────────────────────────────────────────────
def save_predictions(all_pred, cfg):
    n  = len(all_pred["kmeans"])
    df = pd.DataFrame({"window_idx": np.arange(n)})
    for key in all_pred:
        df[f"cluster_{key}"] = all_pred[key]
    out = cfg["output_csv"]
    df.to_csv(out, index=False)
    logger.info(f"  Predictions saved: {Path(out).resolve()}  ({n} windows)")


# ── VibClustNet diagnostic plots ──────────────────────────────────────────────
def plot_vibclustnet_diagnostics(model, X, pred_kmeans, device, figures_dir: Path):
    """
    Three diagnostic plots:
      1. Average temporal attention profile per cluster
      2. CAIM cross-axis attention heatmap per cluster
      3. Original vs reconstructed signal for 3 random samples
    """
    n_channels = model.n_channels
    model.eval()
    all_t_attn, all_caim_w = [], []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i+bs]).to(device)
            xb = model._ensure_channels_first(xb)
            _, enc_inter = model._encode(xb)
            all_t_attn.append(enc_inter["t_attn"].cpu().numpy())
            all_caim_w.append(enc_inter["caim_weights"].cpu().numpy())

    all_t_attn = np.concatenate(all_t_attn, axis=0)
    all_caim_w = np.concatenate(all_caim_w, axis=0)
    clusters   = sorted(set(pred_kmeans))
    n_cls      = len(clusters)

    # 1. Temporal attention per cluster
    fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 3), squeeze=False)
    fig.suptitle("VibClustNet — Avg Temporal Attention per Cluster", fontweight="bold")
    for ax, cid in zip(axes[0], clusters):
        mask = pred_kmeans == cid
        avg  = all_t_attn[mask].mean(axis=(0, 1))
        ax.plot(avg, color=COLORS[cid % len(COLORS)], linewidth=1.0)
        ax.set_title(f"Cluster {cid}", fontsize=9)
        ax.set_xlabel("Time step"); ax.set_ylabel("Attention")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = figures_dir / f"vcn_temporal_attention_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")

    # 2. CAIM cross-axis heatmap per cluster
    axis_labels = [f"ch{i}" for i in range(n_channels)]
    fig, axes = plt.subplots(1, n_cls, figsize=(3.5 * n_cls, 3.5), squeeze=False)
    fig.suptitle("VibClustNet — CAIM Cross-Axis Attention per Cluster", fontweight="bold")
    for ax, cid in zip(axes[0], clusters):
        mask  = pred_kmeans == cid
        avg_w = all_caim_w[mask].mean(axis=0)
        ticks = list(range(n_channels))
        im    = ax.imshow(avg_w, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(ticks); ax.set_xticklabels(axis_labels, fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels(axis_labels, fontsize=7)
        ax.set_title(f"Cluster {cid}", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fname = figures_dir / f"vcn_caim_heatmap_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")

    # 3. Reconstruction for 3 random samples
    rng  = np.random.default_rng(42)
    idxs = rng.choice(len(X), size=min(3, len(X)), replace=False)
    fig, axes = plt.subplots(len(idxs), n_channels,
                             figsize=(5 * n_channels, 4 * len(idxs)), squeeze=False)
    fig.suptitle("VibClustNet — Reconstruction (random samples)", fontweight="bold")
    with torch.no_grad():
        for row, idx in enumerate(idxs):
            xb = torch.tensor(X[idx:idx+1]).to(device)
            xb = model._ensure_channels_first(xb)
            recon, _, _, _ = model(xb)
            orig = xb[0].cpu().numpy()
            rec  = recon[0].cpu().numpy()
            for col in range(n_channels):
                ax = axes[row, col]
                ax.plot(orig[col], label="Original",      alpha=0.85, linewidth=0.8)
                ax.plot(rec[col],  label="Reconstructed", alpha=0.85, linewidth=0.8, ls="--")
                ax.set_title(f"Sample {idx} — {axis_labels[col]}", fontsize=9)
                ax.set_xlabel("Time"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = figures_dir / f"vcn_reconstruction_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


# ── Visualisation ─────────────────────────────────────────────────────────────
COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
          "#8338ec", "#06d6a0", "#fb8500", "#3a86ff", "#ff006e",
          "#c77dff", "#80b918"]

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


def _scatter(ax, xy, lbls, title, proj_name):
    for i, cid in enumerate(sorted(set(lbls))):
        m = np.asarray(lbls) == cid
        ax.scatter(xy[m, 0], xy[m, 1],
                   c="#bbb" if cid == -1 else COLORS[i % len(COLORS)],
                   alpha=0.6, s=8, label=f"Cluster {cid}")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6)
    ax.grid(alpha=0.2)


def project(pca_emb, tag):
    perp = min(30, len(pca_emb) - 1)
    logger.info(f"  t-SNE [{tag}]...")
    return TSNE(n_components=2, random_state=42,
                perplexity=perp, max_iter=1000).fit_transform(pca_emb)


def project_umap(pca_emb, tag):
    logger.info(f"  UMAP [{tag}]...")
    return UMAP(n_components=2, random_state=42,
                n_neighbors=15, min_dist=0.1).fit_transform(pca_emb)


def _plot_grid_unsupervised(coords, proj_name, pred, fname):
    """3×3 grid of all 8 clustering methods — no ground-truth panel."""
    ncols, n_panels = 3, len(_ALL_METHODS)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 6))
    fig.suptitle(f"{proj_name} — All windows (unsupervised, script-26 model)",
                 fontsize=13, fontweight="bold")
    axes_flat = iter(axes.flat)
    for key, title in _ALL_METHODS:
        ax = next(axes_flat)
        _scatter(ax, coords, pred[key], f"{proj_name} — {title}", proj_name)
    for ax in axes_flat:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


def plot_individual(coords, proj_name, prefix, pred, figures_dir: Path):
    """Save one figure per clustering method."""
    for key, title in _ALL_METHODS:
        fig, ax = plt.subplots(figsize=(8, 6))
        _scatter(ax, coords, pred[key], f"{proj_name} — {title}", proj_name)
        plt.tight_layout()
        fname = figures_dir / f"{prefix}_{title}_{RUN_TS}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_stem = Path(__file__).stem
    figures_dir = CONFIG["figures_dir"] / f"{script_stem}_{RUN_TS}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=Path("logs"))

    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Run timestamp: {RUN_TS}")
    logger.info(f"Figures dir  : {figures_dir.resolve()}")

    # ── [1] Load all data ────────────────────────────────────────────────────
    logger.info("\n[1] Load all windowed data (no labels) — global normalisation")
    tr_X, te_X, all_X, n_channels = load_all_data(CONFIG)
    T = all_X.shape[2]
    logger.info(f"  Window length T={T}  Channels={n_channels}")

    # ── [2] Load VibClustNet from checkpoint ─────────────────────────────────
    logger.info("\n[2] Load VibClustNet (script-26 model) from checkpoint")
    ckpt = CONFIG["vcn_checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt.resolve()}")
    model = VibClustNet(T=T, n_channels=n_channels,
                        emb_dim=CONFIG["vcn_embedding_dim"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    logger.info(f"  Loaded: {ckpt.resolve()}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── [3] Embeddings + PCA ─────────────────────────────────────────────────
    logger.info("\n[3] Extract embeddings + PCA")
    tr_emb  = embed(model, tr_X,  device)
    te_emb  = embed(model, te_X,  device)
    all_emb = embed(model, all_X, device)
    tr_norm, tr_pca, te_norm, te_pca, pca_obj = pca_reduce(
        tr_emb, te_emb, CONFIG["pca_variance"])
    all_norm = normalize(all_emb, norm="l2")
    all_pca  = pca_obj.transform(all_norm)
    logger.info(f"  Train PCA: {tr_pca.shape}  All PCA: {all_pca.shape}")

    # ── [4] Select K (cosine silhouette on train split) ──────────────────────
    logger.info("\n[4] Select K (cosine silhouette sweep on train embeddings, K=3..12)")
    k = best_k(tr_pca, k_min=3, k_max=12, figures_dir=figures_dir)

    # ── [5] Clustering (cosine distance, 8 methods) ──────────────────────────
    logger.info(f"\n[5] Clustering  K={k}  (cosine distance, 8 methods)")
    tr_pred, te_pred, all_pred, _ = cluster_all(
        tr_norm, tr_pca, te_norm, te_pca, all_norm, all_pca, k)

    # ── [6] Metrics ──────────────────────────────────────────────────────────
    logger.info("\n[6] Unsupervised metrics (all windows)")
    print_metrics(all_norm, all_pca, all_pred)

    # ── [7] Save predictions ─────────────────────────────────────────────────
    logger.info("\n[7] Save predictions")
    save_predictions(all_pred, CONFIG)

    # ── [8] Visualise (t-SNE + UMAP on all windows) ──────────────────────────
    logger.info("\n[8] t-SNE + UMAP visualisation (all windows)")
    # Sub-sample if large to keep projections tractable
    max_proj = 5000
    sidx = None
    if len(all_pca) > max_proj:
        rng  = np.random.default_rng(CONFIG["seed"])
        sidx = rng.choice(len(all_pca), size=max_proj, replace=False)
        vis_pca  = all_pca[sidx]
        vis_pred = {k: v[sidx] for k, v in all_pred.items()}
        logger.info(f"  Sub-sampled {max_proj}/{len(all_pca)} windows for projections")
    else:
        vis_pca, vis_pred = all_pca, all_pred

    ts = project(vis_pca, "All")
    um = project_umap(vis_pca, "All")

    fname_ts = figures_dir / f"19_tsne_all_clusters_{RUN_TS}.png"
    _plot_grid_unsupervised(ts, "t-SNE", vis_pred, fname_ts)
    plot_individual(ts, "t-SNE", "tsne", vis_pred, figures_dir)

    fname_um = figures_dir / f"19_umap_all_clusters_{RUN_TS}.png"
    _plot_grid_unsupervised(um, "UMAP", vis_pred, fname_um)
    plot_individual(um, "UMAP", "umap", vis_pred, figures_dir)

    # ── [9] Fixed-K experiments (t-SNE + UMAP) ───────────────────────────────
    logger.info("\n[9] Fixed-K experiments  K=(3, 5, 7, 11)")
    experiment_fixed_k(vis_pca, ts, um, figures_dir, ks=(3, 5, 7, 11))

    # ── [10] VibClustNet diagnostic plots ────────────────────────────────────
    logger.info("\n[10] VibClustNet diagnostic plots")
    diag_X    = all_X[sidx] if sidx is not None else all_X
    diag_pred = vis_pred["kmeans"]
    plot_vibclustnet_diagnostics(model, diag_X, diag_pred, device, figures_dir)

    logger.info(f"\nDone!  All plots saved to: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()

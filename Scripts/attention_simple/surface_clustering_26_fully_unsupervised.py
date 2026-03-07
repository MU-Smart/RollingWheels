"""
Road Surface Discovery — VibClustNet Fully Unsupervised Pipeline

All data (labeled + unlabeled CSVs) is pooled and treated as raw vibration
windows.  No surface_id, class labels, or any ground-truth signal is used
at any stage of training, clustering, or evaluation.

Steps
-----
1. Load all windowed data (both CSVs)  →  Z-normalise  →  random 80/20 split
2. Train VibClustNet autoencoder (reconstruction loss only, no clf head)
3. Extract embeddings for ALL windows  →  PCA
4. Cluster (KMeans / Agglomerative / GMM / SBScan / PSO / GSA) on PCA embeddings
5. Evaluate with unsupervised metrics only (Silhouette, DB, CH, Dunn)
6. Visualise: t-SNE  coloured by cluster assignment
7. VibClustNet diagnostic plots (attention, CAIM, reconstruction)
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
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{RUN_TS}_26_unsup.log"
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

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "output_csv"             : Path("26_unsupervised_predictions.csv"),

    # raw windowed data — both files are loaded and pooled
    "windowed_csv_1"         : Path("../../Datasets/ExtractedFeatures/labeled_accelerometer_raw_windows.csv"),
    "windowed_csv_2"         : Path("../../Datasets/ExtractedFeatures/unlabeled_accelerometer_raw_windows.csv"),

    # split
    "test_size"              : 0.2,
    "seed"                   : 42,

    # PCA
    "pca_variance"           : 0.95,

    # VibClustNet hyper-parameters
    "vcn_epochs"             : 50,
    "vcn_batch_size"         : 32,
    "vcn_lr"                 : 1e-3,
    "vcn_patience"           : 50,
    "vcn_embedding_dim"      : 128,
    "vcn_checkpoint"         : Path("vibclustnet_best_26.pth"),

    # Output directories
    "figures_dir"            : Path("figures"),
    "models_dir"             : Path("models"),
}

# ── Embeddings ────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed(model, X, device, bs=512):
    model.eval()
    out = [model.embed(torch.tensor(X[i:i+bs]).to(device)).cpu()
           for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# VibClustNet — raw-window autoencoder (reconstruction only, no clf head)
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
    """Multi-head self-attention over 3 pooled axis feature vectors."""
    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        nh = num_heads if in_ch % num_heads == 0 else 1
        self.attn = nn.MultiheadAttention(in_ch, nh, batch_first=True)
        self.norm = nn.LayerNorm(in_ch)

    def forward(self, axis_feats):
        pooled   = torch.stack([f.mean(dim=-1) for f in axis_feats], dim=1)
        attn_out, weights = self.attn(pooled, pooled, pooled)
        attn_out = self.norm(attn_out + pooled)
        attended = [axis_feats[i] * attn_out[:, i, :].unsqueeze(-1) for i in range(3)]
        return attended, weights


class VibClustNet(nn.Module):
    """
    Autoencoder for 3-axis vibration windows (channels-first).
    Input / output: (batch, 3, T)
    Pure reconstruction objective — no classification head.
    """
    def __init__(self, T: int, emb_dim: int = 256):
        super().__init__()
        self.T = T
        CH = 96
        self.mstcb1   = MSTCB(1,      CH)
        self.mstcb2   = MSTCB(CH,     CH)
        self.caim     = CAIM(CH, num_heads=4)
        self.faag     = FAAG(3 * CH, T)
        self.mstcb3   = MSTCB(3 * CH, CH)
        self.enc_head = nn.Linear(CH, emb_dim)
        # Decoder: projects embedding back to CH channels, then reconstructs
        # through time via conv.  Forces ALL information through the bottleneck.
        self.dec_proj = nn.Linear(emb_dim, CH)
        self.rec_head = nn.Sequential(
            nn.Conv1d(CH, CH, 7, padding="same"), nn.ReLU(),
            nn.Conv1d(CH, CH, 3, padding="same"), nn.ReLU(),
            nn.Conv1d(CH, CH // 2, 3, padding="same"), nn.ReLU(),
            nn.Conv1d(CH // 2, 3, 1),
        )

    def _encode(self, x):
        axes = [x[:, i:i+1, :] for i in range(3)]
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
        h = self.dec_proj(emb)                  # (batch, CH)
        h = h.unsqueeze(-1).expand(-1, -1, T)   # (batch, CH, T)
        recon = self.rec_head(h)                 # (batch, 3, T)
        dec_inter = {
            "d_after_upsample": h.mean(dim=-1),
            "d_after_conv1"   : h.mean(dim=-1),
        }
        return recon, dec_inter

    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)
        T = x.shape[-1]
        emb, enc_inter = self._encode(x)
        recon, dec_inter = self._decode(emb, T)
        return recon, emb, enc_inter, dec_inter

    def embed(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1)
        emb, _ = self._encode(x)
        return F.normalize(emb, dim=-1)


# ── Log-cosh reconstruction loss ──────────────────────────────────────────────
def multi_rec_loss(x_orig, recon):
    if x_orig.shape[1] != 3:
        x_orig = x_orig.permute(0, 2, 1)
    d   = recon - x_orig
    ln2 = torch.tensor(2.0, device=d.device).log()
    return (torch.logaddexp(d, -d) - ln2).sum() / d.numel()


# ── Windowed dataset ───────────────────────────────────────────────────────────
class WindowedDataset(torch.utils.data.Dataset):
    """Dataset wrapping a (N, 3, T) float32 numpy array — no labels."""
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# ── Load windowed csv ──────────────────────────────────────────────────────────
def _load_windows(csv_path: Path) -> np.ndarray:
    """Load a windowed CSV and return (N, 3, T) Z-normalised array.
    Only uses window_id + valueX/Y/Z columns; all other columns ignored."""
    logger.info(f"  Loading: {csv_path}")
    raw = pd.read_csv(csv_path)
    logger.info(f"  Rows: {len(raw)}  Windows: {raw['window_id'].nunique()}")
    windows = []
    for _, group in raw.groupby("window_id", sort=True):
        xyz = group[["valueX", "valueY", "valueZ"]].to_numpy(dtype=np.float32).T  # (3, T)
        windows.append(xyz)
    arr = np.stack(windows).astype(np.float32)   # (N, 3, T)
    mu  = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1,  keepdims=True).clip(1e-8)
    return (arr - mu) / std


def load_all_data(cfg):
    """
    Load both CSVs, pool all windows, random 80/20 split.
    No labels are read or used.

    Returns:
        tr_X  (N_tr, 3, T)
        te_X  (N_te, 3, T)
        all_X (N,    3, T)  — full pool in original order
    """
    arr1 = _load_windows(cfg["windowed_csv_1"])
    arr2 = _load_windows(cfg["windowed_csv_2"])
    all_X = np.concatenate([arr1, arr2], axis=0)
    logger.info(f"  Total windows: {len(all_X)}")

    rng   = np.random.default_rng(cfg["seed"])
    idx   = rng.permutation(len(all_X))
    n_te  = max(1, int(len(all_X) * cfg["test_size"]))
    te_idx, tr_idx = idx[:n_te], idx[n_te:]
    logger.info(f"  Train: {len(tr_idx)}  Test: {len(te_idx)}")
    return all_X[tr_idx], all_X[te_idx], all_X


# ── VibClustNet training ───────────────────────────────────────────────────────
def train_vibclustnet(model, train_loader, val_loader, cfg, device):
    """
    Pure reconstruction training.
    - Early stopping on val reconstruction loss (patience = cfg["vcn_patience"])
    - Gradient clipping max_norm=5.0
    - Saves / reloads best checkpoint
    """
    opt  = torch.optim.Adam(model.parameters(), lr=cfg["vcn_lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-5)
    best = float("inf")
    pat  = 0
    ckpt = cfg["vcn_checkpoint"]

    for ep in range(cfg["vcn_epochs"]):
        model.train()
        tr_rec = 0.0
        for xb in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, _, _, _ = model(xb)
            loss = multi_rec_loss(xb, recon)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_rec += loss.item()
        tr_rec /= len(train_loader)

        model.eval()
        va_rec = 0.0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device)
                recon, _, _, _ = model(xb)
                va_rec += multi_rec_loss(xb, recon).item()
        va_rec /= len(val_loader)

        sched.step(va_rec)
        imp  = va_rec < best - 1e-6
        best, pat = (va_rec, 0) if imp else (best, pat + 1)
        if imp:
            torch.save(model.state_dict(), ckpt)
        lr_now = opt.param_groups[0]["lr"]
        logger.info(f"  [VCN] {ep+1:3d}/{cfg['vcn_epochs']}  "
                    f"rec={tr_rec:.4f}  val_rec={va_rec:.4f}  "
                    f"lr={lr_now:.2e}" + (" *" if imp else ""))

        if pat >= cfg["vcn_patience"]:
            logger.info(f"  Early stop at epoch {ep+1}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def save_model(model, cfg):
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
def pca_reduce(emb_tr, emb_all, variance=0.95):
    """Fit PCA on train embeddings, transform both train and all-data."""
    n_tr  = normalize(emb_tr,  norm="l2")
    n_all = normalize(emb_all, norm="l2")
    pca   = PCA(n_components=variance, svd_solver="full").fit(n_tr)
    logger.info(f"  PCA: {emb_tr.shape[1]}d → {pca.n_components_}d "
                f"(var={pca.explained_variance_ratio_.sum():.3f})")
    return n_tr, pca.transform(n_tr), n_all, pca.transform(n_all), pca


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


# ── Clustering ────────────────────────────────────────────────────────────────
def best_k(pca_emb, k_min, k_max, figures_dir: Path):
    """Select K via silhouette score sweep."""
    scores = {}
    for k in range(k_min, k_max + 1):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(pca_emb)
        scores[k] = silhouette_score(pca_emb, lbl)
        logger.info(f"    K={k:2d}  sil={scores[k]:.4f}")
    bk = max(scores, key=scores.get)
    logger.info(f"  Best K={bk}  sil={scores[bk]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(list(scores.keys()), list(scores.values()), "o-", color="#457b9d")
    ax.axvline(bk, color="#e63946", ls="--", label=f"K={bk}")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette")
    ax.set_title("K selection (silhouette)"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = figures_dir / f"26_k_selection_{RUN_TS}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    logger.info(f"  Saved {fname}")
    return bk


def cluster_all(tr_norm, tr_pca, all_norm, all_pca, k):
    """
    Fit all clustering models on the training PCA embeddings.
    Predict on the full dataset (all_pca / all_norm).
    Returns tr_pred (dict), all_pred (dict), fitted km model.
    """
    km      = KMeans(n_clusters=k, random_state=42, n_init=20).fit(tr_pca)
    agg     = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(tr_norm)
    gmm     = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(tr_pca)
    sbscan  = SBScanClustering(n_clusters_hint=k, min_samples=5).fit(tr_pca)
    pso     = PSOClustering(n_clusters=k, seed=42).fit(tr_pca)
    gsa     = GravitationalSearchClustering(n_clusters=k, seed=42).fit(tr_pca)

    agg_centroids = np.vstack([tr_norm[agg.labels_ == c].mean(axis=0)
                               for c in range(k)])
    agg_centroids = normalize(agg_centroids, norm="l2")

    def _agg_predict(X_norm):
        return np.linalg.norm(X_norm[:, None, :] - agg_centroids[None, :, :], axis=-1).argmin(axis=1)

    tr_pred = {
        "kmeans" : km.labels_,
        "agg"    : agg.labels_,
        "gmm"    : gmm.predict(tr_pca),
        "sbscan" : sbscan.labels_,
        "pso"    : pso.labels_,
        "gsa"    : gsa.labels_,
    }
    all_pred = {
        "kmeans" : km.predict(all_pca),
        "agg"    : _agg_predict(all_norm),
        "gmm"    : gmm.predict(all_pca),
        "sbscan" : sbscan.predict(all_pca),
        "pso"    : pso.predict(all_pca),
        "gsa"    : gsa.predict(all_pca),
    }
    return tr_pred, all_pred, km


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


def evaluate_unsupervised(emb, pred, pseudo_gt=None, metric="euclidean"):
    """
    Pure unsupervised metrics + optional ARI/NMI vs KMeans pseudo-GT
    (for comparing how consistently different methods agree with KMeans,
    not against any ground truth).
    """
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
    # display-name : (dict-key, sklearn-metric, use-pca)
    "KMeans" : ("kmeans", "euclidean", True),
    "Agg"    : ("agg",    "cosine",    False),
    "GMM"    : ("gmm",    "euclidean", True),
    "SBScan" : ("sbscan", "euclidean", True),
    "PSO"    : ("pso",    "euclidean", True),
    "GSA"    : ("gsa",    "euclidean", True),
}


def print_metrics(all_norm, all_pca, all_pred):
    km_ref = all_pred["kmeans"]
    rows   = {}
    for name, (key, metric, use_pca) in _METHOD_CFG.items():
        emb = all_pca if use_pca else all_norm
        rows[name] = evaluate_unsupervised(emb, all_pred[key],
                                           pseudo_gt=km_ref, metric=metric)
    logger.info("\n── ALL DATA — Unsupervised metrics (ARI/NMI vs KMeans pseudo-GT) ──")
    logger.info("\n" + pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))


def save_predictions(all_pred, cfg):
    n  = len(all_pred["kmeans"])
    df = pd.DataFrame({"window_idx": np.arange(n)})
    for key in all_pred:
        df[f"cluster_{key}"] = all_pred[key]
    out = cfg["output_csv"]
    df.to_csv(out, index=False)
    logger.info(f"  Predictions saved: {Path(out).resolve()}  ({n} windows)")


# ── VibClustNet diagnostic plots ──────────────────────────────────────────────
def plot_vibclustnet_diagnostics(model, sample_X, pred_kmeans, device, figures_dir: Path):
    """
    Three diagnostic plots:
      1. Average temporal attention profile per cluster
      2. CAIM cross-axis attention heatmap per cluster
      3. Original vs reconstructed signal for 3 random samples
    """
    model.eval()
    all_t_attn, all_caim_w = [], []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(sample_X), bs):
            xb = torch.tensor(sample_X[i:i+bs]).to(device)
            if xb.shape[1] != 3:
                xb = xb.permute(0, 2, 1)
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
    axis_labels = ["X", "Y", "Z"]
    fig, axes = plt.subplots(1, n_cls, figsize=(3.5 * n_cls, 3.5), squeeze=False)
    fig.suptitle("VibClustNet — CAIM Cross-Axis Attention per Cluster", fontweight="bold")
    for ax, cid in zip(axes[0], clusters):
        mask  = pred_kmeans == cid
        avg_w = all_caim_w[mask].mean(axis=0)
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

    # 3. Reconstruction for 3 random samples
    rng  = np.random.default_rng(42)
    idxs = rng.choice(len(sample_X), size=min(3, len(sample_X)), replace=False)
    fig, axes = plt.subplots(len(idxs), 3, figsize=(15, 4 * len(idxs)), squeeze=False)
    fig.suptitle("VibClustNet — Reconstruction", fontweight="bold")
    axis_names = ["X", "Y", "Z"]
    with torch.no_grad():
        for row, idx in enumerate(idxs):
            xb = torch.tensor(sample_X[idx:idx+1]).to(device)
            if xb.shape[1] != 3:
                xb = xb.permute(0, 2, 1)
            recon, _, _, _ = model(xb)
            orig = xb[0].cpu().numpy()
            rec  = recon[0].cpu().numpy()
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

_ALL_METHODS = [
    ("kmeans", "K-Means"),
    ("agg",    "Agglomerative"),
    ("gmm",    "GMM"),
    ("sbscan", "SBScan"),
    ("pso",    "PSO"),
    ("gsa",    "GravitationalSearch"),
]


def _scatter_clusters(ax, xy, labels, title, proj_name):
    for i, cid in enumerate(sorted(set(labels))):
        m = np.asarray(labels) == cid
        col = "#bbb" if cid == -1 else COLORS[i % len(COLORS)]
        ax.scatter(xy[m, 0], xy[m, 1],
                   c=col, alpha=0.6, s=8, label=f"Cluster {cid}")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6)
    ax.grid(alpha=0.2)


def project_tsne(pca_emb, tag):
    perp = min(30, len(pca_emb) - 1)
    logger.info(f"  t-SNE [{tag}]...")
    return TSNE(n_components=2, random_state=42,
                perplexity=perp, max_iter=1000).fit_transform(pca_emb)


def plot_all_methods_grid(ts, pred, figures_dir: Path):
    """3×2 grid of all clustering methods coloured by cluster."""
    ncols = 3
    nrows = (len(_ALL_METHODS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 6))
    fig.suptitle("t-SNE — All Clustering Methods (fully unsupervised)",
                 fontsize=13, fontweight="bold")
    axes_flat = iter(axes.flat)
    for key, title in _ALL_METHODS:
        ax = next(axes_flat)
        _scatter_clusters(ax, ts, pred[key], f"t-SNE — {title}", "t-SNE")
    for ax in axes_flat:
        ax.set_visible(False)
    plt.tight_layout()
    fname = figures_dir / f"26_tsne_all_methods_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")


def plot_individual_tsne(ts, pred, figures_dir: Path):
    """One figure per clustering method."""
    for key, title in _ALL_METHODS:
        fig, ax = plt.subplots(figsize=(8, 6))
        _scatter_clusters(ax, ts, pred[key], f"t-SNE — {title}", "t-SNE")
        plt.tight_layout()
        fname = figures_dir / f"tsne_{title}_{RUN_TS}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved {fname}")


def plot_fixed_k_grid(pca_emb, ts, figures_dir: Path, ks=(3, 5, 7, 11)):
    """2×2 grid of KMeans at fixed K values."""
    assert len(ks) == 4
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Fixed-K experiment (unsupervised)", fontsize=13, fontweight="bold")
    rows = {}
    for ax, k in zip(axes.flat, ks):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(pca_emb)
        sil = silhouette_score(pca_emb, lbl)
        db  = davies_bouldin_score(pca_emb, lbl)
        rows[f"K={k}"] = dict(Silhouette=sil, DB=db)
        _scatter_clusters(ax, ts, lbl, f"K={k}  (sil={sil:.3f})", "t-SNE")
    plt.tight_layout()
    fname = figures_dir / f"26_fixed_k_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved {fname}")
    logger.info("\n  Fixed-K quality:")
    logger.info(f"  {'K':<8} {'Silhouette':>12} {'DB':>10}")
    for k, r in rows.items():
        logger.info(f"  {k:<8} {r['Silhouette']:>12.4f} {r['DB']:>10.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    figures_dir = CONFIG["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=Path("logs"))

    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Run timestamp: {RUN_TS}")

    # ── [1] Load all data ────────────────────────────────────────────────────
    logger.info("\n[1] Load all windowed data (no labels)")
    tr_X, te_X, all_X = load_all_data(CONFIG)
    T = all_X.shape[2]
    logger.info(f"  Window length T={T}")

    # ── [2] Train VibClustNet (reconstruction only) ──────────────────────────
    logger.info("\n[2] Train VibClustNet autoencoder (reconstruction only)")
    tr_ds = WindowedDataset(tr_X)
    va_ds = WindowedDataset(te_X)
    tr_loader = DataLoader(tr_ds, batch_size=CONFIG["vcn_batch_size"],
                           shuffle=True,  drop_last=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=CONFIG["vcn_batch_size"],
                           shuffle=False, num_workers=0)
    model = VibClustNet(T=T, emb_dim=CONFIG["vcn_embedding_dim"]).to(device)
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = train_vibclustnet(model, tr_loader, va_loader, CONFIG, device)

    logger.info("\n[2b] Save model")
    save_model(model, CONFIG)

    # ── [3] Embed ALL windows ────────────────────────────────────────────────
    logger.info("\n[3] Embed all windows + PCA")
    tr_emb  = embed(model, tr_X, device)
    all_emb = embed(model, all_X, device)
    tr_norm, tr_pca, all_norm, all_pca, _ = pca_reduce(
        tr_emb, all_emb, CONFIG["pca_variance"])
    logger.info(f"  Train PCA: {tr_pca.shape}  All PCA: {all_pca.shape}")

    # ── [4] Select K ─────────────────────────────────────────────────────────
    logger.info("\n[4] Select K (silhouette sweep on train embeddings)")
    k = best_k(tr_pca, k_min=3, k_max=12, figures_dir=figures_dir)

    # ── [5] Cluster ──────────────────────────────────────────────────────────
    logger.info(f"\n[5] Clustering  K={k}")
    tr_pred, all_pred, _ = cluster_all(tr_norm, tr_pca, all_norm, all_pca, k)

    # ── [6] Metrics ──────────────────────────────────────────────────────────
    logger.info("\n[6] Unsupervised metrics (all data)")
    print_metrics(all_norm, all_pca, all_pred)

    # ── [7] Save predictions ─────────────────────────────────────────────────
    logger.info("\n[7] Save predictions")
    save_predictions(all_pred, CONFIG)

    # ── [8] Visualise ────────────────────────────────────────────────────────
    logger.info("\n[8] Visualise (t-SNE on all data)")
    # Sub-sample if large to keep t-SNE tractable
    max_tsne = 5000
    if len(all_pca) > max_tsne:
        rng  = np.random.default_rng(CONFIG["seed"])
        sidx = rng.choice(len(all_pca), size=max_tsne, replace=False)
        ts_pca   = all_pca[sidx]
        ts_pred  = {k: v[sidx] for k, v in all_pred.items()}
        logger.info(f"  Sub-sampled {max_tsne}/{len(all_pca)} windows for t-SNE")
    else:
        ts_pca, ts_pred = all_pca, all_pred

    ts = project_tsne(ts_pca, "All")
    plot_all_methods_grid(ts, ts_pred, figures_dir)
    plot_individual_tsne(ts, ts_pred, figures_dir)
    plot_fixed_k_grid(ts_pca, ts, figures_dir, ks=(3, 5, 7, 11))

    # ── [9] Diagnostic plots ─────────────────────────────────────────────────
    logger.info("\n[9] VibClustNet diagnostic plots")
    # Use sub-sampled set so attention arrays fit in memory
    diag_X = all_X[sidx] if len(all_X) > max_tsne else all_X
    plot_vibclustnet_diagnostics(
        model, diag_X, ts_pred["kmeans"], device, figures_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

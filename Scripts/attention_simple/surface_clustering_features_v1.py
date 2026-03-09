"""
Road Surface Classification — Feature-based Clustering Pipeline

Labeled rows   : surface_id is a known positive integer
Unlabeled rows : surface_id is 0, NaN, or negative

Steps
-----
1. Load pre-extracted feature CSVs  →  Z-normalise  →  stratified 80/20 split
2. Train MLP autoencoder (features → compact embeddings)
3. Extract embeddings  →  PCA
4. Cluster (8 methods) on PCA embeddings
5. Evaluate on test split (Silhouette, DB, CH, ARI, NMI, Dunn)
6. Save all metrics to CSV in versioned run folder
7. Visualise: t-SNE  (grid: all methods + ground truth)
8. Fixed-K experiment
9. Autoencoder diagnostics (reconstruction error)
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
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

# ── Timestamp ─────────────────────────────────────────────────────────────────
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Version / run identity (edit these to tag your run) ───────────────────
    "version_name"          : "v1.0",          # <- change to tag this run
    "run_base_dir"          : Path("runs"),     # all run folders live here

    # ── Data sources ──────────────────────────────────────────────────────────
    "labeled_features_csv"  : Path("../../Datasets/ExtractedFeatures/labeled_accelerometer_features.csv"),
    "unlabeled_features_csv": Path("../../Datasets/ExtractedFeatures/unlabeled_accelerometer_features.csv"),
    "surface_types_csv"     : Path("../../Datasets/surface_types.csv"),
    "output_csv"            : "unlabeled_predictions.csv",   # relative to run_dir
    "unlabeled_id"          : 0,

    # ── Surface class merging ─────────────────────────────────────────────────
    # "A" (5 classes), "B" (7 classes), "C" (9 classes)
    "merge_map_type"        : "B",

    # ── Train / test split ────────────────────────────────────────────────────
    "test_size"             : 0.2,
    "seed"                  : 42,

    # ── PCA ───────────────────────────────────────────────────────────────────
    "pca_variance"          : 0.95,

    # ── MLP Autoencoder hyper-parameters ──────────────────────────────────────
    "ae_epochs"             : 150,
    "ae_batch_size"         : 64,
    "ae_lr"                 : 1e-3,
    "ae_patience"           : 30,
    "ae_embedding_dim"      : 64,
    "ae_cls_weight"         : 1.0,   # weight for auxiliary classification loss
    "ae_checkpoint"         : "ae_best.pth",   # relative to run_dir
}

# ── Derived run directory: runs/{timestamp}_{version_name}/ ───────────────────
RUN_DIR      = CONFIG["run_base_dir"] / f"{RUN_TS}_{CONFIG['version_name']}"
FIGURES_DIR  = RUN_DIR / "figures"
MODELS_DIR   = RUN_DIR / "models"
LOGS_DIR     = RUN_DIR / "logs"


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"run_{RUN_TS}.log"
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("feat_clust")
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


logger = logging.getLogger("feat_clust")

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
MERGE_MAP_TYPE_A = {
    1: 0, 2: 1, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 3, 10: 3, 11: 4, 12: 4,
}
MERGE_MAP_TYPE_B = {
    1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 0, 9: 6, 10: 6, 11: 7, 12: 7,
}
MERGE_MAP_TYPE_C = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5, 8: 6, 9: 7, 10: 7, 11: 8, 12: 8,
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

_MERGE_MAPS  = {"A": MERGE_MAP_TYPE_A, "B": MERGE_MAP_TYPE_B, "C": MERGE_MAP_TYPE_C}
_SUPER_NAMES = {"A": SUPER_NAMES_TYPE_A, "B": SUPER_NAMES_TYPE_B, "C": SUPER_NAMES_TYPE_C}

MERGE_MAP       = _MERGE_MAPS[CONFIG["merge_map_type"]]
SUPER_NAMES     = _SUPER_NAMES[CONFIG["merge_map_type"]]
N_SUPER_CLASSES = len(SUPER_NAMES)


def remap_labels(y):
    return np.array([MERGE_MAP.get(int(sid), int(sid)) for sid in y])


def super_name(sid):
    return SUPER_NAMES.get(int(sid), sname(sid))


# ── Stratified split ──────────────────────────────────────────────────────────
def stratified_split(X, y, test_size, seed):
    rng = np.random.default_rng(seed)
    tr_i, te_i = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        n   = max(1, int(len(idx) * test_size))
        te_i.extend(idx[:n]); tr_i.extend(idx[n:])
        logger.info(f"    {super_name(cls):40s}: {len(idx)-n:5d} train  {n:4d} test")
    return (X[tr_i], y[tr_i], X[te_i], y[te_i])


# ── Data loading from feature CSVs ────────────────────────────────────────────
def load_feature_csv(csv_path: Path, has_labels: bool):
    """
    Load a pre-extracted feature CSV.
    Returns (X, y) where X is (N, F) float32 and y is (N,) int.
    If has_labels=False, y is all -1.
    """
    logger.info(f"  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Rows: {len(df)}  Columns: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c != "surface_id"]
    X = df[feature_cols].to_numpy(dtype=np.float32)

    if has_labels and "surface_id" in df.columns:
        y = df["surface_id"].fillna(0).astype(int).to_numpy()
    else:
        y = np.full(len(X), -1, dtype=int)

    return X, y


def load_feature_data(cfg):
    """
    Load labeled and unlabeled feature CSVs.

    Returns:
        tr_X  (N_tr, F)  — labeled-train + unlabeled features
        tr_y  (N_tr,)   — super-class labels; -1 for unlabeled
        te_X  (N_te, F)  — labeled-test features
        te_y  (N_te,)   — super-class labels for test
        X_u   (N_u,  F)  — unlabeled features
        scaler           — fitted StandardScaler
    """
    # ── Labeled ───────────────────────────────────────────────────────────────
    X_l_raw, labels = load_feature_csv(cfg["labeled_features_csv"], has_labels=True)
    valid = (labels != cfg["unlabeled_id"]) & (labels > 0)
    X_l, y_l = X_l_raw[valid], remap_labels(labels[valid])
    logger.info(f"  Labeled samples: {len(X_l)}  Classes: {sorted(np.unique(y_l))}")

    # ── Unlabeled ─────────────────────────────────────────────────────────────
    X_u_raw, _ = load_feature_csv(cfg["unlabeled_features_csv"], has_labels=False)
    logger.info(f"  Unlabeled samples: {len(X_u_raw)}")

    # ── Fit scaler on labeled train (80%) ─────────────────────────────────────
    tr_X_l, tr_y_l, te_X, te_y = stratified_split(X_l, y_l, cfg["test_size"], cfg["seed"])

    scaler = StandardScaler().fit(tr_X_l)
    tr_X_l  = scaler.transform(tr_X_l).astype(np.float32)
    te_X    = scaler.transform(te_X).astype(np.float32)
    X_u     = scaler.transform(X_u_raw).astype(np.float32)

    if len(X_u) > 0:
        tr_X = np.concatenate([tr_X_l, X_u], axis=0)
        tr_y = np.concatenate([tr_y_l, np.full(len(X_u), -1, dtype=int)], axis=0)
        logger.info(f"  Training: {len(tr_X_l)} labeled + {len(X_u)} unlabeled = {len(tr_X)} total")
    else:
        tr_X, tr_y = tr_X_l, tr_y_l

    return tr_X, tr_y, te_X, te_y, X_u, scaler


# ══════════════════════════════════════════════════════════════════════════════
# MLP Autoencoder
# ══════════════════════════════════════════════════════════════════════════════

class FeatureAutoencoder(nn.Module):
    """
    MLP Autoencoder for pre-extracted feature vectors.
    Optionally adds a classification head for auxiliary supervised loss.

    Input / output: (batch, input_dim)
    Embedding      : (batch, emb_dim)
    """
    def __init__(self, input_dim: int, emb_dim: int = 64, n_classes: int = 0):
        super().__init__()
        h1, h2 = max(input_dim * 2, 128), max(input_dim, 64)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),       nn.BatchNorm1d(h2), nn.ReLU(),
            nn.Linear(h2, emb_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, h2), nn.BatchNorm1d(h2), nn.ReLU(),
            nn.Linear(h2, h1),      nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, input_dim),
        )
        self.classifier = (
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(emb_dim // 2, n_classes),
            ) if n_classes > 0 else None
        )

    def forward(self, x):
        emb   = self.encoder(x)
        recon = self.decoder(emb)
        return recon, F.normalize(emb, dim=-1)

    def embed(self, x):
        return F.normalize(self.encoder(x), dim=-1)

    def classify(self, emb_norm):
        if self.classifier is None:
            raise RuntimeError("Built without n_classes > 0")
        return self.classifier(emb_norm)


# ── Training ──────────────────────────────────────────────────────────────────
def train_autoencoder(model, tr_X, tr_y, te_X, te_y, cfg, device, run_dir: Path):
    ckpt = run_dir / cfg["ae_checkpoint"]
    opt  = torch.optim.Adam(model.parameters(), lr=cfg["ae_lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-5)

    def make_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=cfg["ae_batch_size"],
                          shuffle=shuffle, drop_last=(shuffle and len(X) > cfg["ae_batch_size"]))

    tr_loader = make_loader(tr_X, tr_y, shuffle=True)
    va_loader = make_loader(te_X, te_y, shuffle=False)

    cls_w = cfg.get("ae_cls_weight", 0.0)
    best, pat = float("inf"), 0
    labeled_mask_tr = tr_y >= 0

    for ep in range(cfg["ae_epochs"]):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_rec = tr_cls = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            recon, emb = model(xb)
            rec_loss = F.mse_loss(recon, xb)
            cls_loss = torch.tensor(0.0, device=device)
            if model.classifier is not None and cls_w > 0:
                lbl_mask = yb >= 0
                if lbl_mask.any():
                    logits   = model.classify(emb[lbl_mask])
                    cls_loss = F.cross_entropy(logits, yb[lbl_mask])
            loss = rec_loss + cls_w * cls_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_rec += rec_loss.item()
            tr_cls += cls_loss.item()
        tr_rec /= len(tr_loader)
        tr_cls /= len(tr_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        va_rec = va_cls = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, emb = model(xb)
                va_rec += F.mse_loss(recon, xb).item()
                if model.classifier is not None and cls_w > 0:
                    lbl_mask = yb >= 0
                    if lbl_mask.any():
                        logits   = model.classify(emb[lbl_mask])
                        va_cls  += F.cross_entropy(logits, yb[lbl_mask]).item()
        va_rec /= len(va_loader)
        va_cls /= len(va_loader)
        va_loss = va_rec + cls_w * va_cls

        sched.step(va_loss)
        imp = va_loss < best - 1e-6
        best, pat = (va_loss, 0) if imp else (best, pat + 1)
        if imp:
            torch.save(model.state_dict(), ckpt)
        lr_now = opt.param_groups[0]["lr"]

        # ── NMI probe every 10 epochs ──────────────────────────────────────
        nmi_str = ""
        if (ep + 1) % 10 == 0:
            X_lbl = tr_X[labeled_mask_tr]
            y_lbl = tr_y[labeled_mask_tr]
            if len(X_lbl) >= 2:
                with torch.no_grad():
                    embs = model.embed(
                        torch.tensor(X_lbl, dtype=torch.float32).to(device)
                    ).cpu().numpy()
                k_tmp  = len(np.unique(y_lbl))
                km_tmp = KMeans(n_clusters=k_tmp, random_state=42, n_init=5).fit_predict(embs)
                nmi    = normalized_mutual_info_score(y_lbl, km_tmp)
                nmi_str = f"  NMI={nmi:.4f}"

        logger.info(
            f"  [AE] {ep+1:3d}/{cfg['ae_epochs']}  "
            f"rec={tr_rec:.4f}  cls={tr_cls:.4f}  "
            f"val_rec={va_rec:.4f}  val_cls={va_cls:.4f}  "
            f"lr={lr_now:.2e}{nmi_str}" + (" *" if imp else "")
        )

        if pat >= cfg["ae_patience"]:
            logger.info(f"  Early stop at epoch {ep+1}")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def save_model(model, run_dir: Path):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sd_path   = MODELS_DIR / f"ae_{RUN_TS}_state_dict.pth"
    full_path = MODELS_DIR / f"ae_{RUN_TS}_full.pth"
    torch.save(model.state_dict(), sd_path)
    torch.save(model, full_path)
    logger.info(f"  Model state dict: {sd_path.resolve()}")
    logger.info(f"  Full model      : {full_path.resolve()}")


@torch.no_grad()
def embed(model, X: np.ndarray, device, bs: int = 512) -> np.ndarray:
    model.eval()
    parts = [model.embed(torch.tensor(X[i:i+bs], dtype=torch.float32).to(device)).cpu()
             for i in range(0, len(X), bs)]
    return torch.cat(parts).numpy()


# ── PCA ───────────────────────────────────────────────────────────────────────
def pca_reduce(emb_tr, emb_te, variance=0.95):
    n_tr = normalize(emb_tr, norm="l2")
    n_te = normalize(emb_te, norm="l2")
    pca  = PCA(n_components=variance, svd_solver="full").fit(n_tr)
    logger.info(f"  PCA: {emb_tr.shape[1]}d → {pca.n_components_}d "
                f"(var={pca.explained_variance_ratio_.sum():.3f})")
    return n_tr, pca.transform(n_tr), n_te, pca.transform(n_te), pca


# ══════════════════════════════════════════════════════════════════════════════
# Clustering algorithms (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

class SBScanClustering:
    """DBSCAN with automatic epsilon via k-NN distance elbow."""
    def __init__(self, n_clusters_hint=5, min_samples=5):
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
        xs       = np.linspace(0.0, 1.0, n)
        rng_d    = k_dists[-1] - k_dists[0]
        ys       = (k_dists - k_dists[0]) / (rng_d + 1e-10)
        knee_idx = int(np.argmax(np.abs(ys - xs)))
        self.eps_ = float(k_dists[knee_idx])
        if self.eps_ < 1e-6:
            self.eps_ = float(np.median(k_dists))
        logger.info(f"    SBScan auto-eps={self.eps_:.4f}")
        db           = DBSCAN(eps=self.eps_, min_samples=self.min_samples).fit(X)
        self.labels_ = db.labels_
        mask    = self.labels_ != -1
        n_valid = int(mask.sum())
        n_cls   = len(np.unique(self.labels_[mask])) if n_valid > 0 else 0
        logger.info(f"    SBScan: {n_cls} clusters, {n_valid}/{n} non-noise")
        if n_valid > 1 and n_cls > 1:
            self._knn_clf = KNeighborsClassifier(
                n_neighbors=min(5, n_valid)).fit(X[mask], self.labels_[mask])
        return self

    def predict(self, X):
        if self._knn_clf is None:
            return np.zeros(len(X), dtype=int)
        return self._knn_clf.predict(X)


class PSOClustering:
    def __init__(self, n_clusters=5, n_particles=15, max_iter=50, seed=42):
        self.n_clusters = n_clusters; self.n_particles = n_particles
        self.max_iter = max_iter; self.seed = seed
        self.centroids_ = None; self.labels_ = None

    def _assign(self, X, c):
        return np.linalg.norm(X[:, None] - c[None], axis=-1).argmin(axis=1)

    def _wcss(self, X, c):
        a = self._assign(X, c)
        return sum(((X[a == k] - c[k]) ** 2).sum()
                   for k in range(len(c)) if (a == k).any())

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        D, K, N = X.shape[1], self.n_clusters, self.n_particles
        particles  = X[rng.choice(len(X), N * K, replace=True)].reshape(N, K, D)
        velocities = rng.uniform(-0.1, 0.1, (N, K, D))
        pbest      = particles.copy()
        pbest_fit  = np.array([self._wcss(X, p) for p in particles])
        gbest      = pbest[pbest_fit.argmin()].copy()
        gbest_fit  = pbest_fit.min()
        w, c1, c2  = 0.5, 1.5, 1.5
        for _ in range(self.max_iter):
            r1 = rng.uniform(0, 1, (N, K, D)); r2 = rng.uniform(0, 1, (N, K, D))
            velocities = (w * velocities + c1 * r1 * (pbest - particles)
                          + c2 * r2 * (gbest[None] - particles))
            particles += velocities
            fits = np.array([self._wcss(X, p) for p in particles])
            imp = fits < pbest_fit
            pbest[imp] = particles[imp]; pbest_fit[imp] = fits[imp]
            if fits.min() < gbest_fit:
                gbest = particles[fits.argmin()].copy(); gbest_fit = fits.min()
        self.centroids_ = gbest
        self.labels_    = self._assign(X, gbest)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


class RandomAssignClustering:
    def __init__(self, n_clusters=5, seed=42):
        self.n_clusters = n_clusters; self.seed = seed; self.labels_ = None

    def fit(self, X):
        self.labels_ = np.random.default_rng(self.seed).integers(0, self.n_clusters, len(X))
        return self

    def predict(self, X):
        return np.random.default_rng(self.seed + 1).integers(0, self.n_clusters, len(X))


class GravitationalSearchClustering:
    def __init__(self, n_clusters=5, n_agents=15, max_iter=50,
                 G0=100.0, alpha=20.0, seed=42):
        self.n_clusters = n_clusters; self.n_agents = n_agents
        self.max_iter = max_iter; self.G0 = G0; self.alpha = alpha
        self.seed = seed; self.centroids_ = None; self.labels_ = None

    def _assign(self, X, c):
        return np.linalg.norm(X[:, None] - c[None], axis=-1).argmin(axis=1)

    def _fitness(self, X, c):
        a = self._assign(X, c)
        return sum(((X[a == k] - c[k]) ** 2).sum()
                   for k in range(len(c)) if (a == k).any())

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        D, K, N = X.shape[1], self.n_clusters, self.n_agents
        agents     = X[rng.choice(len(X), N * K, replace=True)].reshape(N, K, D)
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
                best_fit = fits.min(); best_agent = agents[fits.argmin()].copy()
            k_best  = max(2, int(N * (1 - t / self.max_iter)))
            top_idx = np.argsort(fits)[:k_best]
            acc = np.zeros((N, K, D))
            for i in range(N):
                for j in top_idx:
                    if i == j: continue
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
    def __init__(self, n_clusters=5, seed=42):
        self.n_clusters = n_clusters; self.seed = seed
        self.centroids_ = None; self.labels_ = None

    def _assign(self, X, c):
        return np.linalg.norm(X[:, None] - c[None], axis=-1).argmin(axis=1)

    def fit(self, X):
        idxs = np.random.default_rng(self.seed).choice(len(X), self.n_clusters, replace=False)
        self.centroids_ = X[idxs]
        self.labels_    = self._assign(X, self.centroids_)
        return self

    def predict(self, X):
        return self._assign(X, self.centroids_)


# ── K selection ───────────────────────────────────────────────────────────────
def best_k(pca_emb, k_min, k_max, figures_dir: Path):
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
    ax.set_title("K selection"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = figures_dir / f"k_selection_{RUN_TS}.png"
    plt.savefig(fname, dpi=150); plt.close()
    logger.info(f"  Saved {fname}")
    return bk


# ── Cluster all methods ───────────────────────────────────────────────────────
def cluster_all(tr_norm, tr_pca, te_norm, te_pca, k, xu_norm=None, xu_pca=None):
    km     = KMeans(n_clusters=k, random_state=42, n_init=20).fit(tr_pca)
    agg    = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(tr_norm)
    gmm    = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(tr_pca)
    sbscan = SBScanClustering(n_clusters_hint=k, min_samples=5).fit(tr_pca)
    pos    = PSOClustering(n_clusters=k, seed=42).fit(tr_pca)
    rand_a = RandomAssignClustering(n_clusters=k, seed=42).fit(tr_pca)
    gsa    = GravitationalSearchClustering(n_clusters=k, seed=42).fit(tr_pca)
    rand_c = RandomClustering(n_clusters=k, seed=42).fit(tr_pca)

    agg_centroids = np.vstack([tr_norm[agg.labels_ == c].mean(axis=0) for c in range(k)])
    agg_centroids = normalize(agg_centroids, norm="l2")

    def _agg_pred(X_norm):
        return np.linalg.norm(X_norm[:, None] - agg_centroids[None], axis=-1).argmin(axis=1)

    tr_pred = {
        "kmeans": km.labels_, "agg": agg.labels_,
        "gmm": gmm.predict(tr_pca), "sbscan": sbscan.labels_,
        "pos": pos.labels_, "rand_assign": rand_a.labels_,
        "gsa": gsa.labels_, "rand_clust": rand_c.labels_,
    }
    te_pred = {
        "kmeans": km.predict(te_pca), "agg": _agg_pred(te_norm),
        "gmm": gmm.predict(te_pca), "sbscan": sbscan.predict(te_pca),
        "pos": pos.predict(te_pca),
        "rand_assign": RandomAssignClustering(n_clusters=k, seed=99).fit(te_pca).labels_,
        "gsa": gsa.predict(te_pca),
        "rand_clust": RandomClustering(n_clusters=k, seed=99).fit(te_pca).labels_,
    }
    xu_pred = None
    if xu_norm is not None and xu_pca is not None and len(xu_pca) > 0:
        xu_pred = {
            "kmeans": km.predict(xu_pca), "agg": _agg_pred(xu_norm),
            "gmm": gmm.predict(xu_pca), "sbscan": sbscan.predict(xu_pca),
            "pos": pos.predict(xu_pca),
            "rand_assign": RandomAssignClustering(n_clusters=k, seed=77).fit(xu_pca).labels_,
            "gsa": gsa.predict(xu_pca),
            "rand_clust": RandomClustering(n_clusters=k, seed=77).fit(xu_pca).labels_,
        }
    return tr_pred, te_pred, xu_pred, km


# ── Metrics ───────────────────────────────────────────────────────────────────
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


def c2s_map(pred, gt):
    return {c: (-1 if c == -1 else int(pd.Series(gt[np.asarray(pred) == c]).mode()[0]))
            for c in set(pred)}


def evaluate(emb, pred, gt, metric="euclidean"):
    pred = np.asarray(pred); ok = pred != -1
    ev, lv, gv = emb[ok], pred[ok], np.asarray(gt)[ok]
    if len(np.unique(lv)) < 2:
        return dict(Silhouette=np.nan, DB=np.nan, CH=np.nan,
                    ARI=np.nan, NMI=np.nan, Dunn=np.nan)
    return dict(
        Silhouette = silhouette_score(ev, lv, metric=metric),
        DB         = davies_bouldin_score(ev, lv),
        CH         = calinski_harabasz_score(ev, lv),
        ARI        = adjusted_rand_score(gv, lv),
        NMI        = normalized_mutual_info_score(gv, lv),
        Dunn       = dunn_index(ev, lv),
    )


def evaluate_unsupervised(emb, pred, pseudo_gt=None, metric="euclidean"):
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
    "KMeans"    : ("kmeans",      "euclidean", True),
    "Agg"       : ("agg",         "cosine",    False),
    "GMM"       : ("gmm",         "euclidean", True),
    "SBScan"    : ("sbscan",      "euclidean", True),
    "POS"       : ("pos",         "euclidean", True),
    "RandAssign": ("rand_assign", "euclidean", True),
    "GSA"       : ("gsa",         "euclidean", True),
    "RandClust" : ("rand_clust",  "euclidean", True),
}


def compute_metrics(tr_norm, tr_pca, tr_pred, tr_lbl,
                    te_norm, te_pca, te_pred, te_lbl,
                    xu_norm=None, xu_pca=None, xu_pred=None):
    """
    Compute all metrics and return as a dict of DataFrames.
    Keys: 'train', 'test', 'unlabeled'.
    """
    all_dfs = {}

    for split, pred, gt in [("train", tr_pred, tr_lbl), ("test", te_pred, te_lbl)]:
        rows = {}
        for name, (key, metric, use_pca) in _METHOD_CFG.items():
            emb_tr = tr_pca if use_pca else tr_norm
            emb_te = te_pca if use_pca else te_norm
            emb    = emb_tr if split == "train" else emb_te
            rows[name] = evaluate(emb, pred[key], gt, metric)
        df = pd.DataFrame(rows).T
        all_dfs[split] = df
        logger.info(f"\n── {split.upper()} ─────────────────────────────────────")
        logger.info("\n" + df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Generalisation gap
    logger.info("\nGeneralisation gap  (train ARI − test ARI):")
    for name, (key, metric, use_pca) in _METHOD_CFG.items():
        emb_tr = tr_pca if use_pca else tr_norm
        emb_te = te_pca if use_pca else te_norm
        tr_v   = evaluate(emb_tr, tr_pred[key], tr_lbl, metric).get("ARI") or 0
        te_v   = evaluate(emb_te, te_pred[key], te_lbl, metric).get("ARI") or 0
        gap    = (tr_v or 0) - (te_v or 0)
        flag   = "good" if gap < 0.10 else "overfit" if gap > 0.20 else "acceptable"
        logger.info(f"  {name:12s}  train={tr_v:.4f}  test={te_v:.4f}  gap={gap:+.4f}  [{flag}]")

    if xu_pred is not None and xu_norm is not None and xu_pca is not None:
        km_ref = xu_pred["kmeans"]
        rows = {}
        for name, (key, metric, use_pca) in _METHOD_CFG.items():
            emb = xu_pca if use_pca else xu_norm
            rows[name] = evaluate_unsupervised(emb, xu_pred[key], pseudo_gt=km_ref, metric=metric)
        df = pd.DataFrame(rows).T
        all_dfs["unlabeled"] = df
        logger.info("\n── UNLABELED (unsupervised + ARI/NMI vs KMeans pseudo-GT) ──────────")
        logger.info("\n" + df.to_string(float_format=lambda x: f"{x:.4f}"))

    return all_dfs


def save_metrics(all_dfs: dict, run_dir: Path):
    """Save each split's metric DataFrame to a CSV in the run folder."""
    for split, df in all_dfs.items():
        df.index.name = "Method"
        path = run_dir / f"metrics_{split}.csv"
        df.round(6).to_csv(path)
        logger.info(f"  Metrics saved: {path.resolve()}")


def save_unlabeled_predictions(xu_pred, run_dir: Path, filename: str):
    if xu_pred is None:
        return
    n  = len(xu_pred["kmeans"])
    df = pd.DataFrame({"window_idx": np.arange(n)})
    for key in xu_pred:
        df[f"cluster_{key}"] = xu_pred[key]
    out = run_dir / filename
    df.to_csv(out, index=False)
    logger.info(f"  Unlabeled predictions: {out.resolve()}  ({n} windows)")


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


def _scatter(ax, xy, lbls, legend_fn, title, xlabel):
    for i, lbl in enumerate(sorted(set(lbls))):
        m = np.asarray(lbls) == lbl
        ax.scatter(xy[m, 0], xy[m, 1],
                   c="#bbb" if lbl == -1 else COLORS[i % len(COLORS)],
                   alpha=0.6, s=8, label=legend_fn(lbl))
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{xlabel} 1"); ax.set_ylabel(f"{xlabel} 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6)
    ax.grid(alpha=0.2)


def project(pca_emb, tag):
    perp = min(30, len(pca_emb) - 1)
    logger.info(f"  t-SNE [{tag}]...")
    return TSNE(n_components=2, random_state=42,
                perplexity=perp, max_iter=1000).fit_transform(pca_emb)


def _plot_proj_grid(coords, proj_name, pred, c2s, gt, fname):
    ncols, n_panels = 3, len(_ALL_METHODS) + 1
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
    ax = next(axes_flat)
    for i, sid in enumerate(sorted(np.unique(gt))):
        m = np.asarray(gt) == sid
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=COLORS[i % len(COLORS)], alpha=0.6, s=8,
                   label=super_name(sid))
    ax.set_title(f"{proj_name} — Ground Truth", fontsize=9, fontweight="bold")
    ax.set_xlabel(f"{proj_name} 1"); ax.set_ylabel(f"{proj_name} 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6)
    ax.grid(alpha=0.2)
    for ax in axes_flat:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Saved {fname}")


def plot_tsne_individual(ts, pred, c2s, gt, figures_dir: Path):
    for key, title in _ALL_METHODS:
        fig, ax = plt.subplots(figsize=(8, 6))
        leg_fn  = (lambda lbl, k=key:
                   f"{super_name(c2s[k].get(lbl, '?'))} [c{lbl}]")
        _scatter(ax, ts, pred[key], leg_fn, f"t-SNE — {title}", "t-SNE")
        plt.tight_layout()
        fname = figures_dir / f"tsne_{title}_{RUN_TS}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        logger.info(f"  Saved {fname}")
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, sid in enumerate(sorted(np.unique(gt))):
        m = np.asarray(gt) == sid
        ax.scatter(ts[m, 0], ts[m, 1],
                   c=COLORS[i % len(COLORS)], alpha=0.6, s=8, label=super_name(sid))
    ax.set_title("t-SNE — Ground Truth", fontsize=9, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best", framealpha=0.6)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fname = figures_dir / f"tsne_GroundTruth_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Saved {fname}")


def plot_grid(ts, pred, c2s, gt, figures_dir: Path):
    fname = figures_dir / f"tsne_test_clusters_{RUN_TS}.png"
    _plot_proj_grid(ts, "t-SNE", pred, c2s, gt, fname=fname)
    plot_tsne_individual(ts, pred, c2s, gt, figures_dir)


def _fixed_k_grid(pca_emb, proj_coords, proj_name, fname, ks):
    assert len(ks) == 4
    rows = {}
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Fixed-K experiment — {proj_name} (test split)",
                 fontsize=13, fontweight="bold")
    for ax, k in zip(axes.flat, ks):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(pca_emb)
        rows[f"K={k}"] = dict(
            Silhouette=silhouette_score(pca_emb, lbl),
            DB=davies_bouldin_score(pca_emb, lbl),
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
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Saved {fname}")
    return rows


def experiment_fixed_k(pca_emb, ts, figures_dir: Path, ks=(3, 5, 7, 11)):
    logger.info(f"\n  Fixed-K experiment  K={list(ks)}")
    fname     = figures_dir / f"fixed_k_tsne_{RUN_TS}.png"
    rows_ts   = _fixed_k_grid(pca_emb, ts, "t-SNE", fname, ks)
    logger.info("\n  Cluster quality — t-SNE projection:")
    logger.info(f"  {'K':<8} {'Silhouette':>12} {'Davies-Bouldin':>16}  Quality")
    logger.info("  " + "-" * 46)
    for k, r in rows_ts.items():
        sil, db = r["Silhouette"], r["DB"]
        q = "Excellent" if sil > 0.6 else "Good" if sil > 0.4 else "Fair" if sil > 0.2 else "Poor"
        logger.info(f"  {k:<8} {sil:>12.4f} {db:>16.4f}  {q}")
    return pd.DataFrame(rows_ts).T


# ── Autoencoder diagnostics ───────────────────────────────────────────────────
def plot_ae_diagnostics(model, te_X: np.ndarray, te_pred_kmeans: np.ndarray,
                        feature_names: list, device, figures_dir: Path):
    """
    Three diagnostic plots:
    1. Reconstruction error per cluster (box plot)
    2. Per-feature mean reconstruction error
    3. Embedding PCA scatter coloured by cluster
    """
    model.eval()
    with torch.no_grad():
        X_t    = torch.tensor(te_X, dtype=torch.float32).to(device)
        recon, emb = model(X_t)
        errors = (recon - X_t).pow(2).mean(dim=1).cpu().numpy()   # (N,) per-sample MSE
        feat_err = (recon - X_t).pow(2).mean(dim=0).cpu().numpy() # (F,) per-feature MSE
        emb_np = emb.cpu().numpy()

    clusters = sorted(set(te_pred_kmeans))

    # ── 1. Per-cluster reconstruction error box plot ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    data = [errors[te_pred_kmeans == cid] for cid in clusters]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, cid in zip(bp["boxes"], clusters):
        patch.set_facecolor(COLORS[cid % len(COLORS)])
    ax.set_xticks(range(1, len(clusters) + 1))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters], fontsize=8)
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("AE — Reconstruction Error per Cluster (test split)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = figures_dir / f"ae_recon_error_per_cluster_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Saved {fname}")

    # ── 2. Per-feature reconstruction error ───────────────────────────────────
    top_n = min(30, len(feat_err))
    top_idx = np.argsort(feat_err)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(top_n), feat_err[top_idx], color="#457b9d")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [feature_names[i] if i < len(feature_names) else str(i) for i in top_idx],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_ylabel("Mean MSE"); ax.set_title(f"AE — Top {top_n} Features by Reconstruction Error",
                                             fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = figures_dir / f"ae_feature_recon_error_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Saved {fname}")

    # ── 3. Embedding PCA scatter coloured by cluster ──────────────────────────
    if emb_np.shape[1] > 2:
        pca2 = PCA(n_components=2).fit_transform(emb_np)
    else:
        pca2 = emb_np
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cid in enumerate(clusters):
        m = te_pred_kmeans == cid
        ax.scatter(pca2[m, 0], pca2[m, 1],
                   c=COLORS[i % len(COLORS)], alpha=0.6, s=8, label=f"Cluster {cid}")
    ax.set_title("AE Embedding PCA (test split, K-Means colours)", fontweight="bold")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(fontsize=7, markerscale=1.5, loc="best", framealpha=0.6)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fname = figures_dir / f"ae_embedding_pca_{RUN_TS}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Create run directory ──────────────────────────────────────────────────
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging()

    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Run timestamp : {RUN_TS}")
    logger.info(f"Version       : {CONFIG['version_name']}")
    logger.info(f"Run directory : {RUN_DIR.resolve()}")
    logger.info(f"Device        : {device}")

    logger.info("\n[0] Surface names")
    load_surface_names(CONFIG["surface_types_csv"])

    logger.info("\n[1] Load feature data (labeled + unlabeled)")
    tr_X, tr_y, te_X, te_y, X_u, scaler = load_feature_data(CONFIG)
    F_dim = tr_X.shape[1]
    logger.info(f"  Feature dim: {F_dim}")

    # Labeled portion of training set
    labeled_mask = tr_y >= 0
    tr_X_labeled = tr_X[labeled_mask]
    tr_y_labeled = tr_y[labeled_mask]

    logger.info("\n[2] Super-class distribution (labeled train):")
    for sc, n in sorted(zip(*np.unique(tr_y_labeled, return_counts=True))):
        logger.info(f"    {super_name(sc):40s}: {n} samples")

    logger.info("\n[3] Train MLP Autoencoder")
    n_cls_for_clf = max(SUPER_NAMES.keys()) + 1
    model = FeatureAutoencoder(input_dim=F_dim,
                               emb_dim=CONFIG["ae_embedding_dim"],
                               n_classes=n_cls_for_clf).to(device)
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = train_autoencoder(model, tr_X, tr_y, te_X, te_y, CONFIG, device, RUN_DIR)

    logger.info("\n[3b] Save model")
    save_model(model, RUN_DIR)

    logger.info("\n[4] Embeddings + PCA (labeled train/test)")
    tr_emb = embed(model, tr_X_labeled, device)
    te_emb = embed(model, te_X, device)
    tr_norm, tr_pca, te_norm, te_pca, pca_obj = pca_reduce(tr_emb, te_emb, CONFIG["pca_variance"])

    xu_norm = xu_pca = None
    if len(X_u) > 0:
        logger.info(f"\n[4b] Embed unlabeled ({len(X_u)} samples)")
        xu_emb  = embed(model, X_u, device)
        xu_norm = normalize(xu_emb, norm="l2")
        xu_pca  = pca_obj.transform(xu_norm)
        logger.info(f"  Unlabeled embeddings: {xu_norm.shape}  PCA: {xu_pca.shape}")

    logger.info("\n[5] K selection")
    n_surf = N_SUPER_CLASSES
    k = best_k(tr_pca, max(2, n_surf - 2), n_surf + 2, FIGURES_DIR)

    logger.info("\n[6] Clustering (8 methods)")
    tr_pred, te_pred, xu_pred, _ = cluster_all(
        tr_norm, tr_pca, te_norm, te_pca, k,
        xu_norm=xu_norm, xu_pca=xu_pca)

    tr_c2s = {m: c2s_map(tr_pred[m], tr_y_labeled) for m in tr_pred}
    te_c2s = {m: c2s_map(te_pred[m], te_y) for m in te_pred}

    logger.info("\n  K-Means cluster → surface (train):")
    for cid, sid in sorted(tr_c2s["kmeans"].items()):
        logger.info(f"    cluster {cid:2d} → {super_name(sid)}")

    logger.info("\n[7] Metrics (Silhouette, DB, CH, ARI, NMI, Dunn)")
    all_dfs = compute_metrics(tr_norm, tr_pca, tr_pred, tr_y_labeled,
                              te_norm, te_pca, te_pred, te_y,
                              xu_norm=xu_norm, xu_pca=xu_pca, xu_pred=xu_pred)

    logger.info("\n[7b] Save metrics to CSV")
    save_metrics(all_dfs, RUN_DIR)

    logger.info("\n[7c] Save unlabeled predictions")
    save_unlabeled_predictions(xu_pred, RUN_DIR, CONFIG["output_csv"])

    logger.info("\n[8] Visualise (t-SNE on test split)")
    ts = project(te_pca, "Test")
    plot_grid(ts, te_pred, te_c2s, te_y, FIGURES_DIR)

    logger.info("\n[9] Fixed-K experiment (K=3, 5, 7, 11)")
    experiment_fixed_k(te_pca, ts, FIGURES_DIR, ks=(3, 5, 7, 11))

    logger.info("\n[10] Autoencoder diagnostics")
    # Derive feature names from CSV header (excluding surface_id)
    try:
        feat_names = [c for c in pd.read_csv(CONFIG["labeled_features_csv"], nrows=0).columns
                      if c != "surface_id"]
    except Exception:
        feat_names = [f"f{i}" for i in range(F_dim)]
    plot_ae_diagnostics(model, te_X, te_pred["kmeans"], feat_names, device, FIGURES_DIR)

    logger.info(f"\nDone!  All outputs in: {RUN_DIR.resolve()}")
    logger.info(f"  metrics_train.csv      → {(RUN_DIR / 'metrics_train.csv').resolve()}")
    logger.info(f"  metrics_test.csv       → {(RUN_DIR / 'metrics_test.csv').resolve()}")
    if "unlabeled" in all_dfs:
        logger.info(f"  metrics_unlabeled.csv  → {(RUN_DIR / 'metrics_unlabeled.csv').resolve()}")


if __name__ == "__main__":
    main()

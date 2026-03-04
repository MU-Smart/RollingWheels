"""
Road Surface Classification -- Semi-Supervised Vibration Embedding & Clustering
================================================================================
Version 9: Semi-supervised learning with labeled + unlabeled windows.

Unlabeled rows are identified by surface_id == 0 (or NaN / negative).
Set UNLABELED_ID in CONFIG to match whatever value your CSV uses.

Pipeline
--------
Phase 1 — Supervised contrastive (labeled only)
  Labeled windows -> MLPEncoder trained with HardNeg-SupCon loss.
  The encoder learns a geometry where same-surface windows cluster together.

Phase 2 — Pseudo-labeling
  All unlabeled windows are embedded with the Phase-1 encoder.
  A KNN classifier (fit on labeled embeddings) assigns a pseudo-label
  and a confidence score (fraction of neighbours agreeing) to each
  unlabeled window.  Only windows above PSEUDO_CONF_THRESHOLD are kept.

Phase 3 — Semi-supervised fine-tune (labeled + confident pseudo-labeled)
  The encoder is fine-tuned with a mixed batch loss:
    L = L_sup(labeled) + pseudo_weight * L_sup(pseudo-labeled)
  pseudo_weight < 1 down-weights the noisier pseudo-labels.

Clustering & evaluation
  Final embeddings of ALL data (labeled train + test + pseudo-labeled
  unlabeled) are clustered with KMeans / Agglomerative / DBSCAN.
  t-SNE 2-D is shown for the TEST split (labeled) only.
  A separate t-SNE shows the unlabeled windows coloured by their
  pseudo-label assignment.

Output CSV
  unlabeled_predictions.csv — the unlabeled windows with their
  predicted surface_id, confidence score, and cluster assignment.
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    "features_csv"          : Path("../Datasets/ExtractedFeatures/accelerometer_features.csv"),
    "surface_types_csv"     : Path("../Datasets/surface_types.csv"),
    "output_csv"            : Path("unlabeled_predictions.csv"),

    # surface_id value that means "unlabeled" in your CSV (0, -1, or NaN)
    "unlabeled_id"          : 0,

    "pca_variance_threshold": 0.95,

    # Model
    "embedding_dim" : 64,

    # Phase 1 — supervised contrastive
    "epochs_phase1"  : 150,
    "batch_size"     : 256,
    "lr"             : 3e-4,
    "temperature"    : 0.07,
    "hard_neg_ratio" : 0.5,

    # Phase 2 — pseudo-labeling
    "pseudo_k"              : 15,    # KNN neighbours for pseudo-label vote
    "pseudo_conf_threshold" : 0.60,  # min fraction of neighbours agreeing

    # Phase 3 — semi-supervised fine-tune
    "epochs_phase3"  : 75,
    "lr_phase3"      : 1e-4,         # lower LR for fine-tuning
    "pseudo_weight"  : 0.5,          # loss weight for pseudo-labeled batches

    # Train / test split
    "test_size"  : 0.2,
    "split_seed" : 42,
}

# ---------------------------------------------------------------------------
# Surface name helpers
# ---------------------------------------------------------------------------
SURFACE_NAMES: dict = {}

def load_surface_names(csv_path):
    global SURFACE_NAMES
    try:
        df       = pd.read_csv(csv_path)
        id_col   = next(c for c in df.columns if "id"   in c.lower())
        name_col = next(c for c in df.columns if "name" in c.lower())
        SURFACE_NAMES = {int(r[id_col]): str(r[name_col]) for _, r in df.iterrows()}
        print(f"  Loaded {len(SURFACE_NAMES)} surface types:")
        for sid, name in sorted(SURFACE_NAMES.items()):
            print(f"    {sid:3d} -> {name}")
    except Exception as e:
        print(f"  WARNING: Could not load surface names ({e}). Using numeric IDs.")

def surface_name(sid):
    return SURFACE_NAMES.get(int(sid), f"Surface {sid}")

# ---------------------------------------------------------------------------
# Load features CSV  — splits into labeled and unlabeled arrays
# ---------------------------------------------------------------------------
def _find_col(columns, candidates):
    cl = [c.lower() for c in columns]
    for cand in candidates:
        if cand in cl:
            return columns[cl.index(cand)]
    return None


def load_features(config):
    """
    Returns
    -------
    X_lab   : (N_l, F)  labeled feature matrix   (StandardScaler fitted here)
    y_lab   : (N_l,)    surface_id labels
    X_unl   : (N_u, F)  unlabeled feature matrix  (same scaler applied)
    scaler  : fitted StandardScaler
    feature_cols : list[str]
    """
    unlabeled_id = config["unlabeled_id"]
    path = config["features_csv"]
    print(f"  Reading {path} ...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")

    # Drop unnamed index columns
    drop_cols = [c for c in df.columns if re.match(r"^unnamed", c, re.I) or c == "index"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Detect label column
    label_col = _find_col(df.columns,
                          ["surface_id", "surfaceid", "surface_type_id",
                           "label", "surface", "class", "target"])
    if label_col is None:
        raise ValueError(f"Cannot find label column. Columns: {df.columns.tolist()}")
    print(f"  Label column : '{label_col}'")

    # Feature columns = all numeric except label
    feature_cols = [c for c in df.columns
                    if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Feature cols : {len(feature_cols)}  ({feature_cols[:6]} ...)")

    # Split labeled vs unlabeled
    # Treat NaN, 0 (or configured unlabeled_id), and negatives as unlabeled
    raw_labels = pd.to_numeric(df[label_col], errors="coerce")
    is_unl = raw_labels.isna() | (raw_labels == unlabeled_id) | (raw_labels < 0)

    df_lab = df[~is_unl].copy()
    df_unl = df[is_unl].copy()
    print(f"  Labeled rows : {len(df_lab)}  |  Unlabeled rows : {len(df_unl)}")

    def clean(d):
        X = d[feature_cols].values.astype(np.float32)
        bad = ~np.isfinite(X).all(axis=1)
        return X[~bad], d[label_col].values[~bad] if not d.empty else np.array([])

    X_lab_raw, y_raw = clean(df_lab)
    y_lab = y_raw.astype(int)
    X_unl_raw, _    = clean(df_unl)

    print(f"  After NaN drop — labeled: {X_lab_raw.shape}  unlabeled: {X_unl_raw.shape}")
    print(f"  Surface IDs  : {sorted(np.unique(y_lab))}")

    # Fit scaler on labeled data only
    scaler    = StandardScaler()
    X_lab     = scaler.fit_transform(X_lab_raw).astype(np.float32)
    X_unl     = scaler.transform(X_unl_raw).astype(np.float32) if len(X_unl_raw) else np.empty((0, X_lab.shape[1]), np.float32)

    return X_lab, y_lab, X_unl, scaler, feature_cols


# ---------------------------------------------------------------------------
# Stratified split (labeled data only)
# ---------------------------------------------------------------------------
def stratified_split(X, labels, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    print("  Per-surface window split:")
    for surf in sorted(np.unique(labels)):
        idx      = np.where(labels == surf)[0]
        shuffled = rng.permutation(idx)
        n_te     = max(1, int(len(shuffled) * test_size))
        test_idx.extend(shuffled[:n_te].tolist())
        train_idx.extend(shuffled[n_te:].tolist())
        print(f"    {surface_name(surf):30s}: {len(shuffled)-n_te:5d} train  {n_te:4d} test")
    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    print(f"  Total: {len(train_idx)} train | {len(test_idx)} test")
    return (X[train_idx], labels[train_idx], X[test_idx], labels[test_idx])


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
class MLPEncoder(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int = 64, proj_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256),         nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128),         nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, proj_dim),
        )
        print(f"  MLPEncoder: in={feature_dim} -> 512->256->128 -> embed={embedding_dim} -> proj={proj_dim}")

    def embed(self, x):
        return F.normalize(self.encoder(x), p=2, dim=-1)

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
        sim      = z @ z.T
        eye      = torch.eye(B, dtype=torch.bool, device=device)
        lbl2d    = labels.unsqueeze(1)
        pos_mask = (lbl2d == lbl2d.T) & ~eye
        neg_mask = ~pos_mask & ~eye

        k         = max(1, int(neg_mask.sum(1).float().mean().item() * self.hard_neg_ratio))
        neg_sim   = sim.masked_fill(~neg_mask, -1e9)
        topk, _   = neg_sim.topk(k, dim=1)
        hard_neg  = neg_mask & (sim >= topk[:, -1:].detach())

        active    = pos_mask | hard_neg
        ss        = sim / self.temperature
        ss        = ss - ss.masked_fill(~active & ~pos_mask, -1e9).max(1, keepdim=True).values.detach()
        exp_s     = torch.exp(ss).masked_fill(~active, 0.0)
        log_denom = torch.log(exp_s.sum(1, keepdim=True) + 1e-9)
        n_pos     = pos_mask.sum(1).float().clamp(min=1)
        loss      = -((ss - log_denom) * pos_mask.float()).sum(1) / n_pos
        has_pos   = pos_mask.any(1)
        return loss[has_pos].mean() if has_pos.any() else loss.mean()


# ---------------------------------------------------------------------------
# Phase 1 — supervised contrastive training (labeled only)
# ---------------------------------------------------------------------------
def train_supervised(model, X, labels, config, device, tag="Phase1"):
    Xt      = torch.tensor(X, dtype=torch.float32)
    Yt      = torch.tensor(labels, dtype=torch.long)
    sampler = make_balanced_sampler(labels)
    loader  = DataLoader(TensorDataset(Xt, Yt),
                         batch_size=config["batch_size"],
                         sampler=sampler, drop_last=True)
    epochs  = config["epochs_phase1"] if tag == "Phase1" else config["epochs_phase3"]
    lr      = config["lr"]            if tag == "Phase1" else config["lr_phase3"]

    criterion = HardNegSupConLoss(config["temperature"], config["hard_neg_ratio"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, patience, PATIENCE = float("inf"), 0, 30

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            _, proj = model(xb)
            loss    = criterion(proj, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        avg      = total / len(loader)
        improved = avg < best_loss - 1e-4
        best_loss, patience = (avg, 0) if improved else (best_loss, patience + 1)
        print(f"  [{tag}] Epoch {epoch+1:3d}/{epochs} | Loss {avg:.4f}" + (" *" if improved else ""))
        if patience >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    return model


# ---------------------------------------------------------------------------
# Phase 3 — semi-supervised fine-tune (labeled + pseudo-labeled, mixed loss)
# ---------------------------------------------------------------------------
def train_semisupervised(model, X_lab, y_lab, X_psd, y_psd, config, device):
    """
    Each batch is assembled by sampling equally from labeled and pseudo-labeled
    pools, then computing SupCon loss independently on each half and combining:
        L = L_labeled + pseudo_weight * L_pseudo
    """
    criterion   = HardNegSupConLoss(config["temperature"], config["hard_neg_ratio"]).to(device)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=config["lr_phase3"], weight_decay=1e-4)
    epochs      = config["epochs_phase3"]
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pw          = config["pseudo_weight"]
    bs          = config["batch_size"] // 2   # half batch each source

    Xl  = torch.tensor(X_lab, dtype=torch.float32)
    Yl  = torch.tensor(y_lab, dtype=torch.long)
    Xp  = torch.tensor(X_psd, dtype=torch.float32)
    Yp  = torch.tensor(y_psd, dtype=torch.long)

    lab_sampler = make_balanced_sampler(y_lab)
    psd_sampler = make_balanced_sampler(y_psd)
    lab_loader  = DataLoader(TensorDataset(Xl, Yl), batch_size=bs, sampler=lab_sampler, drop_last=True)
    psd_loader  = DataLoader(TensorDataset(Xp, Yp), batch_size=bs, sampler=psd_sampler, drop_last=True)

    best_loss, patience, PATIENCE = float("inf"), 0, 20

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (xl, yl), (xp, yp) in zip(lab_loader, psd_loader):
            xl, yl = xl.to(device), yl.to(device)
            xp, yp = xp.to(device), yp.to(device)
            optimizer.zero_grad()
            _, pl = model(xl)
            _, pp = model(xp)
            loss  = criterion(pl, yl) + pw * criterion(pp, yp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        avg      = total / len(lab_loader)
        improved = avg < best_loss - 1e-4
        best_loss, patience = (avg, 0) if improved else (best_loss, patience + 1)
        print(f"  [Phase3] Epoch {epoch+1:3d}/{epochs} | Loss {avg:.4f}" + (" *" if improved else ""))
        if patience >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    return model


# ---------------------------------------------------------------------------
# Pseudo-labeling via KNN on embeddings
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_embeddings(model, X, device, batch_size=512):
    model.eval()
    Xt, out = torch.tensor(X, dtype=torch.float32), []
    for i in range(0, len(Xt), batch_size):
        out.append(model.embed(Xt[i:i+batch_size].to(device)).cpu())
    return torch.cat(out).numpy()


def pseudo_label(model, X_lab, y_lab, X_unl, config, device):
    """
    Embed labeled + unlabeled, then KNN-vote on unlabeled.

    Returns
    -------
    X_pseudo  : (M, F)  unlabeled windows that passed confidence threshold
    y_pseudo  : (M,)    their pseudo surface_id
    conf      : (M,)    confidence scores
    mask_kept : (N_u,)  boolean mask of which unlabeled rows were kept
    """
    print(f"  Embedding {len(X_lab)} labeled + {len(X_unl)} unlabeled windows...")
    emb_lab = get_embeddings(model, X_lab, device)
    emb_unl = get_embeddings(model, X_unl, device)

    k   = min(config["pseudo_k"], len(emb_lab) - 1)
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(emb_lab, y_lab)

    # predict_proba gives per-class probability = fraction of neighbours
    proba      = knn.predict_proba(emb_unl)          # (N_u, n_classes)
    confidence = proba.max(axis=1)                    # highest-vote fraction
    pseudo_lbl = knn.classes_[proba.argmax(axis=1)]  # winning class

    thresh     = config["pseudo_conf_threshold"]
    mask_kept  = confidence >= thresh

    print(f"  Pseudo-labeling: {mask_kept.sum()} / {len(X_unl)} unlabeled windows "
          f"above confidence threshold {thresh:.0%}")
    if mask_kept.sum() == 0:
        print("  WARNING: No unlabeled windows passed threshold. "
              "Try lowering pseudo_conf_threshold.")

    # Distribution of pseudo-labels
    for sid in sorted(np.unique(pseudo_lbl[mask_kept])):
        n = (pseudo_lbl[mask_kept] == sid).sum()
        print(f"    {surface_name(sid):30s}: {n} pseudo-labeled windows")

    return (X_unl[mask_kept], pseudo_lbl[mask_kept],
            confidence[mask_kept], mask_kept,
            confidence, pseudo_lbl)   # full arrays for output CSV


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def postprocess(embeddings, variance_threshold=0.95):
    X_norm = normalize(embeddings, norm="l2", axis=1)
    pca    = PCA(n_components=variance_threshold, svd_solver="full")
    X_pca  = pca.fit_transform(X_norm)
    print(f"  PCA: {embeddings.shape[1]}d -> {X_pca.shape[1]}d "
          f"(var={pca.explained_variance_ratio_.sum():.4f})")
    return X_norm, X_pca, pca


# ---------------------------------------------------------------------------
# Auto-select K
# ---------------------------------------------------------------------------
def auto_select_k(emb_pca, k_min=2, k_max=12):
    print(f"  Sweeping K {k_min}..{k_max} ...")
    scores = {}
    for k in range(k_min, k_max + 1):
        lbl       = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(emb_pca)
        scores[k] = silhouette_score(emb_pca, lbl)
        print(f"    K={k:2d}  sil={scores[k]:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"  Best K = {best_k}  (sil={scores[best_k]:.4f})")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(list(scores.keys()), list(scores.values()), marker="o", color="#457b9d")
    ax.axvline(best_k, color="#e63946", linestyle="--", label=f"Best K={best_k}")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Auto K selection")
    plt.tight_layout(); plt.savefig("auto_k_selection.png", dpi=150); plt.show()
    return best_k, scores


# ---------------------------------------------------------------------------
# Cluster -> surface mapping & helpers
# ---------------------------------------------------------------------------
def map_clusters_to_surfaces(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    c2s = {}
    for c in sorted(set(pred)):
        c2s[c] = -1 if c == -1 else int(pd.Series(gt[pred == c]).mode()[0])
    return c2s

def cluster_legend_label(cid, c2s):
    return "Noise" if cid == -1 else f"{surface_name(c2s.get(cid,'?'))} [c{cid}]"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(emb, pred, gt=None, metric="euclidean"):
    pred = np.asarray(pred); ok = pred != -1
    ev, lv = emb[ok], pred[ok]
    if len(np.unique(lv)) < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan, "ARI": np.nan, "NMI": np.nan,
                "Quality": "Only 1 cluster"}
    out = {"Silhouette"        : silhouette_score(ev, lv, metric=metric),
           "Davies-Bouldin"    : davies_bouldin_score(ev, lv),
           "Calinski-Harabasz" : calinski_harabasz_score(ev, lv)}
    if gt is not None:
        gv = np.asarray(gt)[ok]
        out["ARI"] = adjusted_rand_score(gv, lv)
        out["NMI"] = normalized_mutual_info_score(gv, lv)
    s = out["Silhouette"]
    out["Quality"] = "Excellent" if s > 0.6 else "Good" if s > 0.4 else "Fair" if s > 0.2 else "Poor"
    return out

def print_comparison(en, ep, pred, gt, split=""):
    rows = {"KMeans"       : evaluate(ep, pred["kmeans"], gt, "euclidean"),
            "Agglomerative": evaluate(en, pred["agg"],    gt, "cosine"),
            "DBSCAN"       : evaluate(ep, pred["dbscan"], gt, "euclidean")}
    df  = pd.DataFrame(rows).T
    tag = f" [{split}]" if split else ""
    print(f"\nClustering Comparison{tag} {'-'*max(0,44-len(tag))}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
COLORS = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261",
          "#8338ec","#06d6a0","#fb8500","#3a86ff","#ff006e"]

def _scatter2d(ax, xy, lbls, c2s, title):
    for i, lbl in enumerate(sorted(set(lbls))):
        m = lbls == lbl
        ax.scatter(xy[m,0], xy[m,1],
                   c="#aaa" if lbl == -1 else COLORS[i % len(COLORS)],
                   alpha=0.65, s=10, label=cluster_legend_label(lbl, c2s))
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, markerscale=1.5, framealpha=0.7, loc="best", title="Cluster -> Surface")
    ax.grid(alpha=0.2)

def _scatter2d_gt(ax, xy, surface_ids):
    """Ground truth panel: colour by true surface_id, label by surface name."""
    for i, sid in enumerate(sorted(np.unique(surface_ids))):
        m = surface_ids == sid
        ax.scatter(xy[m,0], xy[m,1],
                   c=COLORS[i % len(COLORS)], alpha=0.65, s=10,
                   label=surface_name(sid))
    ax.set_title("Ground Truth (true surface_id)", fontweight="bold", fontsize=10)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, markerscale=1.5, framealpha=0.7,
              loc="best", title="True Surface")
    ax.grid(alpha=0.2)


def _compute_projections(pca_emb, prefix=""):
    """Run t-SNE and UMAP on the same PCA embeddings. Returns (tsne_2d, umap_2d)."""
    perp = min(30, len(pca_emb) - 1)
    print(f"  [{prefix}] t-SNE ...")
    tsne_2d = TSNE(n_components=2, random_state=42,
                   perplexity=perp, max_iter=1000).fit_transform(pca_emb)
    print(f"  [{prefix}] UMAP ...")
    umap_2d = umap.UMAP(n_components=2, random_state=42,
                         n_neighbors=min(15, len(pca_emb)-1),
                         min_dist=0.1, metric="euclidean").fit_transform(pca_emb)
    return tsne_2d, umap_2d


def plot_embeddings_2d(tsne_coords, umap_coords, pred, c2s_map, gt, prefix="Test"):
    """
    4-column x 2-row grid:
      Rows    : t-SNE (top) | UMAP (bottom)
      Columns : K-Means | Agglomerative | DBSCAN | Ground Truth (true surface_id)

    All panels share the same embedding — only point colours differ.
    Ground truth uses the actual surface_id from the CSV, not a cluster mapping.
    """
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    fig.suptitle(f"Embeddings 2D — {prefix}", fontsize=14, fontweight="bold")

    panels = [
        ("kmeans", c2s_map["kmeans"], "K-Means"),
        ("agg",    c2s_map["agg"],    "Agglomerative"),
        ("dbscan", c2s_map["dbscan"], "DBSCAN"),
        (None,     None,              "Ground Truth"),
    ]

    for col, (key, c2s, title) in enumerate(panels):
        for row, (coords, proj_name) in enumerate([(tsne_coords, "t-SNE"),
                                                    (umap_coords, "UMAP")]):
            ax = axes[row, col]
            ax.set_xlabel(f"{proj_name} 1")
            ax.set_ylabel(f"{proj_name} 2")
            if key is None:
                # Ground truth: colour by true surface_id directly
                _scatter2d_gt(ax, coords, np.asarray(gt))
                ax.set_title(f"{proj_name} — {title}", fontweight="bold", fontsize=9)
            else:
                _scatter2d(ax, coords, np.asarray(pred[key]), c2s,
                           f"{proj_name} — {title}")

    plt.tight_layout()
    fname = f"embeddings_2d_{prefix.replace(' ','_').replace('—','-')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  Saved {fname}")


def plot_tsne_unlabeled(coords_all, labels_all, n_labeled, pseudo_labels_all, prefix="Unlabeled"):
    """
    Two-panel plot for the unlabeled windows embedded into the same space.
    Left:  coloured by pseudo-label (surface type prediction)
    Right: labeled (small, faded) vs unlabeled (bright) to show coverage
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Unlabeled Windows -- {prefix}", fontsize=13, fontweight="bold")

    # Panel 1: pseudo-label colours
    ax = axes[0]
    unique_psl = sorted(set(pseudo_labels_all))
    for i, lbl in enumerate(unique_psl):
        m = pseudo_labels_all == lbl
        label_str = surface_name(lbl) if lbl >= 0 else "Below threshold"
        ax.scatter(coords_all[m,0], coords_all[m,1],
                   c=COLORS[i % len(COLORS)], alpha=0.6, s=8, label=label_str)
    ax.set_title("Pseudo-label assignment", fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=7, markerscale=1.5, framealpha=0.7, loc="best")
    ax.grid(alpha=0.2)

    # Panel 2: labeled (faded) vs unlabeled (bright)
    ax = axes[1]
    ax.scatter(coords_all[:n_labeled,0], coords_all[:n_labeled,1],
               c="#cccccc", alpha=0.3, s=6, label="Labeled (context)")
    ax.scatter(coords_all[n_labeled:,0], coords_all[n_labeled:,1],
               c="#e63946", alpha=0.7, s=8, label="Unlabeled windows")
    ax.set_title("Labeled vs Unlabeled coverage", fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8, framealpha=0.7, loc="best")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fname = f"tsne_unlabeled_{prefix.replace(' ','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 0. Surface names ---
    print("\n[0/8] Surface names...")
    load_surface_names(CONFIG["surface_types_csv"])

    # --- 1. Load features (labeled + unlabeled) ---
    print("\n[1/8] Loading features CSV...")
    X_lab, y_lab, X_unl, scaler, feature_cols = load_features(CONFIG)
    has_unlabeled = len(X_unl) > 0
    print(f"  Feature dim: {len(feature_cols)}")
    if not has_unlabeled:
        print("  No unlabeled data found — running fully supervised mode.")

    # --- 2. Stratified split on labeled data ---
    print("\n[2/8] Stratified split (labeled data)...")
    tr_X, tr_lbl, te_X, te_lbl = stratified_split(
        X_lab, y_lab, test_size=CONFIG["test_size"], seed=CONFIG["split_seed"]
    )

    # --- 3. Phase 1: supervised contrastive training ---
    print("\n[3/8] Phase 1 — Supervised contrastive training (labeled train only)...")
    model = MLPEncoder(feature_dim=len(feature_cols),
                       embedding_dim=CONFIG["embedding_dim"]).to(device)
    model = train_supervised(model, tr_X, tr_lbl, CONFIG, device, tag="Phase1")

    # --- 4. Pseudo-labeling (skip if no unlabeled data) ---
    X_pseudo, y_pseudo = np.empty((0, len(feature_cols)), np.float32), np.empty(0, int)
    conf_all = pseudo_all = None

    if has_unlabeled:
        print("\n[4/8] Phase 2 — Pseudo-labeling unlabeled windows...")
        X_pseudo, y_pseudo, conf_kept, mask_kept, conf_all, pseudo_all = \
            pseudo_label(model, tr_X, tr_lbl, X_unl, CONFIG, device)

        # --- 5. Phase 3: semi-supervised fine-tune ---
        if len(X_pseudo) > 0:
            print(f"\n[5/8] Phase 3 — Semi-supervised fine-tune "
                  f"({len(tr_X)} labeled + {len(X_pseudo)} pseudo-labeled)...")
            model = train_semisupervised(model, tr_X, tr_lbl, X_pseudo, y_pseudo, CONFIG, device)
        else:
            print("\n[5/8] Phase 3 skipped — no pseudo-labeled windows above threshold.")
    else:
        print("\n[4/8] Pseudo-labeling skipped (no unlabeled data).")
        print("\n[5/8] Phase 3 skipped.")

    # --- 6. Embeddings + PCA ---
    print("\n[6/8] Embeddings...")
    tr_emb = get_embeddings(model, tr_X, device)
    te_emb = get_embeddings(model, te_X, device)
    print(f"  Train: {tr_emb.shape}  Test: {te_emb.shape}")

    tr_norm, tr_pca, pca = postprocess(tr_emb, CONFIG["pca_variance_threshold"])
    te_norm = normalize(te_emb, norm="l2", axis=1)
    te_pca  = pca.transform(te_norm)

    # --- 7. Clustering ---
    print("\n[7/8] Clustering...")
    n_detected = len(np.unique(tr_lbl))
    n_known    = len(SURFACE_NAMES) if SURFACE_NAMES else n_detected
    n_surf     = min(n_detected, n_known)
    print(f"  Classes in train: {n_detected}  |  Known: {n_known}  |  Using n_surf={n_surf}")
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

    # --- 8. Visualise ---
    print("\n[8/8] Visualising...")

    # 2-D t-SNE + UMAP on TEST split
    te_tsne, te_umap = _compute_projections(te_pca, prefix="Test")
    plot_embeddings_2d(te_tsne, te_umap, te_pred, te_c2s, te_lbl,
                       prefix="Accelerometer [Test] — handcrafted features")

    # t-SNE for unlabeled windows (joint embedding with labeled for context)
    if has_unlabeled and len(X_unl) > 0:
        print("  t-SNE + UMAP for unlabeled windows (joint with train for context)...")
        unl_emb     = get_embeddings(model, X_unl, device)
        joint       = normalize(np.vstack([tr_emb, unl_emb]), norm="l2", axis=1)
        pseudo_plot = pseudo_all.copy()
        pseudo_plot[~mask_kept] = -1
        j_tsne, j_umap = _compute_projections(joint, prefix="Unlabeled")
        for proj_coords, proj_name in [(j_tsne, "tSNE"), (j_umap, "UMAP")]:
            plot_tsne_unlabeled(proj_coords, None, len(tr_emb), pseudo_plot,
                                prefix=f"Semi-supervised [{proj_name}]")

        # --- Save predictions CSV ---
        out_df = pd.DataFrame({
            "pseudo_surface_id"  : pseudo_all,
            "pseudo_surface_name": [surface_name(s) for s in pseudo_all],
            "confidence"         : conf_all,
            "above_threshold"    : mask_kept,
            "cluster_kmeans"     : km.predict(pca.transform(normalize(unl_emb, norm="l2", axis=1))),
        })
        out_path = CONFIG["output_csv"]
        out_df.to_csv(out_path, index=False)
        print(f"\n  Unlabeled predictions saved -> {out_path}")
        print(f"  Rows: {len(out_df)}  |  Above threshold: {mask_kept.sum()}")
        print(out_df["pseudo_surface_name"].value_counts().to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()

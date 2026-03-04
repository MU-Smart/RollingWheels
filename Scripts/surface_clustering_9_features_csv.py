"""
Road Surface Classification -- Vibration Embedding & Clustering
===============================================================
Version 8: Load pre-extracted handcrafted features directly from CSV.

No raw signal loading, no resampling, no windowing, no FFT.
The CSV already contains one row per window with all handcrafted features.

Expected CSV layout (auto-detected):
  - A label column  : 'surface_id' | 'label' | 'surface' | 'class' (case-insensitive)
  - Only 'surface_id' is required as metadata; no file-id column needed.
  - All remaining numeric columns are treated as features

Pipeline:
  CSV -> StandardScaler -> (optional PCA) -> ConvEncoder (1-D conv over features)
       -> contrastive embeddings -> KMeans / Agglomerative / DBSCAN
       -> 2-D t-SNE on TEST split
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler, normalize
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
    "features_csv"          : Path("../Datasets/ExtractedFeatures/accelerometer_features.csv"),
    "surface_types_csv"     : Path("../Datasets/surface_types.csv"),
    "pca_variance_threshold": 0.95,

    # Model
    "embedding_dim" : 64,

    # Training
    "epochs"         : 200,
    "batch_size"     : 256,
    "lr"             : 3e-4,
    "temperature"    : 0.07,
    "hard_neg_ratio" : 0.5,

    # Train / test split (by FILE/RECORDING, not by window)
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
# Load features CSV
# ---------------------------------------------------------------------------
def _find_col(columns, candidates):
    """Return the first column whose name matches any candidate (case-insensitive)."""
    cl = [c.lower() for c in columns]
    for cand in candidates:
        if cand in cl:
            return columns[cl.index(cand)]
    return None


def load_features(config):
    """
    Load the handcrafted-features CSV.

    Returns
    -------
    features : np.ndarray  (N, F)   float32 — scaled feature matrix
    labels   : np.ndarray  (N,)     int     — surface_id per window
    feature_cols : list[str]                — names of the feature columns
    """
    path = config["features_csv"]
    print(f"  Reading {path} ...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}   Columns: {df.columns.tolist()}")

    # --- detect label column ---
    label_col = _find_col(df.columns,
                          ["surface_id", "surfaceid", "surface_type_id",
                           "label", "surface", "class", "target"])
    if label_col is None:
        raise ValueError(
            "Cannot find a label column. Expected one of: "
            "surface_id, label, surface, class, target.\n"
            f"Available columns: {df.columns.tolist()}"
        )
    print(f"  Label column   : '{label_col}'")

    # --- detect file-id column ---
    # --- drop index/unnamed columns ---
    drop_cols = [c for c in df.columns
                 if re.match(r"^unnamed", c, re.I) or c == "index"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # --- feature columns = everything numeric except label ---
    feature_cols = [c for c in df.columns
                    if c != label_col
                    and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Feature columns: {len(feature_cols)}  ({feature_cols[:8]} ...)")

    # --- extract arrays ---
    labels = df[label_col].astype(int).values
    X      = df[feature_cols].values.astype(np.float32)

    # --- drop rows with NaN/Inf ---
    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        print(f"  Dropping {bad.sum()} rows with NaN/Inf values.")
        X      = X[~bad]
        labels = labels[~bad]

    print(f"  Features after cleaning: {X.shape}")
    print(f"  Surface IDs in dataset : {sorted(np.unique(labels))}")

    return X, labels, feature_cols


# ---------------------------------------------------------------------------
# Stratified file-level split
# ---------------------------------------------------------------------------
def stratified_split(X, labels, test_size=0.2, seed=42):
    """
    Stratified split by surface class.
    Each class contributes test_size fraction of its windows to the test set.
    """
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
    print(f"  Total: {len(train_idx)} train windows | {len(test_idx)} test windows")
    return (X[train_idx], labels[train_idx],
            X[test_idx],  labels[test_idx])


# print_data_distribution is now inlined into stratified_split


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------
def make_balanced_sampler(labels):
    counts  = np.bincount(labels)
    weights = torch.tensor(1.0 / counts[labels], dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Model  — MLP encoder operating on flat feature vectors
#
# We use an MLP (not a Conv1d) because the input is now a flat feature
# vector (e.g. 100-300 handcrafted features), not a time-series.
# Architecture: feature_dim -> 512 -> 256 -> 128 -> embedding_dim
# Projection head: embedding_dim -> 256 -> proj_dim  (used only for loss)
# ---------------------------------------------------------------------------
class MLPEncoder(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int = 64, proj_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),         nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),         nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, proj_dim),
        )
        print(f"  MLPEncoder: in={feature_dim}  hidden=512/256/128  embed={embedding_dim}  proj={proj_dim}")

    def embed(self, x):
        return F.normalize(self.encoder(x), p=2, dim=-1)

    def forward(self, x):
        z = self.embed(x)
        p = F.normalize(self.projector(z), p=2, dim=-1)
        return z, p


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

        k         = max(1, int(neg_mask.sum(1).float().mean().item() * self.hard_neg_ratio))
        neg_sim   = sim.masked_fill(~neg_mask, -1e9)
        topk, _   = neg_sim.topk(k, dim=1)
        threshold = topk[:, -1:].detach()
        hard_neg  = neg_mask & (sim >= threshold)

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
# Training
# ---------------------------------------------------------------------------
def train(model, X, labels, config, device):
    Xt      = torch.tensor(X, dtype=torch.float32)
    Yt      = torch.tensor(labels, dtype=torch.long)
    sampler = make_balanced_sampler(labels)
    loader  = DataLoader(TensorDataset(Xt, Yt),
                         batch_size=config["batch_size"],
                         sampler=sampler, drop_last=True)

    criterion = HardNegSupConLoss(config["temperature"],
                                  config.get("hard_neg_ratio", 0.5)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_loss, patience_counter, PATIENCE = float("inf"), 0, 30

    for epoch in range(config["epochs"]):
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
        if improved:
            best_loss, patience_counter = avg, 0
        else:
            patience_counter += 1
        marker = " *" if improved else ""
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | Loss {avg:.4f}{marker}")
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    return model


@torch.no_grad()
def get_embeddings(model, X, device, batch_size=512):
    model.eval()
    Xt  = torch.tensor(X, dtype=torch.float32)
    out = []
    for i in range(0, len(Xt), batch_size):
        out.append(model.embed(Xt[i:i+batch_size].to(device)).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def postprocess(embeddings, variance_threshold=0.95):
    X_norm = normalize(embeddings, norm="l2", axis=1)
    pca    = PCA(n_components=variance_threshold, svd_solver="full")
    X_pca  = pca.fit_transform(X_norm)
    print(f"  Dims: {embeddings.shape[1]} -> {X_pca.shape[1]} "
          f"(variance: {pca.explained_variance_ratio_.sum():.4f})")
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
# Cluster -> surface mapping
# ---------------------------------------------------------------------------
def map_clusters_to_surfaces(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    c2s = {}
    for c in sorted(set(pred)):
        c2s[c] = -1 if c == -1 else int(pd.Series(gt[pred == c]).mode()[0])
    return c2s

def cluster_legend_label(cid, c2s):
    return "Noise" if cid == -1 else f"{surface_name(c2s.get(cid, '?'))} [c{cid}]"


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

    # --- 0. Surface names ---
    print("\n[0/6] Surface names...")
    load_surface_names(CONFIG["surface_types_csv"])

    # --- 1. Load pre-extracted features ---
    print("\n[1/6] Loading features CSV...")
    X, labels, feature_cols = load_features(CONFIG)

    # --- 2. Stratified split ---
    print("\n[2/6] Stratified split (by surface class)...")
    tr_X, tr_lbl, te_X, te_lbl = stratified_split(
        X, labels,
        test_size=CONFIG["test_size"], seed=CONFIG["split_seed"]
    )

    # --- Scale: fit on train, apply to both ---
    print("\n  Scaling features (StandardScaler fit on train)...")
    scaler = StandardScaler()
    tr_X   = scaler.fit_transform(tr_X).astype(np.float32)
    te_X   = scaler.transform(te_X).astype(np.float32)

    # --- 3. Train ---
    print(f"\n[3/6] Training MLP encoder  ({len(feature_cols)} features -> {CONFIG['embedding_dim']}-d)...")
    model = MLPEncoder(
        feature_dim   = len(feature_cols),
        embedding_dim = CONFIG["embedding_dim"],
    ).to(device)
    model = train(model, tr_X, tr_lbl, CONFIG, device)

    # --- 4. Embeddings ---
    print("\n[4/6] Embeddings...")
    tr_emb = get_embeddings(model, tr_X, device)
    te_emb = get_embeddings(model, te_X, device)
    print(f"  Train: {tr_emb.shape}  Test: {te_emb.shape}")

    print("  PCA (fit on train, applied to both)...")
    tr_norm, tr_pca, pca = postprocess(tr_emb, CONFIG["pca_variance_threshold"])
    te_norm = normalize(te_emb, norm="l2", axis=1)
    te_pca  = pca.transform(te_norm)
    print(f"  Test PCA shape: {te_pca.shape}")

    # --- 5. Clustering ---
    print("\n[5/6] Clustering...")
    n_detected = len(np.unique(tr_lbl))
    n_known    = len(SURFACE_NAMES) if SURFACE_NAMES else n_detected
    n_surf     = min(n_detected, n_known)
    print(f"  Classes in train: {n_detected}  |  Known surface types: {n_known}  |  Using n_surf={n_surf}")
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

    # --- Evaluate ---
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

    # --- 6. Visualise: 2-D t-SNE on TEST only ---
    print("\n[6/6] Visualising (2-D t-SNE, test split only)...")
    perp  = min(30, len(te_pca) - 1)
    te_2d = TSNE(n_components=2, random_state=42, perplexity=perp, max_iter=1000).fit_transform(te_pca)
    plot_tsne_2d_test(te_2d, te_pred, te_c2s, te_lbl,
                      prefix="Accelerometer [Test]  —  handcrafted features")

    print("\nDone!")


if __name__ == "__main__":
    main()

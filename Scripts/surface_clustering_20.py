"""
Road Surface Classification — Semi-Supervised Clustering
Version 10: clean, minimal pipeline.

Labeled rows   : surface_id is a known positive integer
Unlabeled rows : surface_id is 0, NaN, or negative

Steps
-----
1. Load CSV  →  split labeled / unlabeled  →  StandardScaler
2. Stratified 80/20 split on labeled data
3. Train MLP encoder with SupCon loss (labeled train only)
4. Pseudo-label unlabeled windows via KNN on embeddings
5. Fine-tune encoder on labeled + pseudo-labeled
6. Cluster (KMeans / Agglomerative / DBSCAN) on train embeddings
7. Evaluate on test split
8. Visualise: t-SNE + UMAP  (2×4 grid: 3 methods + ground truth)
9. Save unlabeled_predictions.csv
"""

import re
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
import umap as umap_lib
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "features_csv"       : Path("../Datasets/ExtractedFeatures/accelerometer_features.csv"),
    "surface_types_csv"  : Path("../Datasets/surface_types.csv"),
    "output_csv"         : Path("23_unlabeled_predictions.csv"),
    "unlabeled_id"       : 0,          # surface_id value meaning "no label"

    # model
    "embedding_dim"      : 128,

    # training
    "epochs"             : 200,
    "batch_size"         : 512,
    "lr"                 : 3e-4,
    "temperature"        : 0.07,
    "patience"           : 30,

    # pseudo-labeling
    "pseudo_k"           : 15,
    "pseudo_conf"        : 0.60,       # min neighbour-agreement to accept
    "pseudo_weight"      : 0.5,        # loss weight on pseudo-labeled batches
    "finetune_epochs"    : 75,
    "finetune_lr"        : 1e-4,

    # split
    "test_size"          : 0.2,
    "seed"               : 42,

    # PCA
    "pca_variance"       : 0.95,
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
# into a single super-class before training.  The encoder then learns
# ── Three grouping strategies based on visual TS similarity ──────────────────
#
# Visual observations from raw accelerometer plots:
#   • Paving Smooth (1), Rect Tiles (11)     — steady medium amplitude, even texture
#   • Concrete (2), Asphalt (5)              — similar moderate roughness
#   • Smooth Brick (3), Rough Brick (4)      — irregular but periodic spikes
#   • PB Rough (12)                          — higher amplitude irregular
#   • Indoor Carpet(6)/Lino(7)/Tile(8)       — low amplitude, quiet
#   • Curb Up (9), Curb Down (10)            — sharp isolated spike events
#
# Strategy A — 6 classes: amplitude-based broad grouping
#   0: Smooth outdoor hardscape  (1, 2, 5, 11)
#   1: Rough outdoor hardscape   (3, 4, 12)
#   2: Indoor smooth             (6, 7, 8)
#   3: Curb events               (9, 10)
#
# Strategy B — 5 classes: texture-based
#   0: Paving / Tiles            (1, 11)
#   1: Concrete + Asphalt        (2, 5)
#   2: All Brick + PB Rough      (3, 4, 12)
#   3: Indoor                    (6, 7, 8)
#   4: Curb                      (9, 10)
#
# Strategy C — 7 classes: fine-grained (keep rough/smooth brick split)
#   0: Paving / Tiles            (1, 11)
#   1: Concrete + Asphalt        (2, 5)
#   2: Smooth Brick              (3)
#   3: Rough Brick + PB Rough    (4, 12)
#   4: Indoor                    (6, 7, 8)
#   5: Curb Up                   (9)
#   6: Curb Down                 (10)

GROUPINGS = {
    "A_4class": {
        "map": {1:0, 2:0, 5:0, 11:0,
                3:1, 4:1, 12:1,
                6:2, 7:2, 8:2,
                9:3, 10:3},
        "names": {
            0: "Smooth Outdoor (Paving/Concrete/Asphalt/Tiles)",
            1: "Rough Outdoor (Brick/PB Rough)",
            2: "Indoor (Carpet/Lino/Tile)",
            3: "Curb (Up+Down)",
        },
        "label": "Strategy A — 4 classes (amplitude-based)",
        "file_prefix": "A",
    },
    "B_5class": {
        "map": {1:0, 11:0,
                2:1, 5:1,
                3:2, 4:2, 12:2,
                6:3, 7:3, 8:3,
                9:4, 10:4},
        "names": {
            0: "Paving / Rect. Tiles",
            1: "Concrete + Asphalt",
            2: "Brick + PB Rough",
            3: "Indoor (Carpet/Lino/Tile)",
            4: "Curb (Up+Down)",
        },
        "label": "Strategy B — 5 classes (texture-based)",
        "file_prefix": "B",
    },
    "C_7class": {
        "map": {1:0, 11:0,
                2:1, 5:1,
                3:2,
                4:3, 12:3,
                6:4, 7:4, 8:4,
                9:5,
                10:6},
        "names": {
            0: "Paving / Rect. Tiles",
            1: "Concrete + Asphalt",
            2: "Smooth Brick",
            3: "Rough Brick + PB Rough",
            4: "Indoor (Carpet/Lino/Tile)",
            5: "Curb Up",
            6: "Curb Down",
        },
        "label": "Strategy C — 7 classes (fine-grained)",
        "file_prefix": "C",
    },
}

# Active grouping — set by run_grouping(), do not edit directly
MERGE_MAP   = {}
SUPER_NAMES = {}
N_SUPER_CLASSES = 0

def set_grouping(key):
    global MERGE_MAP, SUPER_NAMES, N_SUPER_CLASSES
    g = GROUPINGS[key]
    MERGE_MAP       = g["map"]
    SUPER_NAMES     = g["names"]
    N_SUPER_CLASSES = len(SUPER_NAMES)
    print(f"  Grouping: {g['label']}  ({N_SUPER_CLASSES} classes)")
    for orig_id, cls in sorted(g["map"].items()):
        print(f"    {orig_id:2d}  {sname(orig_id):30s} -> {cls} ({SUPER_NAMES[cls]})")

def remap_labels(y):
    return np.array([MERGE_MAP.get(int(sid), int(sid)) for sid in y])

def super_name(sid):
    return SUPER_NAMES.get(int(sid), sname(sid))

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(cfg):
    df = pd.read_csv(cfg["features_csv"])
    # drop unnamed index columns
    df = df.loc[:, ~df.columns.str.match(r"^[Uu]nnamed|^index$")]

    # find label column
    label_col = next((c for c in df.columns
                      if c.lower() in ["surface_id","surfaceid","label","surface","class","target"]),
                     None)
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")

    feat_cols = [c for c in df.columns
                 if c != label_col and pd.api.types.is_numeric_dtype(df[c])]

    raw = pd.to_numeric(df[label_col], errors="coerce")
    unl_mask = raw.isna() | (raw == cfg["unlabeled_id"]) | (raw < 0)

    def extract(rows):
        X   = rows[feat_cols].values.astype(np.float32)
        ok  = np.isfinite(X).all(axis=1)
        lbl = pd.to_numeric(rows[label_col], errors="coerce").values
        return X[ok], lbl[ok]

    X_l, y_l = extract(df[~unl_mask])
    X_u, _   = extract(df[unl_mask])
    y_l = y_l.astype(int)

    scaler = StandardScaler().fit(X_l)
    X_l    = scaler.transform(X_l).astype(np.float32)
    X_u    = scaler.transform(X_u).astype(np.float32) if len(X_u) else \
             np.empty((0, X_l.shape[1]), np.float32)

    print(f"  Labeled: {len(X_l)}  Unlabeled: {len(X_u)}  Features: {len(feat_cols)}")
    print(f"  Classes: {sorted(np.unique(y_l))}")
    return X_l, y_l, X_u, feat_cols

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

# ── Model ─────────────────────────────────────────────────────────────────────
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim=128, proj_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512,    256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256,    128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, emb_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def embed(self, x):
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, x):
        z = self.embed(x)
        return z, F.normalize(self.projector(z), dim=-1)

# ── SupCon loss ───────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z, labels):
        B    = z.size(0)
        dev  = z.device
        sim  = z @ z.T
        eye  = torch.eye(B, dtype=torch.bool, device=dev)
        l2d  = labels.unsqueeze(1)
        pos  = (l2d == l2d.T) & ~eye
        neg  = ~pos & ~eye

        # hard negative mining — top 50% most similar negatives
        k        = max(1, int(neg.sum(1).float().mean().item() * 0.5))
        hard_neg = neg & (sim >= sim.masked_fill(~neg, -1e9).topk(k, dim=1).values[:, -1:])

        mask = pos | hard_neg
        s    = sim / self.t
        s    = s - s.masked_fill(~mask & ~pos, -1e9).max(1, keepdim=True).values.detach()
        exp  = torch.exp(s).masked_fill(~mask, 0)
        loss = -((s - torch.log(exp.sum(1, keepdim=True) + 1e-9)) * pos).sum(1) \
               / pos.sum(1).float().clamp(1)
        return loss[pos.any(1)].mean() if pos.any() else loss.mean()

# ── Training utilities ────────────────────────────────────────────────────────
def balanced_loader(X, y, batch_size, drop_last=True):
    counts  = np.bincount(y)
    weights = torch.tensor(1.0 / counts[y], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)),
                      batch_size=batch_size, sampler=sampler, drop_last=drop_last)

def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb)[1], yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def fit(model, X, y, cfg, device, epochs, lr, tag):
    loader    = balanced_loader(X, y, cfg["batch_size"])
    criterion = SupConLoss(cfg["temperature"]).to(device)
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, pat = float("inf"), 0
    for ep in range(epochs):
        loss = run_epoch(model, loader, criterion, opt, device)
        sched.step()
        imp  = loss < best - 1e-4
        best, pat = (loss, 0) if imp else (best, pat + 1)
        print(f"  [{tag}] {ep+1:3d}/{epochs}  loss={loss:.4f}" + (" *" if imp else ""))
        if pat >= cfg["patience"]:
            print(f"  Early stop at epoch {ep+1}"); break
    return model

# ── Embeddings ────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed(model, X, device, bs=512):
    model.eval()
    out = [model.embed(torch.tensor(X[i:i+bs]).to(device)).cpu()
           for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()

# ── Pseudo-labeling ───────────────────────────────────────────────────────────
def pseudo_label(model, X_tr, y_tr, X_unl, cfg, device):
    e_tr  = embed(model, X_tr,  device)
    e_unl = embed(model, X_unl, device)
    knn   = KNeighborsClassifier(n_neighbors=cfg["pseudo_k"], metric="cosine")
    knn.fit(e_tr, y_tr)
    prob  = knn.predict_proba(e_unl)
    conf  = prob.max(1)
    pred  = knn.classes_[prob.argmax(1)]
    keep  = conf >= cfg["pseudo_conf"]
    print(f"  Pseudo-labels: {keep.sum()}/{len(X_unl)} above threshold {cfg['pseudo_conf']:.0%}")
    for sid in sorted(np.unique(pred[keep])):
        print(f"    {super_name(sid):30s}: {(pred[keep]==sid).sum()}")
    return X_unl[keep], pred[keep], conf, pred, keep

# ── Semi-supervised fine-tune ─────────────────────────────────────────────────
def finetune(model, X_l, y_l, X_p, y_p, cfg, device):
    """Alternate labeled and pseudo-labeled batches, down-weight pseudo."""
    criterion = SupConLoss(cfg["temperature"]).to(device)
    opt       = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["finetune_lr"], weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cfg["finetune_epochs"])
    bs        = cfg["batch_size"] // 2
    l_loader  = balanced_loader(X_l, y_l, bs)
    p_loader  = balanced_loader(X_p, y_p, bs)
    pw        = cfg["pseudo_weight"]
    best, pat = float("inf"), 0

    for ep in range(cfg["finetune_epochs"]):
        model.train(); total = 0.0
        for (xl, yl), (xp, yp) in zip(l_loader, p_loader):
            xl, yl = xl.to(device), yl.to(device)
            xp, yp = xp.to(device), yp.to(device)
            opt.zero_grad()
            loss = criterion(model(xl)[1], yl) + pw * criterion(model(xp)[1], yp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()
        sched.step()
        avg = total / len(l_loader)
        imp = avg < best - 1e-4
        best, pat = (avg, 0) if imp else (best, pat + 1)
        print(f"  [Finetune] {ep+1:3d}/{cfg['finetune_epochs']}  loss={avg:.4f}" + (" *" if imp else ""))
        if pat >= cfg["patience"]: print(f"  Early stop"); break
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
    plt.tight_layout(); plt.show()
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

def experiment_fixed_k(pca_emb, ts, um, ks=(3, 5, 7, 11)):
    print(f"\n  Fixed-K experiment  K={list(ks)}")
    rows_ts = _fixed_k_grid(pca_emb, ts, "t-SNE", "20_fixed_k_tsne.png", ks)
    rows_um = _fixed_k_grid(pca_emb, um, "UMAP",  "21_fixed_k_umap.png", ks)

    print("\n  Cluster quality — t-SNE projection:")
    _print_quality(rows_ts)
    print("\n  Cluster quality — UMAP projection:")
    _print_quality(rows_um)
    return pd.DataFrame(rows_ts).T

def _print_quality(rows):
    print(f"  {'K':<8} {'Silhouette':>12} {'Davies-Bouldin':>16}  Quality")
    print("  " + "-" * 46)
    for k, r in rows.items():
        sil, db = r["Silhouette"], r["DB"]
        q = "Excellent" if sil > 0.6 else "Good" if sil > 0.4 else "Fair" if sil > 0.2 else "Poor"
        print(f"  {k:<8} {sil:>12.4f} {db:>16.4f}  {q}")

def cluster_all(tr_norm, tr_pca, te_norm, te_pca, k):
    km  = KMeans(n_clusters=k, random_state=42, n_init=20).fit(tr_pca)
    agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(tr_norm)
    d   = NearestNeighbors(n_neighbors=10).fit(tr_pca).kneighbors(tr_pca)[0]
    eps = float(np.percentile(d[:, -1], 90))
    db  = DBSCAN(eps=eps, min_samples=10).fit(tr_pca)

    tr_pred = {"kmeans": km.labels_, "agg": agg.labels_, "dbscan": db.labels_}
    knn_agg = KNeighborsClassifier(n_neighbors=5).fit(tr_norm, agg.labels_)
    te_pred = {
        "kmeans": km.predict(te_pca),
        "agg"   : knn_agg.predict(te_norm),
        "dbscan": DBSCAN(eps=eps, min_samples=10).fit_predict(te_pca),
    }
    return tr_pred, te_pred, km, eps

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
            "DBSCAN" : evaluate(ep, pred["dbscan"], gt),
        }
        print(f"\n── {split} ──────────────────────────────────────")
        print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nGeneralisation gap  (train ARI − test ARI):")
    for m, ek, em in [("KMeans","kmeans","euclidean"),
                      ("Agg","agg","cosine"),
                      ("DBSCAN","dbscan","euclidean")]:
        tr_v = evaluate(tr_pca if em=="euclidean" else tr_norm, tr_pred[ek], tr_lbl, em)["ARI"] or 0
        te_v = evaluate(te_pca if em=="euclidean" else te_norm, te_pred[ek], te_lbl, em)["ARI"] or 0
        gap  = (tr_v or 0) - (te_v or 0)
        flag = "good" if gap < 0.10 else "overfit" if gap > 0.20 else "acceptable"
        print(f"  {m:8s}  train={tr_v:.4f}  test={te_v:.4f}  gap={gap:+.4f}  [{flag}]")

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
    print(f"  UMAP  [{tag}]...")
    um = umap_lib.UMAP(n_components=2, random_state=42,
                       n_neighbors=min(15, len(pca_emb)-1),
                       min_dist=0.1, metric="euclidean").fit_transform(pca_emb)
    return ts, um

def _plot_proj_grid(coords, proj_name, pred, c2s, gt, fname):
    """1-row × 4-col: K-Means | Agglomerative | DBSCAN | Ground Truth."""
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.suptitle(f"{proj_name} — Test split (super-classes)",
                 fontsize=13, fontweight="bold")
    methods = [
        ("kmeans", lambda lbl: f"{super_name(c2s['kmeans'].get(lbl,'?'))} [c{lbl}]" if lbl!=-1 else "Noise", "K-Means"),
        ("agg",    lambda lbl: f"{super_name(c2s['agg'].get(lbl,'?'))} [c{lbl}]"    if lbl!=-1 else "Noise", "Agglomerative"),
        ("dbscan", lambda lbl: f"{super_name(c2s['dbscan'].get(lbl,'?'))} [c{lbl}]" if lbl!=-1 else "Noise", "DBSCAN"),
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

def plot_grid(ts, um, pred, c2s, gt):
    _plot_proj_grid(ts, "t-SNE", pred, c2s, gt, fname="18_tsne_test_clusters.png")
    _plot_proj_grid(um, "UMAP",  pred, c2s, gt, fname="19_umap_test_clusters.png")

# ── Per-grouping pipeline ─────────────────────────────────────────────────────
def run_grouping(gkey, X_l, y_l_raw, X_u, feat_cols, has_unl, device, fnum):
    g = GROUPINGS[gkey]
    pfx = g["file_prefix"]
    print(f"\n{'='*70}")
    print(f"  {g['label']}")
    print(f"{'='*70}")

    set_grouping(gkey)

    # split + remap
    tr_X, tr_y_raw, te_X, te_y_raw = stratified_split(
        X_l, y_l_raw, CONFIG["test_size"], CONFIG["seed"])
    tr_y = remap_labels(tr_y_raw)
    te_y = remap_labels(te_y_raw)
    print("  Class distribution (train):")
    for sc, n in sorted(zip(*np.unique(tr_y, return_counts=True))):
        print(f"    {super_name(sc):40s}: {n}")

    # train
    model = MLPEncoder(len(feat_cols), CONFIG["embedding_dim"]).to(device)
    model = fit(model, tr_X, tr_y, CONFIG, device,
                CONFIG["epochs"], CONFIG["lr"], f"SupCon-{pfx}")

    # pseudo-label + finetune
    if has_unl:
        X_ps, y_ps, conf_all, pred_all, keep_mask =             pseudo_label(model, tr_X, tr_y, X_u, CONFIG, device)
        if len(X_ps) > 0:
            model = finetune(model, tr_X, tr_y, X_ps, y_ps, CONFIG, device)
    else:
        conf_all = pred_all = keep_mask = None

    # embeddings + PCA
    tr_emb = embed(model, tr_X, device)
    te_emb = embed(model, te_X, device)
    tr_norm, tr_pca, te_norm, te_pca, pca = pca_reduce(tr_emb, te_emb, CONFIG["pca_variance"])

    # clustering
    k = best_k(tr_pca, max(2, N_SUPER_CLASSES-1), N_SUPER_CLASSES+2)
    plt.savefig(f"{fnum:02d}_{pfx}_k_selection.png", dpi=150, bbox_inches="tight")
    plt.close(); fnum += 1

    tr_pred, te_pred, km, eps = cluster_all(tr_norm, tr_pca, te_norm, te_pca, k)
    tr_c2s = {m: c2s_map(tr_pred[m], tr_y) for m in tr_pred}
    te_c2s = {m: c2s_map(te_pred[m], te_y) for m in te_pred}

    print(f"  K-Means cluster -> surface (train):")
    for cid, sid in sorted(tr_c2s["kmeans"].items()):
        print(f"    cluster {cid:2d} -> {super_name(sid)}")

    # metrics
    print_metrics(tr_norm, tr_pca, tr_pred, tr_y,
                  te_norm, te_pca, te_pred, te_y)

    # projections
    ts, um = project(te_pca, pfx)

    # t-SNE grid
    _plot_proj_grid(ts, "t-SNE", te_pred, te_c2s, te_y,
                    fname=f"{fnum:02d}_{pfx}_tsne_clusters.png")
    fnum += 1

    # UMAP grid
    _plot_proj_grid(um, "UMAP", te_pred, te_c2s, te_y,
                    fname=f"{fnum:02d}_{pfx}_umap_clusters.png")
    fnum += 1

    # fixed-K experiment (t-SNE 2x2)
    _fixed_k_grid(te_pca, ts, "t-SNE",
                  f"{fnum:02d}_{pfx}_fixed_k_tsne.png", ks=(3, 5, 7, 11))
    fnum += 1
    _fixed_k_grid(te_pca, um, "UMAP",
                  f"{fnum:02d}_{pfx}_fixed_k_umap.png", ks=(3, 5, 7, 11))
    fnum += 1

    # unlabeled
    if has_unl and pred_all is not None:
        unl_emb = embed(model, X_u, device)
        unl_n   = normalize(unl_emb, norm="l2")
        unl_pca = pca.transform(unl_n)
        ts_u, um_u = project(unl_pca, f"{pfx}-Unl")
        pseudo_plot = pred_all.copy().astype(int)
        pseudo_plot[~keep_mask] = -1

        for coords, pname in [(ts_u, "tSNE"), (um_u, "UMAP")]:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"{g['label']} — Unlabeled [{pname}]", fontweight="bold")
            for i, sid in enumerate(sorted(set(pseudo_plot))):
                m = pseudo_plot == sid
                axes[0].scatter(coords[m,0], coords[m,1],
                                c="#bbb" if sid==-1 else COLORS[i%len(COLORS)],
                                alpha=0.6, s=8,
                                label="Below threshold" if sid==-1 else super_name(sid))
            axes[0].set_title("Pseudo-label assignment")
            axes[0].legend(fontsize=6); axes[0].grid(alpha=0.2)
            sc = axes[1].scatter(coords[:,0], coords[:,1],
                                 c=conf_all, cmap="RdYlGn", vmin=0, vmax=1,
                                 alpha=0.6, s=8)
            plt.colorbar(sc, ax=axes[1], label="Confidence")
            axes[1].set_title("Confidence heatmap"); axes[1].grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(f"{fnum:02d}_{pfx}_unlabeled_{pname}.png",
                        dpi=150, bbox_inches="tight"); plt.show()
            fnum += 1

        out = pd.DataFrame({
            "grouping"           : gkey,
            "pseudo_surface_id"  : pred_all,
            "pseudo_surface_name": [super_name(s) for s in pred_all],
            "confidence"         : conf_all,
            "accepted"           : keep_mask,
            "cluster_kmeans"     : km.predict(unl_pca),
        })
        csv_path = f"{fnum:02d}_{pfx}_unlabeled_predictions.csv"
        out.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}  ({keep_mask.sum()} accepted)")
        fnum += 1

    return fnum


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("[0] Surface names")
    load_surface_names(CONFIG["surface_types_csv"])

    print("\n[1] Load data")
    X_l, y_l_raw, X_u, feat_cols = load_data(CONFIG)
    has_unl = len(X_u) > 0

    # Run all 3 grouping strategies; file numbers start at 18
    fnum = 18
    for gkey in ["A_4class", "B_5class", "C_7class"]:
        fnum = run_grouping(gkey, X_l, y_l_raw, X_u, feat_cols,
                            has_unl, device, fnum)

    print("\nAll groupings done!")

if __name__ == "__main__":
    main()

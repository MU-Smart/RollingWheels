"""
Road Surface Classification — TS2Vec Embedding & Clustering
Converted from ClusteringPretrainedTS2Vec.ipynb

Steps
-----
1. Load CSV files  →  extract windows with labels
2. Stratified 80/20 split on windows
3. Train TS2Vec on train split only
4. Generate embeddings for train + test
5. PCA reduce
6. Best-K sweep (silhouette)
7. Cluster (KMeans / Agglomerative / GMM) on train embeddings
8. Evaluate on test split (Silhouette, DB, CH, ARI, NMI)
9. Save predictions CSV
"""

import glob
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ts2vec import TS2Vec

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "clustering_ts2vec.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir"      : Path("../Datasets/Processed_Data/Labeled_Data_Without_GPS"),
    "window_size"   : 1024,
    "overlap"       : 0.5,
    "acc_epochs"    : 100,
    "output_dims"   : 256,
    "test_size"     : 0.2,
    "seed"          : 42,
    "pca_variance"  : 0.95,
    "output_dir"    : Path(__file__).parent,
}

# ── Data loading ──────────────────────────────────────────────────────────────
def extract_surface_type_id(path: str) -> int | None:
    match = re.search(r"SurfaceTypeID_(\d+)", path)
    return int(match.group(1)) if match else None


def load_file_index(data_dir: Path) -> pd.DataFrame:
    file_paths = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    df = pd.DataFrame(
        {
            "full_path": file_paths,
            "filename": [os.path.basename(p) for p in file_paths],
        }
    )
    df["surface_id"] = df["full_path"].apply(extract_surface_type_id)
    log.info("Surface-ID distribution:\n%s", df["surface_id"].value_counts().to_string())
    return df


# ── Windowing ─────────────────────────────────────────────────────────────────
def extract_windows(files_df: pd.DataFrame, window_size: int, overlap: float):
    step_size = int(window_size * (1 - overlap))
    acc_windows, acc_labels = [], []

    for _, row in tqdm(files_df.iterrows(), total=len(files_df), desc="Processing files"):
        file_path = row["full_path"]
        surface_id = row["surface_id"]

        if "accelerometer" not in file_path.lower():
            continue

        data_df = pd.read_csv(file_path)

        # Pad if shorter than one window
        if len(data_df) < window_size:
            pad_size = window_size - len(data_df)
            pad_df = pd.concat([data_df] * (pad_size // len(data_df) + 1)).iloc[:pad_size]
            data_df = pd.concat([data_df, pad_df], ignore_index=True)

        # Pad last incomplete window
        remainder = len(data_df) % step_size
        if remainder != 0:
            pad_size = window_size - remainder
            pad_df = data_df.iloc[-pad_size:].copy()
            data_df = pd.concat([data_df, pad_df], ignore_index=True)

        for start in range(0, len(data_df) - window_size + 1, step_size):
            window = data_df.iloc[start : start + window_size]
            xyz = window[["valueX", "valueY", "valueZ"]].values  # (window_size, 3)
            acc_windows.append(xyz)
            acc_labels.append(surface_id)

    log.info("Extracted %d accelerometer windows.", len(acc_windows))
    return np.array(acc_windows, dtype=np.float32), np.array(acc_labels, dtype=int)


# ── Stratified split ──────────────────────────────────────────────────────────
def stratified_split(windows: np.ndarray, labels: np.ndarray, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    tr_i, te_i = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        idx = rng.permutation(idx)
        n = max(1, int(len(idx) * test_size))
        te_i.extend(idx[:n])
        tr_i.extend(idx[n:])
        log.info("  Surface %2d: %5d train  %4d test", cls, len(idx) - n, n)
    tr_i = np.array(tr_i)
    te_i = np.array(te_i)
    log.info("Split → train=%d  test=%d", len(tr_i), len(te_i))
    return windows[tr_i], labels[tr_i], windows[te_i], labels[te_i]


# ── Stdout → logger bridge (captures TS2Vec's internal print() calls) ─────────
class _LoggerWriter:
    def __init__(self, level):
        self._level = level
        self._buf = ""

    def write(self, msg):
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._level(line)

    def flush(self):
        if self._buf.strip():
            self._level(self._buf.strip())
            self._buf = ""


# ── Training ──────────────────────────────────────────────────────────────────
def train_ts2vec(windows_np: np.ndarray, n_epochs: int, device: str) -> tuple:
    log.info("Training TS2Vec — shape=%s, epochs=%d", windows_np.shape, n_epochs)
    model = TS2Vec(input_dims=3, device=device, output_dims=CONFIG["output_dims"])

    old_stdout = sys.stdout
    sys.stdout = _LoggerWriter(log.info)
    try:
        loss_log = model.fit(windows_np, n_epochs=n_epochs, verbose=True)
    finally:
        sys.stdout = old_stdout

    log.info("Training done. Final loss: %.6f", loss_log[-1])
    return model, loss_log


# ── Embeddings ────────────────────────────────────────────────────────────────
def generate_embeddings(model: TS2Vec, windows: np.ndarray, label: str) -> np.ndarray:
    log.info("Generating embeddings for %s (shape=%s)...", label, windows.shape)
    # encode returns (N, T, D); take mean over time axis → (N, D)
    emb = model.encode(windows, encoding_window="full_series")
    log.info("Embeddings shape for %s: %s", label, emb.shape)
    return emb.astype(np.float32)


# ── PCA reduction ─────────────────────────────────────────────────────────────
def pca_reduce(emb_tr: np.ndarray, emb_te: np.ndarray, variance: float):
    n_tr = normalize(emb_tr, norm="l2")
    n_te = normalize(emb_te, norm="l2")
    pca = PCA(n_components=variance, svd_solver="full").fit(n_tr)
    log.info("PCA: %dd → %dd (explained_var=%.3f)",
             emb_tr.shape[1], pca.n_components_, pca.explained_variance_ratio_.sum())
    return n_tr, pca.transform(n_tr), n_te, pca.transform(n_te), pca


# ── Best K selection ──────────────────────────────────────────────────────────
def best_k(pca_emb: np.ndarray, k_min: int, k_max: int) -> int:
    scores = {}
    for k in range(k_min, k_max + 1):
        lbl = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(pca_emb)
        scores[k] = silhouette_score(pca_emb, lbl)
        log.info("  K=%2d  silhouette=%.4f", k, scores[k])
    bk = max(scores, key=scores.get)
    log.info("Best K=%d  silhouette=%.4f", bk, scores[bk])
    return bk


# ── Clustering ────────────────────────────────────────────────────────────────
def cluster_all(tr_norm, tr_pca, te_norm, te_pca, k):
    km  = KMeans(n_clusters=k, random_state=42, n_init=20).fit(tr_pca)
    agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(tr_norm)
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(tr_pca)

    tr_pred = {"kmeans": km.labels_, "agg": agg.labels_, "gmm": gmm.predict(tr_pca)}
    knn_agg = KNeighborsClassifier(n_neighbors=5).fit(tr_norm, agg.labels_)
    te_pred = {
        "kmeans": km.predict(te_pca),
        "agg"   : knn_agg.predict(te_norm),
        "gmm"   : gmm.predict(te_pca),
    }
    return tr_pred, te_pred, km


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(emb, pred, gt, metric="euclidean"):
    pred = np.asarray(pred)
    ok = pred != -1
    ev, lv, gv = emb[ok], pred[ok], np.asarray(gt)[ok]
    if len(np.unique(lv)) < 2:
        return dict(Silhouette=float("nan"), DB=float("nan"),
                    CH=float("nan"), ARI=float("nan"), NMI=float("nan"))
    return dict(
        Silhouette = silhouette_score(ev, lv, metric=metric),
        DB         = davies_bouldin_score(ev, lv),
        CH         = calinski_harabasz_score(ev, lv),
        ARI        = adjusted_rand_score(gv, lv),
        NMI        = normalized_mutual_info_score(gv, lv),
    )


def log_metrics(tr_norm, tr_pca, tr_pred, tr_lbl,
                te_norm, te_pca, te_pred, te_lbl):
    for split, en, ep, pred, gt in [
        ("TRAIN", tr_norm, tr_pca, tr_pred, tr_lbl),
        ("TEST",  te_norm, te_pca, te_pred, te_lbl),
    ]:
        rows = {
            "KMeans": evaluate(ep, pred["kmeans"], gt),
            "Agg"   : evaluate(en, pred["agg"],    gt, "cosine"),
            "GMM"   : evaluate(ep, pred["gmm"],    gt),
        }
        df = pd.DataFrame(rows).T
        log.info("\n── %s ──\n%s", split, df.to_string(float_format=lambda x: f"{x:.4f}"))

    log.info("Generalisation gap (train ARI − test ARI):")
    for method, ek, em in [("KMeans", "kmeans", "euclidean"),
                            ("Agg",    "agg",    "cosine"),
                            ("GMM",    "gmm",    "euclidean")]:
        tr_v = evaluate(tr_pca if em == "euclidean" else tr_norm, tr_pred[ek], tr_lbl, em)["ARI"]
        te_v = evaluate(te_pca if em == "euclidean" else te_norm, te_pred[ek], te_lbl, em)["ARI"]
        gap  = (tr_v or 0) - (te_v or 0)
        flag = "good" if gap < 0.10 else "overfit" if gap > 0.20 else "acceptable"
        log.info("  %s  train=%.4f  test=%.4f  gap=%+.4f  [%s]",
                 method, tr_v, te_v, gap, flag)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== Road Surface Clustering — TS2Vec ===")
    log.info("Config: %s", CONFIG)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Using device: %s", device)

    out_dir = CONFIG["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] Load & window ────────────────────────────────────────────────────
    log.info("[1] Loading files and extracting windows")
    files_df = load_file_index(CONFIG["data_dir"])
    acc_np, labels = extract_windows(files_df, CONFIG["window_size"], CONFIG["overlap"])

    # ── [2] Stratified split ─────────────────────────────────────────────────
    log.info("[2] Stratified 80/20 split")
    tr_win, tr_lbl, te_win, te_lbl = stratified_split(
        acc_np, labels, CONFIG["test_size"], CONFIG["seed"]
    )

    # ── [3] Train TS2Vec (train split only) ──────────────────────────────────
    log.info("[3] Training TS2Vec on train split")
    acc_model, acc_loss_log = train_ts2vec(tr_win, CONFIG["acc_epochs"], device)

    model_path = out_dir / "ts2vec_accelerometer.pth"
    torch.save(acc_model.net.state_dict(), model_path)
    log.info("Model saved to %s", model_path)

    # Save loss log
    loss_path = out_dir / "ts2vec_loss_log.csv"
    pd.DataFrame({"epoch": range(1, len(acc_loss_log) + 1), "loss": acc_loss_log}).to_csv(
        loss_path, index=False
    )
    log.info("Loss log saved to %s", loss_path)

    # ── [4] Generate embeddings ──────────────────────────────────────────────
    log.info("[4] Generating embeddings")
    tr_emb = generate_embeddings(acc_model, tr_win, "train")
    te_emb = generate_embeddings(acc_model, te_win, "test")

    # Save raw embeddings
    np.save(out_dir / "embeddings_train.npy", tr_emb)
    np.save(out_dir / "embeddings_test.npy",  te_emb)
    np.save(out_dir / "labels_train.npy", tr_lbl)
    np.save(out_dir / "labels_test.npy",  te_lbl)
    log.info("Embeddings saved.")

    # ── [5] PCA ──────────────────────────────────────────────────────────────
    log.info("[5] PCA reduction (variance=%.2f)", CONFIG["pca_variance"])
    tr_norm, tr_pca, te_norm, te_pca, pca = pca_reduce(tr_emb, te_emb, CONFIG["pca_variance"])

    # ── [6] Best K ───────────────────────────────────────────────────────────
    n_classes = len(np.unique(tr_lbl))
    k_min, k_max = max(2, n_classes - 2), n_classes + 2
    log.info("[6] Best-K sweep: K=%d..%d  (n_classes=%d)", k_min, k_max, n_classes)
    k = best_k(tr_pca, k_min, k_max)

    # ── [7] Clustering ───────────────────────────────────────────────────────
    log.info("[7] Clustering with K=%d (KMeans / Agglomerative / GMM)", k)
    tr_pred, te_pred, km = cluster_all(tr_norm, tr_pca, te_norm, te_pca, k)

    log.info("K-Means cluster → majority surface (train):")
    for cid in sorted(set(tr_pred["kmeans"])):
        mask = tr_pred["kmeans"] == cid
        majority = int(pd.Series(tr_lbl[mask]).mode()[0])
        log.info("  cluster %2d → surface_id=%d  (%d windows)", cid, majority, mask.sum())

    # ── [8] Metrics ──────────────────────────────────────────────────────────
    log.info("[8] Metrics")
    log_metrics(tr_norm, tr_pca, tr_pred, tr_lbl,
                te_norm, te_pca, te_pred, te_lbl)

    # ── [9] Save predictions CSV ──────────────────────────────────────────────
    log.info("[9] Saving predictions")
    te_out = pd.DataFrame({
        "surface_id_gt"    : te_lbl,
        "cluster_kmeans"   : te_pred["kmeans"],
        "cluster_agg"      : te_pred["agg"],
        "cluster_gmm"      : te_pred["gmm"],
    })
    te_csv = out_dir / "ts2vec_test_predictions.csv"
    te_out.to_csv(te_csv, index=False)
    log.info("Test predictions saved to %s  (%d rows)", te_csv, len(te_out))

    tr_out = pd.DataFrame({
        "surface_id_gt"    : tr_lbl,
        "cluster_kmeans"   : tr_pred["kmeans"],
        "cluster_agg"      : tr_pred["agg"],
        "cluster_gmm"      : tr_pred["gmm"],
    })
    tr_csv = out_dir / "ts2vec_train_predictions.csv"
    tr_out.to_csv(tr_csv, index=False)
    log.info("Train predictions saved to %s  (%d rows)", tr_csv, len(tr_out))

    log.info("=== Done ===")


if __name__ == "__main__":
    main()

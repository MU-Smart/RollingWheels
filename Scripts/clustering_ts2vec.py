"""
Road Surface Classification — Vibration Embedding & Clustering
Converted from ClusteringPretrainedTS2Vec.ipynb
"""

import glob
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ts2vec import TS2Vec

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
    "data_dir": Path("../Datasets/Processed_Data/Labeled_Data_Without_GPS"),
    "window_size": 1024,
    "overlap": 0.5,
    "acc_epochs": 120,
    "output_dims": 128,
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
    return np.array(acc_windows, dtype=np.float32), np.array(acc_labels)


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
def train_ts2vec(windows_np: np.ndarray, n_epochs: int, device: str, label: str) -> list:
    log.info("Training TS2Vec for %s — shape=%s, epochs=%d", label, windows_np.shape, n_epochs)
    model = TS2Vec(input_dims=3, device=device, output_dims=CONFIG["output_dims"])

    old_stdout = sys.stdout
    sys.stdout = _LoggerWriter(log.info)
    try:
        loss_log = model.fit(windows_np, n_epochs=n_epochs, verbose=True)
    finally:
        sys.stdout = old_stdout

    log.info("Finished training %s. Final loss: %.6f", label, loss_log[-1])
    return model, loss_log


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_loss(acc_loss_log: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(acc_loss_log, linewidth=2)
    ax.set_title("Accelerometer — TS2Vec Training Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Loss plot saved to %s", out_path)


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

    files_df = load_file_index(CONFIG["data_dir"])

    acc_np, _ = extract_windows(files_df, CONFIG["window_size"], CONFIG["overlap"])

    acc_model, acc_loss_log = train_ts2vec(acc_np, CONFIG["acc_epochs"], device, "Accelerometer")

    script_dir = Path(__file__).parent
    torch.save(acc_model.net.state_dict(), script_dir / "ts2vec_accelerometer.pth")
    log.info("Model saved.")

    plot_loss(acc_loss_log, script_dir / "ts2vec_training_loss.png")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()

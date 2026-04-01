"""
Inference script for Bangla audio deepfake detection.

Mirrors the preprocessing + model definitions from:
`LSTM/Checkpoint 2/bangla_deepfake_detection.ipynb`

Label mapping (from the notebook):
  0 = Real
  1 = Fake
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import csv

import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Config (from the notebook)
# -------------------------
SAMPLE_RATE = 16000
AUDIO_DURATION = 5  # seconds
N_MFCC = 40
MAX_LEN_MFCC = 157
NUM_CLASSES = 2


def load_audio(filepath: str | Path, sr: int = SAMPLE_RATE, duration: int = AUDIO_DURATION):
    """Load audio file, resample to target sr, pad/truncate to fixed duration."""
    y, _sr_orig = librosa.load(str(filepath), sr=sr, duration=duration)
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return y, sr


def extract_mfcc(
    filepath: str | Path,
    sr: int = SAMPLE_RATE,
    duration: int = AUDIO_DURATION,
    n_mfcc: int = N_MFCC,
    max_len: int = MAX_LEN_MFCC,
):
    """Extract MFCC features from an audio file."""
    y, _ = load_audio(filepath, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, time)
    # Pad or truncate time axis
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T  # (time, n_mfcc) — sequence-first for LSTM


def extract_raw_audio(filepath: str | Path, sr: int = SAMPLE_RATE, duration: int = AUDIO_DURATION):
    """Load and normalize raw audio waveform (for WaveNet)."""
    y, _ = load_audio(filepath, sr=sr, duration=duration)
    # Normalize to [-1, 1]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y


# -----------------
# LSTM (from notebook)
# -----------------
class LSTMModel(nn.Module):
    """
    RNN-based LSTM model for deepfake audio detection using MFCC features.

    - Bidirectional LSTM layers
    - Dropout (0.5)
    - Fully connected layers with ReLU
    - Returns raw logits (use softmax for probabilities)
    """

    def __init__(
        self,
        input_size: int = N_MFCC,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time_steps, n_mfcc)
        lstm_out, (_h_n, _c_n) = self.lstm(x)
        out = lstm_out[:, -1, :]  # (batch, hidden*2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# --------------------
# WaveNet (from notebook)
# --------------------
class CausalConv1d(nn.Module):
    """Causal convolution: output at time t depends only on inputs at time <= t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future timesteps (causal)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class WaveNetResidualBlock(nn.Module):
    """Single WaveNet residual block with dilated causal convolution."""

    def __init__(self, residual_channels: int, skip_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.dilated_conv = CausalConv1d(
            residual_channels, 2 * residual_channels, kernel_size, dilation
        )
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x: torch.Tensor):
        out = self.dilated_conv(x)
        # Gated activation (tanh * sigmoid)
        tanh_out = torch.tanh(out[:, : out.size(1) // 2, :])
        sig_out = torch.sigmoid(out[:, out.size(1) // 2 :, :])
        gated = tanh_out * sig_out

        # Skip connection
        skip = self.skip_conv(gated)

        # Residual connection
        residual = self.residual_conv(gated) + x
        return residual, skip


class WaveNetClassifier(nn.Module):
    """
    WaveNet-based classifier for deepfake audio detection.
    Uses dilated causal convolutions with residual + skip connections.
    Returns raw logits (use softmax for probabilities).
    """

    def __init__(
        self,
        in_channels: int = 1,
        residual_channels: int = 32,
        skip_channels: int = 32,
        kernel_size: int = 2,
        num_blocks: int = 10,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.input_conv = CausalConv1d(in_channels, residual_channels, kernel_size=1)

        self.residual_blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2**i
            self.residual_blocks.append(
                WaveNetResidualBlock(residual_channels, skip_channels, kernel_size, dilation)
            )

        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, num_classes, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, length)
        x = self.input_conv(x)

        skip_sum = 0
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        out = F.relu(skip_sum)
        out = F.relu(self.output_conv1(out))
        out = self.output_conv2(out)
        out = self.global_pool(out).squeeze(-1)  # (batch, num_classes)
        return out


def _safe_torch_load_state_dict(path: Path, device: torch.device) -> Dict[str, Any]:
    # Torch >=2.1 supports weights_only=...; keep compatible with older versions.
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _iter_audio_files(root: Path) -> Iterable[Path]:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


@torch.no_grad()
def _predict_lstm(model: LSTMModel, device: torch.device, audio_path: Path) -> Dict[str, Any]:
    mfcc = extract_mfcc(audio_path).astype(np.float32)  # (157, 40)
    x = torch.from_numpy(mfcc).unsqueeze(0).to(device)  # (1, T, C)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    return {
        "lstm_pred": pred,
        "lstm_label": "fake" if pred == 1 else "real",
        "lstm_prob_real": float(probs[0]),
        "lstm_prob_fake": float(probs[1]),
    }


@torch.no_grad()
def _predict_wavenet(model: WaveNetClassifier, device: torch.device, audio_path: Path) -> Dict[str, Any]:
    raw = extract_raw_audio(audio_path).astype(np.float32)  # (80000,)
    x = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, L)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    return {
        "wavenet_pred": pred,
        "wavenet_label": "fake" if pred == 1 else "real",
        "wavenet_prob_real": float(probs[0]),
        "wavenet_prob_fake": float(probs[1]),
    }


def _resolve_default_samples_dir() -> Path:
    here = Path(__file__).resolve().parent
    cand = here / "samples"
    if cand.exists() and cand.is_dir():
        return cand
    # fallback: workspace-root "samples" (common convention)
    up2 = here.parent.parent
    cand2 = up2 / "samples"
    if cand2.exists() and cand2.is_dir():
        return cand2
    return cand  # best-effort, will error later with nice message


def main() -> int:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Run LSTM/WaveNet inference on audio samples.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=_resolve_default_samples_dir(),
        help="Directory containing audio files (recursively).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=here / "predictions.csv",
        help="Where to write CSV with predictions.",
    )
    parser.add_argument(
        "--model_lstm",
        type=Path,
        default=(here / "best_lstm_model.pth"),
        help="Path to LSTM .pth (state_dict).",
    )
    parser.add_argument(
        "--model_wavenet",
        type=Path,
        default=(here / "best_wavenet_model.pth"),
        help="Path to WaveNet .pth (state_dict). Optional; if missing WaveNet is skipped.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device preference.",
    )
    args = parser.parse_args()

    # Helpful diagnostics for common environment issues (seen on Windows/conda setups).
    np_ver = tuple(int(x) for x in np.__version__.split(".")[:2])
    if np_ver >= (2, 4):
        print(
            "WARNING: Detected NumPy >= 2.4. Some audio stack packages (e.g., numba/librosa) "
            "may fail. The training notebook pins numpy==2.3.3; consider using that environment."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input_dir not found or not a directory: {input_dir}")
        return 2

    audio_files = sorted(_iter_audio_files(input_dir))
    if not audio_files:
        print(f"ERROR: no audio files found under: {input_dir}")
        return 3

    # Load models
    if not args.model_lstm.exists():
        # Try to fall back to the checkpoint in the LSTM folder (as per your training run)
        alt = (here.parent.parent / "LSTM" / "Checkpoint 2" / "best_lstm_model.pth").resolve()
        if alt.exists():
            args.model_lstm = alt
        else:
            print(f"ERROR: LSTM model not found: {args.model_lstm}")
            return 4

    lstm = LSTMModel().to(device)
    lstm.load_state_dict(_safe_torch_load_state_dict(args.model_lstm, device))
    lstm.eval()

    wavenet: Optional[WaveNetClassifier] = None
    model_wavenet_path = args.model_wavenet
    if model_wavenet_path.exists():
        wavenet = WaveNetClassifier().to(device)
        wavenet.load_state_dict(_safe_torch_load_state_dict(model_wavenet_path, device))
        wavenet.eval()
    else:
        # optional fallback to LSTM checkpoint folder
        alt_wn = (here.parent.parent / "LSTM" / "Checkpoint 2" / "best_wavenet_model.pth").resolve()
        if alt_wn.exists():
            wavenet = WaveNetClassifier().to(device)
            wavenet.load_state_dict(_safe_torch_load_state_dict(alt_wn, device))
            wavenet.eval()

    rows: List[Dict[str, Any]] = []
    for p in audio_files:
        row: Dict[str, Any] = {
            "path": str(p),
            "filename": p.name,
        }
        try:
            row.update(_predict_lstm(lstm, device, p))
            if wavenet is not None:
                row.update(_predict_wavenet(wavenet, device, p))
        except Exception as e:
            row["error"] = repr(e)
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write CSV without pandas to avoid compiled-extension issues in mismatched environments.
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Print a concise summary
    print(f"Device: {device}")
    print(f"Input:  {input_dir} ({len(audio_files)} files)")
    print(f"Output: {args.output_csv}")
    preview_cols = [
        c
        for c in ("filename", "lstm_label", "lstm_prob_fake", "wavenet_label", "wavenet_prob_fake", "error")
        if any(c in r for r in rows)
    ]
    if preview_cols:
        # simple aligned print
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in preview_cols}
        header = " ".join(c.ljust(widths[c]) for c in preview_cols)
        print(header)
        print("-" * len(header))
        for r in rows:
            print(" ".join(str(r.get(c, "")).ljust(widths[c]) for c in preview_cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


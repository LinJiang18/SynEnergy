#!/usr/bin/env python3
"""Sample unconditional sequences from a trained model (pure noise, no masks)."""

import argparse
import json
import os
import sys

import numpy as np
import torch

from main_model import CSDI_Unconditional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_generated_npy_path(model_dir: str) -> str:
    """<model_dir>/<PREFIX>_generated_data.npy where PREFIX is the run folder name (e.g. Output/ECG -> ECG)."""
    tag = os.path.basename(os.path.normpath(model_dir))
    if not tag:
        tag = "run"
    return os.path.join(model_dir, f"{tag}_generated_data.npy")


def parse_args():
    p = argparse.ArgumentParser(
        description="Unconditional sampling. Use --dataset PREFIX to use Output/<PREFIX>/; "
        "num_samples defaults to value saved at training (filename sample_* or N)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "-E",
        "--dataset",
        type=str,
        metavar="PREFIX",
        dest="dataset",
        help="Same prefix as training: <DiffTime>/Output/<PREFIX>/ (e.g. -E ECG)",
    )
    g.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Training output folder containing model.pth and config.json",
    )
    g.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model.pth (config/norm in the same directory unless overridden)",
    )
    p.add_argument("--config_path", type=str, default=None, help="config.json override")
    p.add_argument("--norm_stats_path", type=str, default=None, help="norm_stats.npz override")
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Override; default: num_samples from training config (data filename sample_* or N)",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Time length T; default: value saved in config.json (training data T)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npy path (N, T, 1). Default: <model_dir>/<PREFIX>_generated_data.npy (PREFIX = run folder name)",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.dataset is not None:
        prefix = args.dataset.strip()
        model_dir = os.path.join(_SCRIPT_DIR, "Output", prefix)
        model_path = os.path.join(model_dir, "model.pth")
    elif args.run_dir is not None:
        model_dir = os.path.abspath(args.run_dir)
        model_path = os.path.join(model_dir, "model.pth")
    else:
        model_path = os.path.abspath(args.model_path)
        model_dir = os.path.dirname(model_path)

    config_path = args.config_path or os.path.join(model_dir, "config.json")
    norm_path = args.norm_stats_path or os.path.join(model_dir, "norm_stats.npz")

    if not os.path.isfile(model_path):
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(config_path):
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    unc = config.get("unconditional") or {}
    seq_len = args.seq_len
    if seq_len is None:
        seq_len = unc.get("seq_len")
    if seq_len is None:
        print(
            "ERROR: seq_len not in config.json; pass --seq_len explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    seq_len = int(seq_len)

    num_samples = args.num_samples
    if num_samples is None:
        num_samples = unc.get("num_samples")
    if num_samples is None:
        print(
            "ERROR: num_samples not in config.json; re-train with current train_unconditional.py or pass --num_samples.",
            file=sys.stderr,
        )
        sys.exit(1)
    num_samples = int(num_samples)

    out_path = args.output
    if out_path is None:
        out_path = _default_generated_npy_path(model_dir)

    device = torch.device(args.device)
    model = CSDI_Unconditional(config, str(device)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    print(f"Generating num_samples={num_samples} (seq_len={seq_len})")

    arr = model.sample_unconditional(
        num_samples=num_samples,
        seq_len=seq_len,
        batch_size=args.batch_size,
    )
    if arr.shape != (num_samples, seq_len, 1):
        print(f"ERROR: unexpected output shape {arr.shape}", file=sys.stderr)
        sys.exit(1)

    if os.path.isfile(norm_path):
        z = np.load(norm_path)
        use_zscore = (
            bool(np.asarray(z["use_zscore"]).item())
            if "use_zscore" in z.files
            else False
        )
        if use_zscore:
            mean = float(np.asarray(z["mean"]).reshape(()))
            std = float(np.asarray(z["std"]).reshape(()))
            arr = arr * std + mean
            print(f"Inverse z-score applied: mean={mean:.6g}, std={std:.6g}")
    else:
        print("Warning: norm_stats.npz not found; saving in normalized space.", file=sys.stderr)

    np.save(out_path, arr.astype(np.float32))
    print(f"Saved generated data shape {arr.shape} (N, T, 1) to {out_path}")


if __name__ == "__main__":
    main()

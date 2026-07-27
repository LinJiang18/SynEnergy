#!/usr/bin/env python3
"""Export transportation (1320, 144, 1) array to multisample Parquet: 144 time steps × 1321 columns.

Same multisample convention as ``export_electricity_multisample_parquet.py``: fms-dgt requires
strictly more matching columns than ``train_samples``; we add TOY-1320 as a duplicate of TOY-0
so train_samples=1320 is valid while all 1320 real trajectories can be sampled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    default_npy = (
        repo_root / "Data/Extreme_Data/Transportation/transportation_seq_144_sample_1320.npy"
    )
    default_out = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "public"
        / "time_series"
        / "transportation_seq_144_sample_1320.parquet"
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=default_npy,
        help=(
            "Path to .npy shaped (n_samples, seq_len, 1), (n_samples, seq_len), "
            f"or flat (n_samples*seq_len,) (default: {default_npy})"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help=f"Output Parquet path (default: {default_out})",
    )
    p.add_argument("--n-samples", type=int, default=1320, help="Number of trajectories")
    p.add_argument("--seq-len", type=int, default=144, help="Time steps per trajectory")
    p.add_argument(
        "--column-prefix",
        type=str,
        default="TOY-",
        help='Column prefix (must match task train_channels / data), default "TOY-"',
    )
    args = p.parse_args()

    x = np.load(args.input)
    n = args.n_samples * args.seq_len

    if x.ndim == 1:
        if x.size != n:
            raise SystemExit(f"Expected {n} values, got {x.size} from {args.input}")
        mat = x.reshape(args.n_samples, args.seq_len)
    elif x.ndim == 2:
        if x.shape != (args.n_samples, args.seq_len):
            raise SystemExit(
                f"Expected shape ({args.n_samples}, {args.seq_len}), got {x.shape} from {args.input}"
            )
        mat = x
    elif x.ndim == 3:
        if x.shape != (args.n_samples, args.seq_len, 1):
            raise SystemExit(
                f"Expected shape ({args.n_samples}, {args.seq_len}, 1), got {x.shape} from {args.input}"
            )
        mat = x.reshape(args.n_samples, args.seq_len)
    else:
        raise SystemExit(f"Expected 1d, 2d, or 3d array, got shape {x.shape} from {args.input}")

    df = pd.DataFrame(
        mat.T,
        columns=[f"{args.column_prefix}{i}" for i in range(args.n_samples)],
    )
    dup_col = f"{args.column_prefix}{args.n_samples}"
    df[dup_col] = df[f"{args.column_prefix}0"].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {df.shape} -> {args.output}")


if __name__ == "__main__":
    main()

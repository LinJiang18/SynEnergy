#!/usr/bin/env python3
"""Export (200000, 1) toy series to multisample Parquet: 200 time steps × 1001 columns.

fms-dgt multisample requires strictly more matching columns than ``train_samples``
(see preprocess_multisample_data). We add TOY-1000 as a duplicate of TOY-0 so
train_samples=1000 is valid while all 1000 real trajectories can be sampled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    default_npy = repo_root / "Data/Extreme_Data/toyexample_timeseries/topexample_seq_200_sample_1000.npy"
    default_out = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "public"
        / "time_series"
        / "toyexample_seq_200_sample_1000.parquet"
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=default_npy,
        help=f"Path to .npy shaped (200000, 1) or (200000,) (default: {default_npy})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=default_out,
        help=f"Output Parquet path (default: {default_out})",
    )
    p.add_argument("--n-samples", type=int, default=1000, help="Number of real segments (default 1000)")
    p.add_argument("--seq-len", type=int, default=200, help="Length of each segment (default 200)")
    args = p.parse_args()

    x = np.load(args.input)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = args.n_samples * args.seq_len
    if x.size != n:
        raise SystemExit(f"Expected {n} values, got {x.size} from {args.input}")

    flat = x.reshape(-1)[:n]
    mat = flat.reshape(args.n_samples, args.seq_len)
    df = pd.DataFrame(
        mat.T,
        columns=[f"TOY-{i}" for i in range(args.n_samples)],
    )
    df[f"TOY-{args.n_samples}"] = df["TOY-0"].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {df.shape} -> {args.output}")


if __name__ == "__main__":
    main()

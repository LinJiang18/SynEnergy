#!/usr/bin/env python3
"""Train unconditional univariate time-series diffusion on a single (N, T, 1) .npy file."""

import argparse
import glob
import json
import os
import re
import sys
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset_npy import get_dataloader_npy
from main_model import CSDI_Unconditional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def default_name_from_data_path(data_path: str) -> str:
    """e.g. ECG_seq_250_sample_1000.npy -> ECG; mydata.npy -> mydata."""
    stem = os.path.splitext(os.path.basename(data_path))[0]
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def resolve_data_path_from_dataset_prefix(prefix: str) -> str:
    """Pick <DiffTime>/data/<prefix>*.npy (prefix match); sort; use first if multiple."""
    data_dir = os.path.join(_SCRIPT_DIR, "data")
    pattern = os.path.join(data_dir, f"{prefix}*.npy")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(
            f"ERROR: No file matching {prefix}*.npy under {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(matches) > 1:
        print(
            f"Warning: multiple matches for prefix {prefix!r}, using first:\n  {matches[0]}",
            file=sys.stderr,
        )
    return matches[0]


def parse_sample_count_from_stem(stem: str) -> Optional[int]:
    """e.g. ECG_seq_250_sample_1000 -> 1000 from 'sample_1000'."""
    m = re.search(r"(?i)sample_(\d+)", stem)
    return int(m.group(1)) if m else None


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def zscore_fit(data: np.ndarray):
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std < 1e-12:
        raise ValueError("Standard deviation is too small for z-score; use --no_zscore or check data.")
    return mean, std


def zscore_apply(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (data - mean) / std


def parse_args():
    p = argparse.ArgumentParser(
        description="Unconditional CSDI / DiffTime training. "
        "Defaults: batch_size=16, lr=1e-3, max_epochs=200; seq_len=T from data; "
        "output_dir=<this>/Output/<name>/. "
        "Use --dataset PREFIX to load <DiffTime>/data/PREFIX*.npy."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "-E",
        "--dataset",
        type=str,
        metavar="PREFIX",
        dest="dataset",
        help="Load data from <DiffTime>/data/<PREFIX>*.npy (e.g. -E ECG or --dataset ECG)",
    )
    g.add_argument(
        "--data_path",
        type=str,
        help="Explicit path to .npy with shape (N, T, 1)",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Subfolder under Output/ for this run (e.g. ECG). Default: filename prefix before first '_', or --dataset value.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory. Default: <DiffTime>/Output/<name>/",
    )
    p.add_argument("--config", type=str, default="config/base.yaml", help="Base YAML for diffusion/model hyperparameters")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--max_epochs", type=int, default=200, help="Ignored if --max_steps is set")
    p.add_argument("--max_steps", type=int, default=None, help="If set, train for this many optimizer steps and ignore max_epochs")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_zscore", action="store_true", help="Disable z-score normalization")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset is not None:
        data_path = resolve_data_path_from_dataset_prefix(args.dataset.strip())
    else:
        data_path = args.data_path

    raw = np.load(data_path, allow_pickle=False)
    if raw.ndim != 3 or raw.shape[-1] != 1:
        print(f"ERROR: Expected shape (N, T, 1), got {raw.shape}", file=sys.stderr)
        sys.exit(1)
    n0, t0, _ = raw.shape
    seq_len = int(t0)
    stem = os.path.splitext(os.path.basename(data_path))[0]
    num_samples_meta = parse_sample_count_from_stem(stem)
    if num_samples_meta is None:
        num_samples_meta = int(n0)

    run_name = args.name
    if run_name is None:
        run_name = (
            args.dataset.strip()
            if args.dataset is not None
            else default_name_from_data_path(data_path)
        )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(_SCRIPT_DIR, "Output", run_name)
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Data file: {data_path}")
    print(
        f"Loaded data shape: {tuple(raw.shape)} (N, T, 1), seq_len T={seq_len}; "
        f"sample count for generation default: {num_samples_meta} (from filename sample_* or N)"
    )
    print(f"Output directory: {output_dir} (name={run_name!r})")

    use_zscore = not args.no_zscore
    mean, std = 0.0, 1.0
    train_array = raw.astype(np.float32, copy=True)
    if use_zscore:
        mean, std = zscore_fit(train_array)
        train_array = zscore_apply(train_array, mean, std).astype(np.float32)
        np.savez(
            os.path.join(output_dir, "norm_stats.npz"),
            mean=np.float32(mean),
            std=np.float32(std),
            use_zscore=np.array(True),
        )
        print(f"Z-score: mean={mean:.6g}, std={std:.6g} (saved to norm_stats.npz)")
    else:
        np.savez(
            os.path.join(output_dir, "norm_stats.npz"),
            mean=np.float32(0.0),
            std=np.float32(1.0),
            use_zscore=np.array(False),
        )
        print("Normalization: disabled (identity; saved to norm_stats.npz)")

    tmp_path = os.path.join(output_dir, "_train_data.npy")
    np.save(tmp_path, train_array)

    cfg_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(_SCRIPT_DIR, args.config)
    )
    config = load_yaml_config(cfg_path)
    config["model"]["is_unconditional"] = True
    config["train"]["batch_size"] = args.batch_size
    config["train"]["lr"] = args.learning_rate
    config["unconditional"] = {
        "seq_len": seq_len,
        "name": run_name,
        "data_path": os.path.abspath(data_path),
        "dataset_prefix": args.dataset.strip() if args.dataset is not None else None,
        "num_samples": num_samples_meta,
    }
    if args.max_steps is None:
        config["train"]["epochs"] = args.max_epochs
    else:
        config["train"]["epochs"] = 10**9

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved training config to {os.path.join(output_dir, 'config.json')}")

    ckpt_path = os.path.join(output_dir, "model.pth")
    try:
        train_loader = get_dataloader_npy(
            tmp_path,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        device = torch.device(args.device)
        model = CSDI_Unconditional(config, str(device)).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-6
        )

        global_step = 0
        epoch = 0

        if args.max_steps is not None:
            pbar = tqdm(total=args.max_steps, mininterval=1.0, desc="train")
            model.train()
            train_iter = iter(train_loader)
            while global_step < args.max_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    epoch += 1
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                optimizer.zero_grad(set_to_none=True)
                loss = model(batch, is_train=1)
                loss.backward()
                optimizer.step()
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.6f}", epoch=epoch, refresh=False)
                pbar.update(1)
                if global_step % 100 == 0 or global_step == 1:
                    print(f"step={global_step} loss={loss.item():.6f}")
            pbar.close()
        else:
            max_epochs = config["train"]["epochs"]
            for epoch in range(max_epochs):
                model.train()
                running = 0.0
                n_batches = 0
                with tqdm(train_loader, mininterval=1.0, desc=f"epoch {epoch}") as it:
                    for batch in it:
                        optimizer.zero_grad(set_to_none=True)
                        loss = model(batch, is_train=1)
                        loss.backward()
                        optimizer.step()
                        running += loss.item()
                        n_batches += 1
                        it.set_postfix(loss=f"{loss.item():.6f}", refresh=False)
                avg = running / max(n_batches, 1)
                print(f"epoch={epoch + 1}/{max_epochs} train_loss={avg:.6f}")

        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to: {ckpt_path}")
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()

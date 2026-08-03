import os

import torch
import argparse
import numpy as np

from engine.logger import Logger, DecompTrace, save_trace_npz
from engine.solver import Trainer
from controlnet.pipeline import run_sample as controlnet_run_sample
from controlnet.pipeline import run_train as controlnet_run_train
from Data.build_dataloader import (
    build_dataloader,
    load_sample_cond,
    unnormalize_sample,
)
from Utils.io_utils import (
    load_yaml_config,
    resolve_pipeline_config,
    seed_everything,
    instantiate_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser("Diffusion-TS")

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Enable heavy-tailed (Student-t) noise for diffusion.",
    )
    parser.add_argument(
        "--condition",
        action="store_true",
        help="Enable conditional generation; cond is read from config file paths.",
    )
    parser.add_argument(
        "--controlnet",
        action="store_true",
        help="ControlNet pipeline: train AE + z_a then ControlNet (requires --condition).",
    )
    parser.add_argument(
        "--stage2",
        action="store_true",
        help="Deprecated alias for --controlnet.",
    )
    parser.add_argument(
        "--var_loss_weight",
        type=float,
        default=None,
        help="Override config model.params.var_loss_weight (weight for variance-matching loss).",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help=(
            "Seed for stratified condition sampling at sample time "
            "(overrides sample.params.sample_seed in config)."
        ),
    )

    args = parser.parse_args()
    if args.stage2:
        args.controlnet = True

    args.device = "cuda"
    args.output_dir = os.path.join("OUTPUT", args.name)

    return args


def _validate_condition_config(config, args):
    if not args.condition and not args.controlnet:
        return

    model_params = config.setdefault("model", {}).setdefault("params", {})
    if "cond_dim" not in model_params:
        raise ValueError("model.params.cond_dim is required when using --condition.")
    if "num_classes_per_cond" not in model_params:
        raise ValueError("model.params.num_classes_per_cond is required when using --condition.")

    cond_dim = int(model_params["cond_dim"])
    num_classes = model_params["num_classes_per_cond"]
    if len(num_classes) != cond_dim:
        raise ValueError(
            "model.params.num_classes_per_cond length must match cond_dim "
            f"({len(num_classes)} vs {cond_dim})."
        )

    if args.train or args.controlnet:
        cond_path = config.get("train", {}).get("params", {}).get("cond_path")
        if not cond_path:
            raise ValueError("train.params.cond_path is required when using --condition for training.")

    if args.sample:
        cond_path = config.get("sample", {}).get("params", {}).get("cond_path")
        if not cond_path:
            raise ValueError("sample.params.cond_path is required when using --condition for sampling.")


def main():
    args = parse_args()

    if args.controlnet and not args.condition:
        raise ValueError("--controlnet requires --condition.")

    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    config = load_yaml_config(args.config_file)
    config = resolve_pipeline_config(config, controlnet=bool(args.controlnet))
    config.setdefault("model", {}).setdefault("params", {})
    config["model"]["params"]["mode"] = bool(args.train or args.controlnet)
    config["model"]["params"]["heavy"] = bool(args.heavy)
    config["model"]["params"]["use_condition"] = bool(args.condition or args.controlnet)
    if args.var_loss_weight is not None:
        config["model"]["params"]["var_loss_weight"] = float(args.var_loss_weight)

    _validate_condition_config(config, args)

    if args.sample and args.sample_seed is not None:
        config.setdefault("sample", {}).setdefault("params", {})["sample_seed"] = int(
            args.sample_seed
        )

    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(args.output_dir)
    logger.save_config(config)

    if args.controlnet:
        model = instantiate_from_config(config["model"]).cuda()
        if args.train:
            controlnet_run_train(config, args, model, logger)
        else:
            controlnet_run_sample(config, args, model, logger)
        return

    model = instantiate_from_config(config["model"]).cuda()

    if args.train:
        dataloader_info = build_dataloader(config, args)
        dataloader = dataloader_info["dataloader"]
        trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader, logger=logger)
        trainer.train()
    else:
        sample_params = config["sample"]["params"]
        num_samples = sample_params["num_samples"]
        dataloader_info = build_dataloader(config, args)
        normalizer = dataloader_info["normalizer"]

        sample_cond = None
        sample_indices = None
        cfg_scale = float(config["model"]["params"].get("cfg_scale", 1.5))
        if args.condition:
            sample_cond, sample_indices = load_sample_cond(
                config, num_samples, return_indices=True
            )

        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=dataloader_info["dataloader"],
            logger=logger,
        )
        trainer.load("backbone")
        trace = None
        if sample_params.get("whe_trace"):
            trace = DecompTrace()
            trainer.ema.ema_model.set_trace(trace)
        samples = trainer.sample(
            num_samples=num_samples,
            batch_size=sample_params["batch_size"],
            sample_cond=sample_cond,
            cfg_scale=cfg_scale,
        )

        np.save(os.path.join(args.output_dir, f"generate_data_norm_{args.name}.npy"), samples)
        unnorm_samples = unnormalize_sample(samples, normalizer)
        np.save(os.path.join(args.output_dir, f"generate_data_unnorm_{args.name}.npy"), unnorm_samples)

        if args.condition and sample_cond is not None:
            np.save(os.path.join(args.output_dir, f"generate_cond_{args.name}.npy"), sample_cond)
            if sample_indices is not None:
                np.save(
                    os.path.join(args.output_dir, f"generate_sample_idx_{args.name}.npy"),
                    sample_indices,
                )

        if sample_params.get("whe_trace") and trace is not None:
            save_trace_npz(trace, os.path.join(args.output_dir, f"trace_{args.name}.npz"), normalizer)


if __name__ == "__main__":
    main()

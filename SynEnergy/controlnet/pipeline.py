"""ControlNet pipeline: AE + z_a then ControlNet training/sampling."""

from __future__ import annotations

import os

import numpy as np
import torch

from controlnet.ae.residual_learner import run_ae_phase
from controlnet.data.dataloader import (
    build_controlnet_dataloader,
    load_controlnet_sample_pairs,
    load_training_bundle,
)
from controlnet.engine.trainer import ControlNetTrainer
from controlnet.io.intermediate import load_intermediates, save_intermediates
from Data.build_dataloader import unnormalize_sample


def run_train(config, args, model, logger) -> None:
    """Run AE phase then ControlNet training in one pass."""
    bundle = load_training_bundle(config, args)
    origin_norm = bundle["origin_norm"]
    cond_array = bundle["cond_array"]
    normalizer = bundle["normalizer"]
    baseline_norm = bundle["baseline_norm"]

    ae_model, z_a, intermediates = run_ae_phase(
        data_norm=origin_norm,
        cond=cond_array,
        config=config,
        device=args.device,
        logger=logger,
        output_dir=args.output_dir,
        baseline_norm=baseline_norm,
    )
    save_intermediates(args.output_dir, ae_model, intermediates, config)
    if logger is not None:
        logger.log_info(f"Saved AE intermediates under {args.output_dir}/intermediate/")

    dataloader = build_controlnet_dataloader(
        config, args, origin_norm, cond_array, z_a, baseline_norm
    )
    trainer = ControlNetTrainer(
        config=config,
        args=args,
        model=model,
        dataloader=dataloader,
        normalizer=normalizer,
        ae_model=ae_model,
        logger=logger,
    )
    trainer.train()


def run_sample(config, args, model, logger) -> None:
    """Sample with trained ControlNet using stored z_a table."""
    sample_params = config["sample"]["params"]
    num_samples = sample_params["num_samples"]
    cfg_scale = float(config["model"]["params"].get("cfg_scale", 1.5))

    bundle = load_training_bundle(config, args)
    normalizer = bundle["normalizer"]

    sample_cond, sample_z_a, sample_indices = load_controlnet_sample_pairs(
        config, args.output_dir, num_samples
    )

    intermediates = load_intermediates(args.output_dir, config, device=args.device)
    ae_model = intermediates["ae_model"]

    trainer = ControlNetTrainer.for_sampling(
        config, args, model, ae_model=ae_model, logger=logger
    )
    trainer.load_controlnet(verbose=True)

    samples = trainer.sample(
        num_samples=num_samples,
        batch_size=sample_params["batch_size"],
        sample_cond=sample_cond,
        sample_z_a=sample_z_a,
        cfg_scale=cfg_scale,
        z_a_dropout=0.0,
    )

    np.save(os.path.join(args.output_dir, f"generate_data_norm_{args.name}.npy"), samples)
    unnorm_samples = unnormalize_sample(samples, normalizer)
    np.save(
        os.path.join(args.output_dir, f"generate_data_unnorm_{args.name}.npy"),
        unnorm_samples,
    )
    np.save(os.path.join(args.output_dir, f"generate_cond_{args.name}.npy"), sample_cond)
    np.save(os.path.join(args.output_dir, f"generate_z_a_{args.name}.npy"), sample_z_a)
    np.save(
        os.path.join(args.output_dir, f"generate_sample_idx_{args.name}.npy"),
        sample_indices,
    )

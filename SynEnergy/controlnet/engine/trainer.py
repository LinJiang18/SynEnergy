"""ControlNet trainer: frozen backbone + trainable ControlNet + norm-space anomaly loss."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from ema_pytorch import EMA
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm.auto import tqdm

from controlnet.ae.cond_grouped_residual import (
    cond_group_stats_path,
    load_cond_group_stats,
    uses_cond_grouped_residual,
)
from controlnet.data.dataloader import (
    resolve_backbone_output_dir,
    resolve_backbone_run_name,
)
from controlnet.loss.anomaly_loss import ControlNetAnomalyLoss
from Utils.io_utils import checkpoint_run_dir, get_model_parameters_info, instantiate_from_config


def cycle(dl):
    while True:
        for data in dl:
            yield data


class ControlNetTrainer:
    CONTROLNET_CKPT = "checkpoint_controlnet.pt"

    def __init__(
        self,
        config,
        args,
        model,
        dataloader=None,
        normalizer=None,
        ae_model=None,
        logger=None,
    ):
        self.model = model
        self.device = self.model.betas.device
        self.config = config
        self.args = args
        self.logger = logger
        self.normalizer = normalizer
        self.dataloader = dataloader
        self.dl = cycle(dataloader) if dataloader is not None else None

        cn = config.get("controlnet", config.get("stage2", {}))
        solver = config["solver"]
        self.train_num_steps = int(solver.get("max_epochs", 200))
        self.gradient_accumulate_every = int(solver.get("gradient_accumulate_every", 1))
        self.save_cycle = int(solver.get("save_cycle", 50))
        self.lambda_z = float(cn.get("lambda_z", 0.0))
        self.lambda_anom = float(cn.get("lambda_anom", 0.2))
        self.lambda_zero = float(cn.get("lambda_zero", 0.3))
        self.anom_on_x0 = bool(cn.get("anom_on_x0", True))
        self.use_aux_loss = (
            self.anom_on_x0
            or self.lambda_z > 0.0
            or self.lambda_anom > 0.0
            or self.lambda_zero > 0.0
        )
        self.log_frequency = 100
        self.step = 0

        base = solver["results_folder"]
        name = args.name
        seq_len = config["model"]["params"]["seq_length"]
        self.results_folder = checkpoint_run_dir(base, name, seq_len)
        os.makedirs(self.results_folder, exist_ok=True)

        self.model.freeze_backbone()
        self.model.unfreeze_controlnet()

        start_lr = float(solver.get("base_lr", 1e-4))
        ema_decay = solver["ema"]["decay"]
        ema_update_every = solver["ema"]["update_interval"]

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = Adam(trainable, lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(
            self.device
        )

        sc_cfg = solver["scheduler"]
        sc_cfg["params"]["optimizer"] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        self.anomaly_loss_fn = None
        self._ae_model = ae_model
        if ae_model is not None:
            seq_length = int(config["model"]["params"]["seq_length"])
            ae_cfg = {"num_days": cn.get("num_days", 30), "slots_per_day": cn.get("slots_per_day", 6), "seq_len": seq_length}
            zone_baseline = None
            zone_scale = None
            if uses_cond_grouped_residual(ae_cfg):
                stats = load_cond_group_stats(cond_group_stats_path(args.output_dir))
                zone_baseline = torch.as_tensor(
                    stats["baseline"], device=self.device, dtype=torch.float32
                )
                zone_scale = torch.as_tensor(
                    stats["scale"], device=self.device, dtype=torch.float32
                )
            self.anomaly_loss_fn = ControlNetAnomalyLoss(
                ae_model=ae_model,
                controlnet_cfg=cn,
                seq_length=seq_length,
                zone_baseline=zone_baseline,
                zone_scale=zone_scale,
            )
        self._register_ae_decoder(ae_model)

    def _register_ae_decoder(self, ae_model) -> None:
        if ae_model is None:
            return
        for target in (self.model, getattr(self.ema, "ema_model", None)):
            if target is None:
                continue
            register = getattr(target, "register_ae_decoder", None)
            if register is not None:
                register(ae_model)

    @classmethod
    def for_sampling(cls, config, args, model, ae_model=None, logger=None):
        """Minimal trainer for inference (no dataloader)."""
        return cls(
            config=config,
            args=args,
            model=model,
            ae_model=ae_model,
            logger=logger,
        )

    def _combine_loss(
        self,
        diff_loss: torch.Tensor,
        loss_terms: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return (
            diff_loss
            + self.lambda_z * loss_terms["loss_z"]
            + self.lambda_anom * loss_terms["loss_anom"]
            + self.lambda_zero * loss_terms["loss_zero"]
        )

    def _load_backbone(self) -> None:
        cn = self.config.get("controlnet", self.config.get("stage2", {}))
        backbone_run = resolve_backbone_run_name(self.config, self.args)
        seq_len = self.config["model"]["params"]["seq_length"]
        base = self.config["solver"]["results_folder"]
        backbone_dir = checkpoint_run_dir(base, backbone_run, seq_len)
        ckpt_name = cn.get("backbone_ckpt", "checkpoint_backbone.pt")
        ckpt_path = backbone_dir / ckpt_name
        if not ckpt_path.is_file():
            alt = Path(resolve_backbone_output_dir(self.config, self.args)) / ckpt_name
            if alt.is_file():
                ckpt_path = alt
            else:
                raise FileNotFoundError(
                    f"Backbone checkpoint not found: {ckpt_path} or {alt}"
                )
        data = torch.load(str(ckpt_path), map_location=self.device)
        state = data.get("model", data.get("ema", data))
        if isinstance(state, dict) and "model" in data and "ema" in data:
            state = data["model"]
        self.model.load_backbone_weights(state, strict=False)
        if self.logger is not None:
            self.logger.log_info(f"Loaded frozen backbone from {ckpt_path}")

    def save(self, verbose: bool = False) -> None:
        ckpt_path = self.results_folder / self.CONTROLNET_CKPT
        if self.logger is not None and verbose:
            self.logger.log_info(f"Save ControlNet to {ckpt_path}")
        payload = {
            "step": self.step,
            "controlnet": self.model.controlnet_state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "model": self.model.state_dict(),
        }
        torch.save(payload, str(ckpt_path))

    def load_controlnet(self, verbose: bool = False) -> None:
        ckpt_path = self.results_folder / self.CONTROLNET_CKPT
        if self.logger is not None and verbose:
            self.logger.log_info(f"Resume ControlNet from {ckpt_path}")
        data = torch.load(str(ckpt_path), map_location=self.device)
        if "controlnet" in data:
            self.model.load_controlnet_weights(data["controlnet"])
        elif "model" in data:
            self.model.load_state_dict(data["model"], strict=False)
        self.step = data.get("step", 0)
        if "opt" in data:
            self.opt.load_state_dict(data["opt"])
        if "ema" in data:
            self.ema.load_state_dict(data["ema"], strict=False)
        self._register_ae_decoder(getattr(self, "_ae_model", None))

    def train(self) -> None:
        if self.dl is None:
            raise RuntimeError("ControlNetTrainer.train() requires a dataloader.")
        if self.use_aux_loss and self.anomaly_loss_fn is None:
            raise RuntimeError("Auxiliary losses require ae_model.")

        self._load_backbone()
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info(f"{self.args.name}: start ControlNet training...")

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.0
                total_diff = 0.0
                total_z = 0.0
                total_anom = 0.0
                total_zero = 0.0
                total_w_z = 0.0
                total_w_anom = 0.0
                total_w_zero = 0.0

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    data, cond, z_a = batch[:3]
                    data = data.to(device)
                    cond = cond.to(device)
                    z_a = z_a.to(device)
                    baseline = batch[3].to(device) if len(batch) > 3 else None

                    if self.use_aux_loss:
                        diff_loss, x0_pred = self.model(
                            data, cond=cond, z_a=z_a, return_x0=True, target=data
                        )
                        loss_terms = self.anomaly_loss_fn(
                            x_gen=x0_pred,
                            x_ori=data.detach(),
                            cond=cond,
                            z_a=z_a if self.lambda_z > 0.0 else None,
                            baseline=baseline,
                        )
                        loss = self._combine_loss(diff_loss, loss_terms)
                    else:
                        diff_loss = self.model(data, cond=cond, z_a=z_a, target=data)
                        loss = diff_loss
                        loss_terms = {
                            "loss_z": torch.tensor(0.0, device=device),
                            "loss_anom": torch.tensor(0.0, device=device),
                            "loss_zero": torch.tensor(0.0, device=device),
                        }

                    loss = loss / self.gradient_accumulate_every
                    loss.backward()

                    total_loss += loss.item()
                    total_diff += diff_loss.item()
                    total_z += loss_terms["loss_z"].item()
                    total_anom += loss_terms["loss_anom"].item()
                    total_zero += loss_terms["loss_zero"].item()
                    total_w_z += self.lambda_z * loss_terms["loss_z"].item()
                    total_w_anom += self.lambda_anom * loss_terms["loss_anom"].item()
                    total_w_zero += self.lambda_zero * loss_terms["loss_zero"].item()

                pbar.set_description(
                    f"loss: {total_loss:.6f} diff: {total_diff:.6f} "
                    f"z: {total_z:.4f} anom: {total_anom:.4f} zero: {total_zero:.4f} "
                    f"w_z: {total_w_z:.4f} w_anom: {total_w_anom:.4f} w_zero: {total_w_zero:.4f}"
                )
                clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                if self.logger is not None and self.step % self.log_frequency == 0:
                    self.logger.add_scalar("train/loss", total_loss, self.step)
                    self.logger.add_scalar("train/diff_loss", total_diff, self.step)
                    self.logger.add_scalar("train/z_loss", total_z, self.step)
                    self.logger.add_scalar("train/anom_loss", total_anom, self.step)
                    self.logger.add_scalar("train/zero_loss", total_zero, self.step)

                if self.step % self.save_cycle == 0:
                    self.save(verbose=True)

                pbar.update(1)

        self.save(verbose=True)
        print("ControlNet training complete")
        if self.logger is not None:
            self.logger.log_info(
                f"ControlNet training done, time: {time.time() - tic:.2f}s"
            )

    def sample(
        self,
        num_samples: int,
        batch_size: int,
        sample_cond=None,
        sample_z_a=None,
        cfg_scale: float = 1.5,
        z_a_dropout: float = 0.0,
    ) -> np.ndarray:
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info(
                f"Begin ControlNet sampling {num_samples} (batch_size={batch_size})"
            )

        all_samples = []
        for b in range(0, num_samples, batch_size):
            actual = min(batch_size, num_samples - b)
            cond_batch = None
            z_batch = None
            if sample_cond is not None:
                cond_batch = torch.from_numpy(sample_cond[b : b + actual]).long().to(
                    self.device
                )
            if sample_z_a is not None:
                z_batch = torch.from_numpy(sample_z_a[b : b + actual]).float().to(
                    self.device
                )
            with torch.no_grad():
                sample = self.ema.ema_model.generate_mts(
                    batch_size=actual,
                    cond=cond_batch,
                    cfg_scale=cfg_scale,
                    z_a=z_batch,
                    z_a_dropout=z_a_dropout,
                )
            all_samples.append(sample.detach().cpu().numpy())

        samples = np.concatenate(all_samples, axis=0)
        if self.logger is not None:
            self.logger.log_info(
                f"Sampling done, {samples.shape[0]} samples, time: {time.time() - tic:.2f}s"
            )
        return samples

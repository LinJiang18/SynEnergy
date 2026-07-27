"""DiffusionTS with frozen backbone and trainable z_a ControlNet."""

from __future__ import annotations

from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from Models.interpretable_diffusion.controlnet import ZControlNet
from Models.interpretable_diffusion.gaussian_diffusion import DiffusionTS
from Models.interpretable_diffusion.model_utils import extract


class DiffusionTSControlNet(DiffusionTS):
    """
    Stage 2 model: tractID -> frozen backbone AdaLayerNorm;
    mirrored trainable Encoder + global z_a hint + AE decode dense map -> controls.
    """

    def __init__(
        self,
        z_a_dim: int = 64,
        z_a_dropout: float = 0.1,
        control_scale: float = 0.1,
        control_t_start: int = 100,
        use_decode_map: bool = True,
        decode_map_channels: int = 2,
        use_p_hat_gate: bool = False,
        gate_z_hint: bool = False,
        gate_control_output: bool = True,
        p_hat_gate_mode: str = "linear",
        p_hat_gate_power: float = 1.0,
        p_hat_gate_floor: float = 0.0,
        p_hat_gate_detach: bool = True,
        p_hat_gate_soft_tau: float = 0.1,
        p_hat_gate_soft_k: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.z_a_dim = int(z_a_dim)
        self.z_a_dropout = float(z_a_dropout)
        self.control_scale = float(control_scale)
        self.use_decode_map = bool(use_decode_map)
        self.decode_map_channels = int(decode_map_channels)
        # Ramp control in over t in [0, control_t_start); 0 disables ramp (constant scale).
        self.control_t_start = int(control_t_start)
        d_model = kwargs.get("d_model") or 64
        if d_model is None:
            d_model = 64
        n_layer_enc = kwargs.get("n_layer_enc", 2)
        n_heads = kwargs.get("n_heads", 4)
        attn_pd = kwargs.get("attn_pd", 0.0)
        resid_pd = kwargs.get("resid_pd", 0.0)
        mlp_hidden_times = kwargs.get("mlp_hidden_times", 4)

        self.controlnet = ZControlNet(
            z_a_dim=self.z_a_dim,
            seq_length=self.seq_length,
            d_model=d_model,
            n_layer_enc=n_layer_enc,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            use_decode_map=self.use_decode_map,
            decode_map_channels=self.decode_map_channels,
            use_p_hat_gate=use_p_hat_gate,
            gate_z_hint=gate_z_hint,
            gate_control_output=gate_control_output,
            p_hat_gate_mode=p_hat_gate_mode,
            p_hat_gate_power=p_hat_gate_power,
            p_hat_gate_floor=p_hat_gate_floor,
            p_hat_gate_detach=p_hat_gate_detach,
            p_hat_gate_soft_tau=p_hat_gate_soft_tau,
            p_hat_gate_soft_k=p_hat_gate_soft_k,
        )

    def register_ae_decoder(self, ae_decoder: nn.Module) -> None:
        """Register frozen AE on the control branch for decode(z_a) dense hints."""
        if not self.use_decode_map:
            return
        device = next(self.parameters()).device
        self.controlnet.register_ae_decoder(ae_decoder.to(device))

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            if not name.startswith("controlnet."):
                param.requires_grad = False

    def unfreeze_controlnet(self) -> None:
        for param in self.controlnet.parameters():
            param.requires_grad = True

    def _encoder_input_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone encoder input: Conv_MLP(x) + positional encoding (frozen weights)."""
        with torch.no_grad():
            emb = self.model.emb(x)
            return self.model.pos_enc(emb)

    def _control_timestep_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Per-sample control strength in [0, control_scale].

        Reverse diffusion runs t: T-1 -> 0. With control_t_start=100, control is off
        for t >= 100 (backbone-only early phase) and linearly ramps to control_scale
        at t=0 over the final 100 timesteps. Training uses the same schedule.
        Set control_t_start=0 to apply constant control_scale at all t.
        """
        base = float(self.control_scale)
        t_start = int(getattr(self, "control_t_start", 0))
        if t_start <= 0:
            return torch.full(
                (t.shape[0], 1, 1),
                base,
                device=t.device,
                dtype=torch.float32,
            )
        ramp = (float(t_start) - t.float()).clamp(min=0.0) / float(t_start)
        return (ramp * base).view(-1, 1, 1)

    def _control_signals(
        self,
        x: torch.Tensor,
        z_a: torch.Tensor,
        t: torch.Tensor,
        label_emb: Optional[torch.Tensor] = None,
        padding_masks: Optional[torch.Tensor] = None,
        z_a_dropout: Optional[float] = None,
    ) -> List[torch.Tensor]:
        dropout = self.z_a_dropout if z_a_dropout is None else float(z_a_dropout)
        x_emb = self._encoder_input_emb(x)
        controls = self.controlnet(
            z_a=z_a,
            t=t,
            x_emb=x_emb,
            label_emb=label_emb,
            z_a_dropout=dropout,
            padding_masks=padding_masks,
        )
        weight = self._control_timestep_weight(t)
        return [c * weight for c in controls]

    def output(
        self,
        x,
        t,
        padding_masks=None,
        label_emb=None,
        z_a=None,
        encoder_controls=None,
        z_a_dropout=None,
    ):
        if encoder_controls is None and z_a is not None:
            encoder_controls = self._control_signals(
                x,
                z_a,
                t,
                label_emb=label_emb,
                padding_masks=padding_masks,
                z_a_dropout=z_a_dropout,
            )
        trend, season, trend_cum, season_cum = self.model(
            x,
            t,
            padding_masks=padding_masks,
            label_emb=label_emb,
            encoder_controls=encoder_controls,
        )
        return trend + season

    def guided_output(
        self,
        x,
        t,
        cond=None,
        cfg_scale=1.0,
        padding_masks=None,
        z_a=None,
        z_a_dropout=None,
    ):
        if not self.use_condition or cond is None:
            return self.output(
                x,
                t,
                padding_masks=padding_masks,
                label_emb=None,
                z_a=z_a,
                z_a_dropout=z_a_dropout,
            )

        scale = float(cfg_scale)
        if scale == 1.0:
            label_emb = self._encode_cond(cond, cond_dropout=0.0)
            return self.output(
                x,
                t,
                padding_masks=padding_masks,
                label_emb=label_emb,
                z_a=z_a,
                z_a_dropout=z_a_dropout,
            )

        label_cond = self._encode_cond(cond, cond_dropout=0.0)
        label_uncond = self.cond_encoder.null_label_emb(cond.shape[0], x.device)
        x0_cond = self.output(
            x,
            t,
            padding_masks=padding_masks,
            label_emb=label_cond,
            z_a=z_a,
            z_a_dropout=z_a_dropout,
        )
        x0_uncond = self.output(
            x,
            t,
            padding_masks=padding_masks,
            label_emb=label_uncond,
            z_a=z_a,
            z_a_dropout=z_a_dropout,
        )
        return x0_uncond + scale * (x0_cond - x0_uncond)

    def model_predictions(
        self,
        x,
        t,
        clip_x_start=False,
        padding_masks=None,
        cond=None,
        cfg_scale=1.0,
        z_a=None,
        z_a_dropout=None,
    ):
        from Models.interpretable_diffusion.model_utils import identity

        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        x_start = self.guided_output(
            x,
            t,
            cond=cond,
            cfg_scale=cfg_scale,
            padding_masks=padding_masks,
            z_a=z_a,
            z_a_dropout=z_a_dropout,
        )
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(
        self,
        x,
        t,
        clip_denoised=True,
        cond=None,
        cfg_scale=1.0,
        z_a=None,
        z_a_dropout=None,
    ):
        _, x_start = self.model_predictions(
            x,
            t,
            cond=cond,
            cfg_scale=cfg_scale,
            z_a=z_a,
            z_a_dropout=z_a_dropout,
        )
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(
        self,
        x,
        t,
        clip_denoised=True,
        heavy=False,
        nu=4.0,
        heavy_var_correction=True,
        cond=None,
        cfg_scale=1.0,
        z_a=None,
        z_a_dropout=None,
    ):
        import math

        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            clip_denoised=clip_denoised,
            cond=cond,
            cfg_scale=cfg_scale,
            z_a=z_a,
            z_a_dropout=z_a_dropout,
        )
        if t > 0:
            if heavy:
                noise = self._student_t_noise_like(x, nu, heavy_var_correction)
            else:
                noise = torch.randn_like(x)
        else:
            noise = 0.0
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    @torch.no_grad()
    def sample(self, shape, cond=None, cfg_scale=None, z_a=None, z_a_dropout=0.0):
        from tqdm.auto import tqdm

        device = self.betas.device
        cfg_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        img = torch.randn(shape, device=device)
        total_T = self.num_timesteps

        for t in tqdm(reversed(range(0, total_T)), desc="sampling loop time step", total=total_T):
            heavy = bool(getattr(self, "heavy", False))
            nu = float(getattr(self, "heavy_nu", 8.0))
            hvc = bool(getattr(self, "heavy_var_correction", True))
            img, _ = self.p_sample(
                img,
                t,
                heavy=heavy,
                nu=nu,
                heavy_var_correction=hvc,
                cond=cond,
                cfg_scale=cfg_scale,
                z_a=z_a,
                z_a_dropout=z_a_dropout,
            )
        return img

    def generate_mts(
        self,
        batch_size=16,
        cond=None,
        cfg_scale=None,
        z_a=None,
        z_a_dropout=0.0,
    ):
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        if self.fast_sampling:
            return sample_fn(
                (batch_size, self.seq_length, self.feature_size),
                cond=cond,
                cfg_scale=cfg_scale,
                z_a=z_a,
                z_a_dropout=z_a_dropout,
            )
        return sample_fn(
            (batch_size, self.seq_length, self.feature_size),
            cond=cond,
            cfg_scale=cfg_scale,
            z_a=z_a,
            z_a_dropout=z_a_dropout,
        )

    def _train_loss(
        self,
        x_start,
        t,
        target=None,
        noise=None,
        padding_masks=None,
        cond=None,
        z_a=None,
        z_a_dropout=None,
    ):
        heavy = bool(getattr(self, "heavy", False))
        nu = float(getattr(self, "heavy_nu", 8.0))
        heavy_var_correction = bool(getattr(self, "heavy_var_correction", True))

        if noise is None:
            if heavy:
                noise = self._student_t_noise_like(x_start, nu, heavy_var_correction)
            else:
                noise = torch.randn_like(x_start)

        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        label_emb = self._encode_cond(cond)
        model_out = self.output(
            x,
            t,
            padding_masks,
            label_emb=label_emb,
            z_a=z_a,
            z_a_dropout=z_a_dropout,
        )
        train_loss = self.loss_fn(model_out, target, reduction="none")

        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(target.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = (
                self.loss_fn(torch.real(fft1), torch.real(fft2), reduction="none")
                + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction="none")
            )
            train_loss = train_loss + self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, "b ... -> b (...)", "mean")
        b = x_start.shape[0]
        train_loss = train_loss.view(b, -1).mean(dim=1)

        if getattr(self, "var_loss_weight", 0.0) > 0.0:
            x_hat_flat = model_out.view(b, -1)
            x_gt_flat = x_start.view(b, -1)
            std_hat = x_hat_flat.std(dim=1, unbiased=False)
            std_gt = x_gt_flat.std(dim=1, unbiased=False)
            l_var = (std_hat - std_gt).abs()
            train_loss = train_loss + self.var_loss_weight * l_var

        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean(), model_out

    def forward(self, x, cond=None, z_a=None, return_x0=False, **kwargs):
        b, c, n, device, feature_size = *x.shape, x.device, self.feature_size
        assert n == feature_size, f"number of variable must be {feature_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        loss, x0_pred = self._train_loss(
            x_start=x,
            t=t,
            cond=cond,
            z_a=z_a,
            **kwargs,
        )
        if return_x0:
            return loss, x0_pred
        return loss

    def load_backbone_weights(self, state_dict, strict: bool = False) -> None:
        """Load DiffusionTS backbone weights; sync mirrored control encoder."""
        backbone_sd = {
            k: v for k, v in state_dict.items() if not k.startswith("controlnet.")
        }
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        if strict:
            extra = [k for k in missing if not k.startswith("controlnet.")]
            if extra:
                raise RuntimeError(f"Missing backbone keys: {extra[:5]}...")
        del missing, unexpected
        self.controlnet.mirror_from_backbone(self.model.encoder)

    def controlnet_state_dict(self):
        return self.controlnet.state_dict()

    def load_controlnet_weights(self, state_dict) -> None:
        missing, unexpected = self.controlnet.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                f"ControlNet load: missing keys (new layers init from scratch): {missing}"
            )
        if unexpected:
            print(f"ControlNet load: unexpected keys ignored: {unexpected}")

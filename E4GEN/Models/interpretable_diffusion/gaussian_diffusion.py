import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract
from Models.interpretable_diffusion.condition_encoder import ConditionEncoder


# gaussian diffusion class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DiffusionTS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps = 50,
            fast_sampling=False,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            mode = True,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,

            heavy=False,
            heavy_nu=8.0,
            heavy_var_correction=True,

            use_condition=False,
            cond_dim=1,
            num_classes_per_cond=None,
            cond_dropout=0.15,
            cfg_scale=1.5,

            # sanity check
            sanity_prob=0.002,
            sanity_quantile=0.999,
            sanity_compare_ref=True,
            sanity_check_xt=True,

            **kwargs
    ):
        super().__init__()

        self.heavy = bool(heavy)
        self.heavy_nu = float(heavy_nu)
        self.heavy_var_correction = bool(heavy_var_correction)
        self.sanity_prob = float(sanity_prob)
        self.sanity_quantile = float(sanity_quantile)
        self.sanity_compare_ref = bool(sanity_compare_ref)
        self.sanity_check_xt = bool(sanity_check_xt)

        self.use_condition = bool(use_condition)
        self.cond_dim = int(cond_dim)
        self.cond_dropout = float(cond_dropout)
        self.cfg_scale = float(cfg_scale)
        if num_classes_per_cond is None:
            num_classes_per_cond = []
        self.num_classes_per_cond = [int(n) for n in num_classes_per_cond]

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.fast_sampling = fast_sampling
        self.mode = mode
        self._trace = None
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)
        self.var_loss_weight = float(kwargs.get("var_loss_weight", 0.0))
        _d_model = d_model if d_model is not None else 64

        if self.use_condition:
            if len(self.num_classes_per_cond) != self.cond_dim:
                raise ValueError(
                    f"num_classes_per_cond length {len(self.num_classes_per_cond)} "
                    f"must match cond_dim {self.cond_dim}"
                )
            self.cond_encoder = ConditionEncoder(self.num_classes_per_cond, _d_model)
        else:
            self.cond_encoder = None

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        if self.fast_sampling:
            self.sampling_timesteps = sampling_timesteps
        else:
            self.sampling_timesteps = timesteps

        assert self.sampling_timesteps <= timesteps

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def set_trace(self, trace):
        self._trace = trace

    def _encode_cond(self, cond, cond_dropout=None):
        if not self.use_condition or cond is None or self.cond_encoder is None:
            return None
        dropout = self.cond_dropout if cond_dropout is None else float(cond_dropout)
        return self.cond_encoder(cond, cond_dropout=dropout)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, t, padding_masks=None, label_emb=None):
        trend, season, trend_cum, season_cum = self.model(
            x, t, padding_masks=padding_masks, label_emb=label_emb,
        )
        model_output = trend + season

        return model_output

    def guided_output(self, x, t, cond=None, cfg_scale=1.0, padding_masks=None):
        if not self.use_condition or cond is None:
            return self.output(x, t, padding_masks=padding_masks, label_emb=None)

        scale = float(cfg_scale)
        if scale == 1.0:
            label_emb = self._encode_cond(cond, cond_dropout=0.0)
            return self.output(x, t, padding_masks=padding_masks, label_emb=label_emb)

        label_cond = self._encode_cond(cond, cond_dropout=0.0)
        label_uncond = self.cond_encoder.null_label_emb(cond.shape[0], x.device)
        x0_cond = self.output(x, t, padding_masks=padding_masks, label_emb=label_cond)
        x0_uncond = self.output(x, t, padding_masks=padding_masks, label_emb=label_uncond)
        return x0_uncond + scale * (x0_cond - x0_uncond)

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None, cond=None, cfg_scale=1.0):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.guided_output(x, t, cond=cond, cfg_scale=cfg_scale, padding_masks=padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True, cond=None, cfg_scale=1.0):
        _, x_start = self.model_predictions(x, t, cond=cond, cfg_scale=cfg_scale)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def _student_t_noise_like(self, x, nu, heavy_var_correction):
        if nu <= 2.0:
            raise ValueError(f"heavy_nu must be > 2, got {nu}")
        z = torch.randn_like(x)
        B = x.shape[0]
        kappa = torch.distributions.Chi2(df=nu).sample((B,)).to(
            device=x.device, dtype=x.dtype
        )
        scale = torch.sqrt(kappa / nu).view((B,) + (1,) * (x.ndim - 1))
        noise = z / scale
        if heavy_var_correction:
            noise = noise * math.sqrt((nu - 2.0) / nu)
        return noise

    def p_sample(self, x, t, clip_denoised=True, heavy=False, nu=4.0, heavy_var_correction=True,
                 cond=None, cfg_scale=1.0):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(
                x=x, t=batched_times, clip_denoised=clip_denoised,
                cond=cond, cfg_scale=cfg_scale,
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
    def sample(self, shape, cond=None, cfg_scale=None):
        device = self.betas.device
        cfg_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)

        img = torch.randn(shape, device=device)
        total_T = self.num_timesteps

        for t in tqdm(reversed(range(0, total_T)),
                      desc='sampling loop time step', total=total_T):
            heavy = bool(getattr(self, "heavy", False))
            nu = float(getattr(self, "heavy_nu", 8.0))
            hvc = bool(getattr(self, "heavy_var_correction", True))
            img, x_start = self.p_sample(
                img, t,
                heavy=heavy, nu=nu, heavy_var_correction=hvc,
                cond=cond, cfg_scale=cfg_scale,
            )

            if (
                getattr(self, "_trace", None) is not None
                and (not self.mode)
                and (t % 10 == 0)
            ):
                self._trace.log(t, img, x_start)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True, cond=None, cfg_scale=None):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta
        cfg_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            pred_noise, x_start = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised,
                cond=cond, cfg_scale=cfg_scale,
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c_coef = (1 - alpha_next - sigma ** 2).sqrt()
            if bool(getattr(self, "heavy", False)):
                nu = float(getattr(self, "heavy_nu", 8.0))
                hvc = bool(getattr(self, "heavy_var_correction", True))
                noise = self._student_t_noise_like(img, nu, hvc)
            else:
                noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c_coef * pred_noise + \
                  sigma * noise

        return img

    def generate_mts(self, batch_size=16, cond=None, cfg_scale=None):
        feature_size, seq_length = self.feature_size, self.seq_length
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn(
            (batch_size, seq_length, feature_size),
            cond=cond,
            cfg_scale=cfg_scale,
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None, cond=None):
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
        model_out = self.output(x, t, padding_masks, label_emb=label_emb)
        train_loss = self.loss_fn(model_out, target, reduction='none')

        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)

            fourier_loss = (
                    self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')
                    + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            )
            train_loss = train_loss + self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        B = x_start.shape[0]
        train_loss = train_loss.view(B, -1).mean(dim=1)

        if getattr(self, "var_loss_weight", 0.0) > 0.0:
            x_hat_flat = model_out.view(B, -1)
            x_gt_flat = x_start.view(B, -1)
            std_hat = x_hat_flat.std(dim=1, unbiased=False)
            std_gt = x_gt_flat.std(dim=1, unbiased=False)
            l_var = (std_hat - std_gt).abs()
            train_loss = train_loss + self.var_loss_weight * l_var

        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)

        return train_loss.mean()

    def forward(self, x, cond=None, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, cond=cond, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x


if __name__ == '__main__':
    pass

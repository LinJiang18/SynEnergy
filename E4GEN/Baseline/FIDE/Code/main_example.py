"""
FIDE-style training on a pre-windowed numpy dataset (e.g. toyexample_seq_200_sample_1000.npy).

Pipeline mirrors main_temperature_example.py except:
- No CSV / monthly climatology; global StandardScaler on all values replaces process_data windows.
- Outputs only two arrays in the same *naming style* as the temperature script:
    {OUTPUT_STEM}_ori_sequences_degC.npy   # original scale (same as input), not literal Celsius
    {OUTPUT_STEM}_ddpm_sequences_degC.npy  # generated, mapped back with inverse scaler
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.fft import irfft, rfft, rfftfreq
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config (swap dataset by changing these)
# -----------------------------------------------------------------------------
DATA_FILENAME = "precipitation_seq_90_sample_1056.npy"
OUTPUT_STEM = "precipitation"  # -> precipitation_ori_sequences_degC.npy, etc.
RESULTS_SUBDIR = "FIDE"
    
is_fit_AR = False
is_regularizer = True
freq_enhance_c = 1.1
percentage_of_freq_enhanced = 20
batch_size = 2000
n_epochs = 400

_FIDE_ROOT = Path(__file__).resolve().parent.parent
_BASELINE_ROOT = _FIDE_ROOT.parent
RESULTS_DIR = _BASELINE_ROOT / "Results" / RESULTS_SUBDIR
DATA_PATH = _FIDE_ROOT / "Data" / DATA_FILENAME


def _load_windowed_npy(path: Path) -> np.ndarray:
    """
    Return float32 (N, L, 1).

    If the basename matches ``seq_<L>_sample_<N>``:
    - Flattened ``(N*L,)``, ``(N*L, 1)``, or ``(N*L, 1, 1)`` -> reshape ``(N, L, 1)`` (C-order, each sample's L steps contiguous).
    - Already ``(N, L, 1)`` or ``(N, L)`` -> add channel if needed.
    - ``(L, N, 1)`` -> transpose to ``(N, L, 1)``.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Data not found: {path}\nPlace {path.name} under {_FIDE_ROOT / 'Data'}."
        )
    x = np.load(path)
    x = np.asarray(x, dtype=np.float32)

    m = re.search(r"seq_(\d+)_sample_(\d+)", path.name)
    if m:
        seq_exp, n_exp = int(m.group(1)), int(m.group(2))
        total = n_exp * seq_exp
        if x.ndim == 1:
            if x.size != total:
                raise ValueError(
                    f"{path.name}: expected {total} elements for seq_{seq_exp}_sample_{n_exp}, got {x.size}"
                )
            x = x.reshape(n_exp, seq_exp, 1)
        elif x.ndim == 2:
            if x.shape == (total, 1):
                x = x.reshape(n_exp, seq_exp, 1)
            elif x.shape == (n_exp, seq_exp):
                x = x[..., np.newaxis]
            else:
                raise ValueError(
                    f"{path.name}: 2D shape {x.shape} not (N*L,1)=({total},1) nor (N,L)=({n_exp},{seq_exp})"
                )
        elif x.ndim == 3:
            if x.shape == (total, 1, 1):
                x = x.reshape(n_exp, seq_exp, 1)
            elif x.shape == (seq_exp, n_exp, 1):
                x = np.transpose(x, (1, 0, 2))
            elif x.shape == (n_exp, seq_exp, 1):
                pass
            else:
                raise ValueError(
                    f"{path.name}: 3D shape {x.shape} not (N,L,1)=({n_exp},{seq_exp},1), "
                    f"(L,N,1)=({seq_exp},{n_exp},1), nor flat ({total},1,1)"
                )
        else:
            raise ValueError(f"{path.name}: expected 1D–3D array when name matches seq_*_sample_*, got {x.ndim}D")
        return x

    # No seq/sample hint in filename: require explicit (N, L) or (N, L, 1)
    if x.ndim == 2:
        x = x[..., np.newaxis]
    if x.ndim != 3:
        raise ValueError(
            f"Expected (N,L) or (N,L,1), or use a filename like *_seq_200_sample_1000.npy for flat storage; got {x.shape}"
        )
    return x


def _set_seq_len_env(length: int) -> None:
    os.environ["FIDE_SEQ_LEN"] = str(int(length))
    # If model was already imported in this process, force reload
    for name in ("model", "train_utilities", "sampling"):
        if name in sys.modules:
            del sys.modules[name]


def main() -> None:
    ori_native = _load_windowed_npy(DATA_PATH)
    num_samples, seq_len, dim = ori_native.shape
    if dim != 1:
        raise ValueError(f"Only univariate (C=1) supported for now, got C={dim}")

    _set_seq_len_env(seq_len)

    # Late imports so model picks up FIDE_SEQ_LEN
    from data_process import fit_AR_model
    from general_utilities import (
        CMD,
        CRPS,
        KL_JS_divergence,
        KS_Test,
        fitting_gev_and_sampling,
    )
    from model import TransformerModel
    import train_utilities as train_ut
    from sampling import ddpm_sample
    from temperature_inverse import invert_fft_enhancement

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler().fit(ori_native.reshape(-1, 1))
    x_scaled = scaler.transform(ori_native.reshape(-1, 1)).reshape(num_samples, seq_len, 1)
    real_data = torch.from_numpy(x_scaled.astype(np.float32)).to(device)

    t = (
        torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
        .view(1, seq_len, 1)
        .expand(num_samples, seq_len, 1)
    )
    print(f"Loaded {DATA_PATH.name}: ori_native {ori_native.shape}, scaled tensor {real_data.shape}")

    if is_fit_AR:
        fit_AR_model(real_data, true_coeffs=[0.5], order=1, seq_len=seq_len)

    block_maxima_real_data_value, _ = torch.max(real_data, dim=1)
    block_maxima_real_data_value = block_maxima_real_data_value.reshape(-1, 1, 1)
    block_maxima_real_data = block_maxima_real_data_value

    bm_flat = block_maxima_real_data_value.detach().cpu().numpy().reshape(-1)
    bm_samples_gev, gev_model = fitting_gev_and_sampling(
        bm_flat, num_samples, plot=False
    )
    KS_Test(bm_flat, bm_samples_gev)
    CMD(bm_flat, bm_samples_gev)
    KL_JS_divergence(bm_flat, bm_samples_gev)
    CRPS(bm_flat, bm_samples_gev)

    rd = real_data.detach().cpu().numpy()
    rd_fft = rfft(rd, axis=1)
    rd_freq = rfftfreq(seq_len)
    top_n = max(1, int((rd_fft.shape[1] * percentage_of_freq_enhanced) / 100))
    top_indices = np.argsort(rd_freq)[-top_n:]
    boosted = rd_fft.copy()
    for i in range(boosted.shape[0]):
        boosted[i, :, 0][top_indices] *= freq_enhance_c
    rd_enh = irfft(boosted.reshape(-1, rd_freq.shape[0]), n=seq_len, axis=1)
    real_data = torch.from_numpy(rd_enh.reshape(num_samples, seq_len, 1).astype(np.float32)).to(device)

    eff_bs = min(batch_size, num_samples)
    if eff_bs < 2 and num_samples >= 2:
        eff_bs = num_samples
    drop_last = num_samples >= eff_bs and (num_samples % eff_bs != 0)

    model = TransformerModel(dim=1, hidden_dim=64, max_i=train_ut.diffusion_steps).to(device)
    train_ut.model = model
    train_ut.is_regularizer = is_regularizer
    train_ut.gev_model = gev_model
    optim = torch.optim.Adam(model.parameters())

    training_loss_history: list[float] = []
    loader = DataLoader(
        TensorDataset(real_data, t, block_maxima_real_data),
        batch_size=eff_bs,
        shuffle=False,
        drop_last=drop_last,
    )
    for epoch in tqdm(range(n_epochs)):
        for input_b, time_b, bm_b in loader:
            optim.zero_grad()
            loss, ddpm_loss, reg_loss = train_ut.get_loss(input_b, time_b, bm_b)
            loss.backward()
            optim.step()
        if epoch < 25 or epoch % 20 == 0:
            print(
                f"Iteration: {epoch} --- Loss: {loss.cpu().item():.6f}, "
                f"DDPM: {ddpm_loss.cpu().item():.6f}, Reg: {reg_loss.cpu().item():.6f}"
            )
        training_loss_history.append(float(loss.detach().cpu().item()))

    if training_loss_history:
        print(
            f"Training loss (last): {training_loss_history[-1]:.6f}; "
            f"min: {min(training_loss_history):.6f}"
        )

    bm_cond = torch.tensor(bm_samples_gev, dtype=torch.float32, device=device).reshape(-1, 1, 1)
    t_sample = (
        torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)
        .view(1, seq_len, 1)
        .expand(num_samples, seq_len, 1)
    )
    samples_ddpm = ddpm_sample(t_sample, bm_cond)
    seq_np = samples_ddpm.detach().cpu().numpy().astype(np.float32)

    mean = float(scaler.mean_[0])
    scale = float(scaler.scale_[0]) if scaler.scale_[0] != 0 else 1.0
    seq_2d = invert_fft_enhancement(
        seq_np.reshape(num_samples, seq_len),
        seq_len,
        freq_enhance_c,
        percentage_of_freq_enhanced,
    )
    seq_native = (seq_2d * scale + mean).astype(np.float32)[..., np.newaxis]

    ori_out = np.asarray(ori_native, dtype=np.float32)
    if ori_out.ndim == 2:
        ori_out = ori_out[..., np.newaxis]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ori_name = f"{OUTPUT_STEM}_ori_sequences_degC.npy"
    gen_name = f"{OUTPUT_STEM}_ddpm_sequences_degC.npy"
    np.save(RESULTS_DIR / ori_name, ori_out)
    np.save(RESULTS_DIR / gen_name, seq_native.astype(np.float32))
    print(f"Saved {ori_name}, {gen_name} -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()

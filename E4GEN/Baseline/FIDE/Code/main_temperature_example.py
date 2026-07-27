import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from scipy.fft import rfft, rfftfreq, irfft

from torch.utils.data import DataLoader, TensorDataset

from data_process import process_data, fit_AR_model
from general_utilities import (
    CMD,
    CRPS,
    KL_JS_divergence,
    KS_Test,
    fitting_gev_and_sampling,
)
from model import TransformerModel
from sampling import ddpm_sample
import train_utilities as train_ut
from temperature_inverse import model_space_to_celsius

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#setting parameters
time_series_seq_len = 30
seq_len = time_series_seq_len
n_seq = 1
is_fit_AR = False
# FIDE/Data next to this file's Code/ directory (see README)
_FIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASELINE_ROOT = os.path.dirname(_FIDE_ROOT)
RESULTS_DIR = os.path.join(_BASELINE_ROOT, "Results", "FIDE")
data_folder_path = os.path.join(_FIDE_ROOT, "Data")
data_path = os.path.join(data_folder_path, "temperature_raw.csv")
#processing data
data = pd.read_csv(data_path, parse_dates=['Date'], on_bad_lines='skip')
data.columns = ['date', 'temp']
no_digit = data[~data.temp.str[0].str.isdigit()]
data = data[~data.index.isin(no_digit.index)]
data['temp'] = data['temp'].astype(float)

data.set_index('date', inplace=True)

# Extract year and month from the date
data['year'] = data.index.year
data['month'] = data.index.month

# Calculate monthly mean and standard deviation
monthly_stats = data.groupby(['year', 'month'])['temp'].agg(['mean', 'std']).reset_index()

# Merge the monthly statistics back to the original data
data = pd.merge(data, monthly_stats, on=['year', 'month'], how='left', suffixes=('', '_monthly'))

# Standardize the temperature based on monthly mean and standard deviation
data["temp_standardized"] = (data["temp"] - data["mean"]) / data["std"]

years_full = data["year"].to_numpy(dtype=np.int32)
months_full = data["month"].to_numpy(dtype=np.int32)
n_windows = len(years_full) - time_series_seq_len
window_cal = np.zeros((n_windows, time_series_seq_len, 2), dtype=np.int32)
for i in range(n_windows):
    window_cal[i, :, 0] = years_full[i : i + time_series_seq_len]
    window_cal[i, :, 1] = months_full[i : i + time_series_seq_len]

# Drop unnecessary columns
data.drop(["year", "month", "mean", "std"], axis=1, inplace=True)

# Real °C windows (N, seq_len, 1) — aligned row i with training window i / generated sample i
_temp_series_degC = data["temp"].to_numpy(dtype=np.float32)
ori_degC = np.stack(
    [_temp_series_degC[i : i + seq_len] for i in range(n_windows)],
    axis=0,
)[..., np.newaxis]

real_data = np.array(data["temp_standardized"]).reshape(-1, 1)
print(f"Real data shape: {real_data.shape}")

real_data, series_scaler = process_data(real_data, time_series_seq_len, return_scaler=True)
real_data = real_data.to(device)
t = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, seq_len, 1).expand(real_data.shape[0], seq_len, 1).to(device)
real_data = real_data.to(dtype=torch.float32)
print(f"Real Data: Mean: {torch.mean(real_data)} Std:{torch.std(real_data)}")
if is_fit_AR:
    fit_AR_model(real_data, true_coeffs=[0.5], order=1, seq_len=seq_len)

block_maxima_real_data_value, block_maxima_real_data_pos = torch.max(real_data, dim=1)
block_maxima_real_data_value = block_maxima_real_data_value.reshape(-1,1,1)
block_maxima_real_data_pos = block_maxima_real_data_pos.reshape(-1,1,1)
block_maxima_real_data = block_maxima_real_data_value #torch.cat((block_maxima_real_data_value, block_maxima_real_data_pos), dim=1)

print(f"Shape of real data, time steps (t), block maxima after processing: {real_data.shape,  t.shape, block_maxima_real_data.shape}")
num_samples = block_maxima_real_data.shape[0]
print(f"Number of samples: {num_samples}")

#Fitting GEV and Metrics

block_maxima_real_data_value = block_maxima_real_data_value.cpu().numpy().reshape(-1)
block_maxima_real_data_pos = block_maxima_real_data_pos.cpu().numpy().reshape(-1)
bm_samples_gev, gev_model = fitting_gev_and_sampling(
    block_maxima_real_data_value, num_samples, plot=False
)
KS_Test(block_maxima_real_data_value, bm_samples_gev)
CMD(block_maxima_real_data_value, bm_samples_gev)
KL_JS_divergence(block_maxima_real_data_value, bm_samples_gev)
CRPS(block_maxima_real_data_value, bm_samples_gev)

# frequency enhancement

if not isinstance(real_data, np.ndarray):
    real_data = real_data.cpu().numpy()
real_data_fft = rfft(real_data, axis=1)
real_data_freq = rfftfreq(real_data.shape[1])
print(real_data_freq.shape)

freq_enhance_c = 1.1
percentage_of_freq_enhanced = 20
top_freq_enhanced = int((real_data_fft.shape[1] * percentage_of_freq_enhanced) / 100)

high_freq_enhanced_fft_result = real_data_fft.copy()
top_indices = np.argsort(real_data_freq)[-top_freq_enhanced:]

# Iterate over all datapoints along the second dimension
for i in range(real_data_fft.shape[0]):
    high_freq_enhanced_fft_result[i, :, 0][top_indices] *= freq_enhance_c

# print(real_data_fft[0], high_freq_enhanced_fft_result[0], real_data_fft[-1], high_freq_enhanced_fft_result[-1])

print(high_freq_enhanced_fft_result.shape)
real_data = irfft(high_freq_enhanced_fft_result.reshape(-1, real_data_freq.shape[0]))
real_data  =  torch.from_numpy(real_data.reshape(real_data.shape[0], real_data.shape[1], 1)).to(device)

#training

is_regularizer = True

batch_size = 2000
n_epochs = 400

#model initialization

model = TransformerModel(dim=1, hidden_dim=64, max_i=train_ut.diffusion_steps).to(device)
train_ut.model = model
train_ut.is_regularizer = is_regularizer
train_ut.gev_model = gev_model
optim = torch.optim.Adam(model.parameters())

training_loss_history = np.array([])

train_loader = DataLoader(TensorDataset(real_data, t, block_maxima_real_data), batch_size=batch_size, shuffle=False, drop_last=True)
for epoch in tqdm(range(n_epochs)):
    for i, (input, time, bm) in enumerate(train_loader):
      optim.zero_grad()
      loss, ddpm_loss, reg_loss = train_ut.get_loss(input, time, bm)
      loss.backward()
      optim.step()
    if epoch<25 or epoch%20==0: print(f"Iteration: {epoch} --- Loss: {loss.cpu().item()}, DDPM Loss: {ddpm_loss.cpu().item()}, Regularizer Loss: {reg_loss.cpu().item()}")
    # if epoch<25 or epoch%20==0: print(f"Iteration: {epoch} --- Loss: {loss.cpu().item()}")
    training_loss_history = np.append(training_loss_history, loss.item())
    # break

if len(training_loss_history) > 0:
    print(
        f"Training loss (last): {training_loss_history[-1]:.6f}; "
        f"min: {np.min(training_loss_history):.6f}"
    )

# --- Final DDPM samples (standardized monthly space, same as training) ---
os.makedirs(RESULTS_DIR, exist_ok=True)
bm_cond = torch.tensor(bm_samples_gev, dtype=torch.float32, device=device).reshape(-1, 1, 1)
t_sample = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device).view(1, seq_len, 1).expand(
    num_samples, seq_len, 1
)
samples_ddpm = ddpm_sample(t_sample, bm_cond)
# (N, 30, 1) — same rank as training input x for direct comparison
seq_np = samples_ddpm.detach().cpu().numpy().astype(np.float32)

if seq_np.shape[0] != window_cal.shape[0]:
    raise RuntimeError("window_calendar rows must match number of generated sequences")

seq_degC = model_space_to_celsius(
    seq_np,
    seq_len=seq_len,
    c=freq_enhance_c,
    percentage_of_freq_enhanced=percentage_of_freq_enhanced,
    global_mean=float(series_scaler.mean_[0]),
    global_scale=float(series_scaler.scale_[0]),
    window_cal=window_cal,
    monthly_stats=monthly_stats,
)

# Baseline reproduction: save one ground-truth window npy and one generated window npy (°C only)
np.save(os.path.join(RESULTS_DIR, "temperature_ori_sequences_degC.npy"), ori_degC)
np.save(os.path.join(RESULTS_DIR, "temperature_ddpm_sequences_degC.npy"), seq_degC)
print(
    f"Saved (degC only): temperature_ori_sequences_degC.npy, "
    f"temperature_ddpm_sequences_degC.npy -> {RESULTS_DIR}"
)

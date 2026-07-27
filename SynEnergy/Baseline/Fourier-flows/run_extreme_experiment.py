# coding: utf-8
# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#
# Multivariate (N, T, C) experiment driver: internally uses flat dim T*C for flows;
# sampled output is reshaped to (n_samples, T, C).

from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import scipy.stats as st

import warnings

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _data_file(filename):
    return os.path.normpath(os.path.join(_SCRIPT_DIR, "data", filename))

from SequentialFlows import FourierFlow, RealNVP

from metrics.PRcurve import computeF1
from metrics.MAE import computeMAE


def mean_confidence_interval(data, confidence=0.95):

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2.0, n - 1)

    return m, h


def normalize_ntc(X):
    """Per-channel min–max to [0, 1] over all N and T."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError("Expected X with shape (N, T, C), got %s" % (X.shape,))
    N, T, C = X.shape
    out = np.empty_like(X)
    for c in range(C):
        v = X[:, :, c]
        mn = np.min(v)
        mx = np.max(v)
        out[:, :, c] = (v - mn) / (mx - mn + 1e-7)
    return out


def shuffle_ntc(X, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    idx = rng.permutation(X.shape[0])
    return X[idx]


def _flow_time_dim(flat_len):
    """
    FourierFlow + DFT assume an odd N_fft so that spectral size 2*(floor(N_fft/2)+1) equals N_fft+1.
    For even flat_len (e.g. T*C=250), pad time domain to flat_len+1 (odd).
    """
    if flat_len % 2 == 1:
        return flat_len
    return flat_len + 1


def ntc_to_flow_flat_list(X_ntc):
    """
    (N, T, C) -> (list of N vectors of length flow_dim, flat_len T*C, flow_dim).
    If flow_dim > flat_len, prepends zeros on the time-domain vector (same idea as run_experiment_2 hstack(0,...)).
    """
    N, T, C = X_ntc.shape
    flat_len = T * C
    flow_dim = _flow_time_dim(flat_len)
    pad = flow_dim - flat_len
    out = []
    for i in range(N):
        v = X_ntc[i].reshape(-1).astype(np.float64, copy=False)
        if pad:
            v = np.concatenate((np.zeros(pad, dtype=np.float64), v))
        out.append(v)
    return out, flat_len, flow_dim


def flat_samples_to_ntc(samples_flat, T, C, flat_len, flow_dim):
    """Strip leading padding if any, then (n_samples, T * C) -> (n_samples, T, C)."""
    arr = np.asarray(samples_flat)
    if arr.ndim != 2 or arr.shape[1] != flow_dim:
        raise ValueError(
            "Expected sample array (n, flow_dim) with flow_dim=%d, got %s" % (flow_dim, arr.shape)
        )
    trim = flow_dim - flat_len
    if trim:
        arr = arr[:, trim:]
    return arr.reshape(arr.shape[0], T, C)


def computeMAE_ntc(X_real_ntc, X_synth_ntc):
    """Train/eval predictive metric per channel; return mean MAE over C (same logic as metrics/MAE)."""
    N, T, C = X_real_ntc.shape
    if X_synth_ntc.shape[1:] != (T, C):
        raise ValueError("Synthetic shape %s incompatible with T=%d, C=%d" % (X_synth_ntc.shape, T, C))
    scores = []
    for c in range(C):
        Xr = [X_real_ntc[i, :, c] for i in range(N)]
        Xs = [X_synth_ntc[i, :, c] for i in range(X_synth_ntc.shape[0])]
        scores.append(computeMAE(Xr, Xs))
    return float(np.mean(scores))


# Match run_experiment_2 "energy"/"lung" Fourier depth (n_flows=5); hidden=200 unchanged.
FF_model_params = {"hidden": 200, "n_flows": 5, "normalize": True}
FF_train_params = {
    "epochs": 1000,
    "batch_size": 500,
    "learning_rate": 1e-3,
    "display_step": 100,
}
RNVP_model_params = {"hidden": 200, "n_flows": 3}
RNVP_train_params = {
    "epochs": 500,
    "batch_size": 500,
    "learning_rate": 1e-4,
    "display_step": 100,
}


def run_extreme_experiments(
    X_ntc, baselines, num_experiments=5, n_samples=10000, shuffle_seed=None
):
    """
    Args:
        X_ntc: numpy array (N, T, C), will be min–max normalized per channel and sample-shuffled.
        baselines: e.g. ["Fourier flow", "RealNVP"]
    Returns:
        F1_scores, MAE_scores (per baseline, list over experiment repeats),
        last_samples_ntc: (n_samples, T, C) from last baseline / last repeat (for quick inspection)
    """
    X_ntc = normalize_ntc(X_ntc)
    X_ntc = shuffle_ntc(X_ntc, seed=shuffle_seed)
    N, T, C = X_ntc.shape
    X_flat, flat_len, flow_dim = ntc_to_flow_flat_list(X_ntc)
    assert flat_len == T * C

    F1_scores = dict.fromkeys(baselines)
    MAE_scores = dict.fromkeys(baselines)
    for b in baselines:
        F1_scores[b] = []
        MAE_scores[b] = []

    last_samples_ntc = None

    for baseline in baselines:

        print("Baseline: ", baseline)
        print("-----------------------")

        for k in range(num_experiments):

            print("Experiment number: ", k)

            if baseline == "Fourier flow":
                model = FourierFlow(**FF_model_params, fft_size=flow_dim)
                train_kw = FF_train_params
            elif baseline == "RealNVP":
                model = RealNVP(**RNVP_model_params, T=flow_dim)
                train_kw = RNVP_train_params
            else:
                raise ValueError("Baseline %s not implemented." % baseline)

            _ = model.fit(X_flat, **train_kw)

            samples_flat = model.sample(n_samples)
            samples_ntc = flat_samples_to_ntc(samples_flat, T, C, flat_len, flow_dim)
            last_samples_ntc = samples_ntc

            f1 = computeF1(X_flat, [samples_flat[i] for i in range(samples_flat.shape[0])])
            mae = computeMAE_ntc(X_ntc, samples_ntc)

            F1_scores[baseline].append(f1)
            MAE_scores[baseline].append(mae)

            print("F1 score", F1_scores[baseline][-1])
            print("MAE score", MAE_scores[baseline][-1])

    return F1_scores, MAE_scores, last_samples_ntc


def _ensure_ntc(X):
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, :, np.newaxis]
    if X.ndim != 3:
        raise ValueError("Expected (N, T) or (N, T, C), got shape %s" % (X.shape,))
    return X.astype(np.float64)


if __name__ == "__main__":
    # --- Edit defaults here (under Baseline/Fourier-flows/data/). Optional batch override:
    #     FFLOWS_PROJECT_NAME / FFLOWS_DATA_FILE (see run_all_extreme_datasets.sh).
    project_name = "windspeed"
    data_file = "windspeed_seq_90_sample_1056.npy"

    input_data = np.load(_data_file(data_file))

    baselines = ["Fourier flow"]
    num_experiments = 1
    shuffle_seed = 0

    X_ntc = _ensure_ntc(input_data)
    N, T, C = X_ntc.shape
    n_samples = N

    results_dir = os.path.join(_SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_prefix = os.path.join(results_dir, "%s_fourier_flows_extreme" % project_name)
    samples_path = out_prefix + "_synthetic_samples.npy"

    print("Project: %s" % project_name)
    print("Input shape (N, T, C): (%d, %d, %d)" % (N, T, C))
    print("-----------------------")

    F1_scores, MAE_scores, samples_ntc = run_extreme_experiments(
        X_ntc,
        baselines,
        num_experiments=num_experiments,
        n_samples=n_samples,
        shuffle_seed=shuffle_seed,
    )

    print("Generated samples shape (last run): ", samples_ntc.shape)

    for baseline in baselines:
        f1_m, f1_h = mean_confidence_interval(F1_scores[baseline])
        mae_m, mae_h = mean_confidence_interval(MAE_scores[baseline])
        print("Results for ", baseline)
        print("  F1:  mean=%.6f  +/- %.6f (95%% CI half-width)" % (f1_m, f1_h))
        print("  MAE: mean=%.6f  +/- %.6f (95%% CI half-width)" % (mae_m, mae_h))

    np.save(samples_path, samples_ntc.astype(np.float32))

    print("Wrote %s" % samples_path)

from __future__ import annotations

import matplotlib.pyplot as plt
from scipy.stats import genextreme
from scipy.stats import kstest
import numpy as np
from scipy import stats
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.spatial import distance
from scipy.stats import entropy
from properscoring import crps_ensemble


def fitting_gev_and_sampling(ymax, num_samples, plot: bool = True):
    # Fit the data to a GEV distribution
    shape, loc, scale = genextreme.fit(ymax)
    # Generate a GEV distribution using the fitted parameters
    gev_distribution = genextreme(shape, loc=loc, scale=scale)
    # Perform a Kolmogorov-Smirnov test for goodness of fit
    ks_statistic, p_value = kstest(ymax, cdf="genextreme", args=(shape, loc, scale))
    print(f"Kolmogorov–Smirnov test: K-S Statistic: {ks_statistic}; p-value: {p_value}")

    gev_samples = gev_distribution.rvs(size=num_samples)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].hist(ymax, bins=30, density=True, alpha=0.5, color="b", label="Histogram")
        sns.kdeplot(ymax, color="b", label="Smoothed (KDE)", ax=axs[0])
        x = np.linspace(min(ymax), max(ymax), 100)
        axs[0].plot(x, gev_distribution.pdf(x), "r-", lw=2, label="GEV PDF")
        axs[0].legend()
        axs[0].set_title("Training/Fitted Data")

        axs[1].hist(gev_samples, bins=30, density=True, alpha=0.5, color="b", label="Histogram")
        sns.kdeplot(gev_samples, color="b", label="Smoothed (KDE)", ax=axs[1])
        axs[1].plot(x, gev_distribution.pdf(x), "r-", lw=2, label="GEV PDF")
        axs[1].legend()
        axs[1].set_title("Sampled Data")
        plt.tight_layout()
        plt.show()
    return gev_samples, gev_distribution


def plot_kde(real, generated, x_axis_label="Max Value", title="KDE Density Plot of Max Values (Real vs GEV Fitted)"):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    kde1 = sns.kdeplot(real, color='red', label="Real", fill=True)
    kde2 = sns.kdeplot(generated, color='green', label="Generated")

    plt.xlabel(x_axis_label)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.show()


def KS_Test(real, generated):
    ks_statistic, p_value = kstest(real, generated)
    print(f"Kolmogorov–Smirnov test: K-S Statistic: {ks_statistic}; p-value: {p_value}")


# def wasserstein_distance(real, generated):
#   print(f"Wassertein Distance: {wasserstein_distance(real, generated)}")

def CMD(real, generated):
    print(f"Cramer Von Mises Distance: {stats.cramervonmises_2samp(real, generated)}")


def KL_JS_divergence(p_samples, q_samples, num_bins: int | None = None):
    if num_bins is None:
        num_bins = min(100, max(10, len(p_samples)))
    lo = float(np.min(np.minimum(p_samples, q_samples)))
    hi = float(np.max(np.maximum(p_samples, q_samples)))
    if hi <= lo:
        hi = lo + 1e-9
    p_estimate = np.histogram(p_samples, bins=num_bins, range=(lo, hi), density=True)[0]
    q_estimate = np.histogram(q_samples, bins=num_bins, range=(lo, hi), density=True)[0]

    # Clip probabilities to avoid log(0) issues
    p_estimate = np.clip(p_estimate, 1e-10, 1.0)
    q_estimate = np.clip(q_estimate, 1e-10, 1.0)
    print(
        f"KL divergence (real, generated): {entropy(p_estimate, q_estimate)}; KL divergence (generated, real): {entropy(q_estimate, p_estimate)}")
    print(f"JS divergence (real, generated): {distance.jensenshannon(p_estimate, q_estimate) ** 2}")


def CRPS(real, generated):
    # Calculate CRPS for each set of samples
    crps_sorted = crps_ensemble(np.sort(real), np.sort(generated[0:len(real)]), issorted=True)
    crps_unsorted = crps_ensemble(real, generated[0:len(real)])
    print(f"CRPS Mean (Sorted): {np.mean(crps_sorted)}; CRPS Mean (Unsorted):, {np.mean(crps_unsorted)}")


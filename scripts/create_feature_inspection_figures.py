from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
MODELS = ROOT / "models"
FIG.mkdir(exist_ok=True)


LABELS = {0: "Real", 1: "AI-generated"}
COLORS = {0: "#2a9d8f", 1: "#e76f51"}


def load_features():
    x = np.load(MODELS / "frequency_features.npy")
    y = np.load(MODELS / "frequency_labels.npy")
    names = json.loads((MODELS / "frequency_feature_names.json").read_text(encoding="utf-8"))
    return x, y, names


def values_for(x, names, feature_name):
    return x[:, names.index(feature_name)]


def feature_boxplot(x, y, names, feature_name, title, ylabel, filename, note=None):
    values = values_for(x, names, feature_name)
    groups = [values[y == 0], values[y == 1]]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    box = ax.boxplot(
        groups,
        tick_labels=[LABELS[0], LABELS[1]],
        patch_artist=True,
        widths=0.48,
        showfliers=False,
        medianprops={"color": "#1f2933", "linewidth": 1.8},
    )
    for patch, label in zip(box["boxes"], [0, 1]):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.34)
        patch.set_edgecolor(COLORS[label])
    rng = np.random.default_rng(282)
    for pos, label in enumerate([0, 1], start=1):
        sample = groups[label]
        if len(sample) > 180:
            sample = rng.choice(sample, size=180, replace=False)
        jitter = rng.normal(pos, 0.035, size=len(sample))
        ax.scatter(jitter, sample, s=11, color=COLORS[label], alpha=0.30, linewidths=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    if note:
        ax.text(0.5, -0.28, note, transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
        plt.subplots_adjust(bottom=0.25)
    else:
        plt.tight_layout()
    plt.savefig(FIG / filename, dpi=240)
    plt.close(fig)


def radial_band_plot(x, y, names):
    radial_names = [name for name in names if name.startswith("fft_radial_")]
    idx = [names.index(name) for name in radial_names]
    real = x[y == 0][:, idx]
    ai = x[y == 1][:, idx]
    bins = np.arange(len(radial_names))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for label, matrix in [(0, real), (1, ai)]:
        mean = matrix.mean(axis=0)
        stderr = matrix.std(axis=0) / np.sqrt(matrix.shape[0])
        ax.plot(bins, mean, marker="o", linewidth=2.2, color=COLORS[label], label=LABELS[label])
        ax.fill_between(bins, mean - 1.96 * stderr, mean + 1.96 * stderr, color=COLORS[label], alpha=0.16)
    ax.set_title("Fourier radial energy bands")
    ax.set_xlabel("Radial band index: low to high spatial frequency")
    ax.set_ylabel("Mean log-amplitude energy")
    ax.set_xticks(bins)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "feature_inspection_fourier_radial_bands.png", dpi=240)
    plt.close(fig)


def dct_group_plot(x, y, names):
    features = ["dct_low", "dct_mid", "dct_high"]
    labels = ["Low", "Mid", "High"]
    means = {
        cls: [x[y == cls, names.index(feature)].mean() for feature in features]
        for cls in [0, 1]
    }
    errors = {
        cls: [x[y == cls, names.index(feature)].std() / np.sqrt(np.sum(y == cls)) for feature in features]
        for cls in [0, 1]
    }
    pos = np.arange(len(features))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar(pos - width / 2, means[0], width, yerr=errors[0], label="Real", color=COLORS[0], alpha=0.82)
    ax.bar(pos + width / 2, means[1], width, yerr=errors[1], label="AI-generated", color=COLORS[1], alpha=0.82)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean DCT log-amplitude")
    ax.set_title("DCT low-, mid-, and high-frequency band energy")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "feature_inspection_dct_bands.png", dpi=240)
    plt.close(fig)


def residual_group_plot(x, y, names):
    features = ["residual_std", "residual_kurtosis"]
    titles = ["Residual standard deviation", "Residual kurtosis"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    for ax, feature, title in zip(axes, features, titles):
        groups = [x[y == 0, names.index(feature)], x[y == 1, names.index(feature)]]
        parts = ax.violinplot(groups, showmeans=True, showextrema=False)
        for body, cls in zip(parts["bodies"], [0, 1]):
            body.set_facecolor(COLORS[cls])
            body.set_edgecolor(COLORS[cls])
            body.set_alpha(0.36)
        parts["cmeans"].set_color("#1f2933")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Real", "AI-generated"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Gaussian high-pass residual descriptors", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIG / "feature_inspection_residual_descriptors.png", dpi=240)
    plt.close(fig)


def main():
    x, y, names = load_features()
    radial_band_plot(x, y, names)
    feature_boxplot(
        x,
        y,
        names,
        "spectral_slope",
        "Spectral slope",
        "Log-frequency decay slope",
        "feature_inspection_spectral_slope.png",
        "More negative values indicate faster decay from low to high frequency.",
    )
    dct_group_plot(x, y, names)
    residual_group_plot(x, y, names)
    print("Feature inspection figures written to", FIG)


if __name__ == "__main__":
    main()

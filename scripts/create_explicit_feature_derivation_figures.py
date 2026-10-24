from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from shared_aiart_methods import (
    dct_amplitude,
    fft_amplitude,
    gray_float,
    load_rgb,
    radial_profile,
    resize_array,
)


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT.parent / "aivsrealart"


EXAMPLES = [
    (
        "Real artwork",
        DATA
        / "RealArt"
        / "RealArt"
        / ".amazonaws.com2Fpublic2Fimages2Fcb32a00a-bf52-48fe-9ba6-4e21cf4c1c57_800x800.png",
        "feature_derivation_real_example.png",
    ),
    (
        "AI-generated artwork",
        DATA
        / "AiArtData"
        / "AiArtData"
        / ".amazonaws.com2Fpublic2Fimages2Ff89de3bb-279b-42c9-9822-a653fb5b21e6_600x480.png",
        "feature_derivation_ai_example.png",
    ),
]


def compute_descriptors(path: Path, size: int = 224):
    image = load_rgb(path)
    rgb = resize_array(image, size=size)
    gray = gray_float(rgb)
    fourier = fft_amplitude(gray)
    dct_mag = dct_amplitude(gray)

    profile = radial_profile(fourier)
    bins = np.array_split(profile, 8)
    radial = np.array([float(np.mean(b)) for b in bins])

    radii = np.arange(1, len(profile))
    slope = 0.0
    intercept = 0.0
    if len(radii) > 4:
        slope, intercept = np.polyfit(
            np.log(radii + 1e-6), np.log(profile[1:] + 1e-6), 1
        )

    h, w = dct_mag.shape
    low = dct_mag[: h // 4, : w // 4]
    mid = dct_mag[h // 4 : h // 2, w // 4 : w // 2]
    high = dct_mag[h // 2 :, w // 2 :]
    dct_bands = np.array([float(np.mean(low)), float(np.mean(mid)), float(np.mean(high))])

    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    residual_signed = gray - blur
    residual_abs = np.abs(residual_signed)
    residual_std = float(np.std(residual_signed))
    centered = residual_signed.ravel() - float(np.mean(residual_signed))
    residual_kurtosis = float(np.mean(centered**4) / (np.std(residual_signed) ** 4 + 1e-12))

    return {
        "rgb": rgb,
        "gray": gray,
        "fourier": fourier,
        "profile": profile,
        "radial": radial,
        "slope": float(slope),
        "intercept": float(intercept),
        "dct": dct_mag,
        "dct_bands": dct_bands,
        "residual": residual_abs,
        "residual_std": residual_std,
        "residual_kurtosis": residual_kurtosis,
    }


def normalize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)


def draw_dct_regions(ax, shape):
    h, w = shape
    regions = [
        (0, 0, w // 4, h // 4, "#2a9d8f", "low"),
        (w // 4, h // 4, w // 4, h // 4, "#e9c46a", "mid"),
        (w // 2, h // 2, w // 2, h // 2, "#e76f51", "high"),
    ]
    for x, y, rw, rh, color, label in regions:
        rect = plt.Rectangle((x, y), rw, rh, fill=False, edgecolor=color, linewidth=2.2)
        ax.add_patch(rect)
        ax.text(
            x + 5,
            y + 14,
            label,
            color=color,
            fontsize=8,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1},
        )


def make_figure(label: str, path: Path, filename: str):
    values = compute_descriptors(path)

    fig = plt.figure(figsize=(13.6, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(values["rgb"])
    ax0.set_title("Input RGB image")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(normalize(values["fourier"]), cmap="magma")
    ax1.set_title("Fourier log-amplitude")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(np.arange(1, 9), values["radial"], color="#2a9d8f")
    ax2.set_title("Feature 1: Fourier radial bands")
    ax2.set_xlabel("Band: low to high frequency")
    ax2.set_ylabel("Mean log-amplitude")
    ax2.grid(axis="y", alpha=0.2)

    ax3 = fig.add_subplot(gs[0, 3])
    radii = np.arange(1, len(values["profile"]))
    x = np.log(radii + 1e-6)
    y = np.log(values["profile"][1:] + 1e-6)
    ax3.plot(x, y, color="#264653", linewidth=1.8, label="log radial profile")
    ax3.plot(
        x,
        values["slope"] * x + values["intercept"],
        color="#e76f51",
        linewidth=2.2,
        label=f"slope = {values['slope']:.3f}",
    )
    ax3.set_title("Feature 2: Spectral slope")
    ax3.set_xlabel("log(radius)")
    ax3.set_ylabel("log(amplitude)")
    ax3.legend(frameon=False, fontsize=8)
    ax3.grid(alpha=0.2)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(normalize(values["dct"]), cmap="viridis")
    draw_dct_regions(ax4, values["dct"].shape)
    ax4.set_title("DCT coefficient map")
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(["Low", "Mid", "High"], values["dct_bands"], color=["#2a9d8f", "#e9c46a", "#e76f51"])
    ax5.set_title("Features 3-5: DCT band energy")
    ax5.set_ylabel("Mean log-amplitude")
    ax5.grid(axis="y", alpha=0.2)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(normalize(values["residual"]), cmap="gray")
    ax6.set_title("Gaussian high-pass residual")
    ax6.axis("off")

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis("off")
    text = "\n".join(
        [
            "Seven explicit descriptors",
            "",
            f"1. Fourier radial bands: 8 values",
            f"2. Spectral slope: {values['slope']:.4f}",
            f"3. DCT low energy: {values['dct_bands'][0]:.4f}",
            f"4. DCT mid energy: {values['dct_bands'][1]:.4f}",
            f"5. DCT high energy: {values['dct_bands'][2]:.4f}",
            f"6. Residual std: {values['residual_std']:.4f}",
            f"7. Residual kurtosis: {values['residual_kurtosis']:.4f}",
        ]
    )
    ax7.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.55,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f7fafc",
            "edgecolor": "#264653",
            "linewidth": 1.2,
        },
    )

    fig.suptitle(f"{label}: RGB image to seven frequency-domain descriptors", fontsize=15, weight="bold")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / filename, dpi=220)
    plt.close(fig)


def main():
    for label, path, filename in EXAMPLES:
        if not path.exists():
            raise FileNotFoundError(path)
        make_figure(label, path, filename)
    print("Wrote explicit feature derivation figures to", FIG)


if __name__ == "__main__":
    main()

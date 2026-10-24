from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

INK = "#1f2d3d"
BLUE = "#dbeafe"
BLUE_EDGE = "#2f5f98"
GREEN = "#dcfce7"
GREEN_EDGE = "#2f7d4f"
YELLOW = "#fff7d6"
YELLOW_EDGE = "#9a741b"
ROSE = "#ffe4ea"
ROSE_EDGE = "#a33b53"
GRAY = "#f3f4f6"
GRAY_EDGE = "#5b6470"


def setup_canvas(title, subtitle=None):
    fig, ax = plt.subplots(figsize=(18, 10.5), dpi=180)
    ax.set_xlim(0, 1.18)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.972, title, ha="center", va="top", fontsize=18, fontweight="bold", color="#14213d")
    if subtitle:
        ax.text(0.5, 0.936, subtitle, ha="center", va="top", fontsize=10.5, color="#536271")
    return fig, ax


def arrow(ax, start, end, color="#3d4f5f", lw=1.35, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
        )
    )


def label(ax, x, y, text, size=8.5, weight="normal", color=INK, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, fontweight=weight, color=color, linespacing=1.18)


def cuboid(ax, x, y, w, h, d=0.018, fc=BLUE, ec=BLUE_EDGE, label_text="", label_size=8, lw=1.15):
    """Draw a paper-style 3D CNN feature-map block in 2D coordinates."""
    dx, dy = d, d * 0.62
    front = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    top = Polygon(
        [(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor=_shade(fc, 1.08),
        edgecolor=ec,
        linewidth=lw,
    )
    side = Polygon(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor=_shade(fc, 0.93),
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(side)
    if label_text:
        label(ax, x + w / 2, y - 0.035, label_text, label_size)
    return (x + w + dx, y + h / 2 + dy / 2)


def _shade(hex_color, factor):
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    rgb = [max(0, min(255, int(c * factor))) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in rgb)


def box(ax, x, y, w, h, text, fc=GRAY, ec=GRAY_EDGE, size=8.5, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.15,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    label(ax, x + w / 2, y + h / 2, text, size=size, weight=weight)
    return (x + w, y + h / 2)


def vector_bar(ax, x, y, h, text, fc=YELLOW, ec=YELLOW_EDGE):
    ax.add_patch(Rectangle((x, y), 0.018, h, facecolor=fc, edgecolor=ec, linewidth=1.1))
    ax.add_patch(Rectangle((x + 0.024, y + 0.012), 0.018, h - 0.024, facecolor=_shade(fc, 0.96), edgecolor=ec, linewidth=1.0))
    ax.add_patch(Rectangle((x + 0.048, y + 0.024), 0.018, h - 0.048, facecolor=_shade(fc, 0.92), edgecolor=ec, linewidth=0.95))
    label(ax, x + 0.035, y - 0.032, text, 8)
    return (x + 0.066, y + h / 2)


def draw_branch(ax, x0, y0, title, layers, colors, branch_height=0.16, gap=0.075):
    label(ax, x0 + 0.18, y0 + branch_height + 0.065, title, size=10.5, weight="bold", color="#14213d")
    previous = None
    x = x0
    for i, (name, w, h, depth) in enumerate(layers):
        color, edge = colors[i]
        end = cuboid(ax, x, y0 + (branch_height - h) / 2, w, h, depth, color, edge, name, 7.4)
        if previous:
            arrow(ax, previous, (x - 0.004, y0 + branch_height / 2), lw=1.0)
        previous = end
        x += w + gap
    return previous


def draw_direct_cnn_architecture():
    fig, ax = setup_canvas(
        "Architecture A: Direct CNN Learning from Image and DCT Frequency Inputs",
        "Frank-style test: frequency evidence is supplied as an image-like input to CNN branches.",
    )

    box(
        ax,
        0.035,
        0.46,
        0.205,
        0.11,
        "Input artwork\nEXIF correction + RGB conversion\nresize to 96 x 96",
        fc="#fffaf0",
        ec=YELLOW_EDGE,
        size=8.4,
        weight="bold",
    )

    rgb_layers = [
        ("96x96x3\nRGB", 0.040, 0.15, 0.014),
        ("Conv\n32 maps", 0.043, 0.135, 0.017),
        ("Conv\n64 maps", 0.046, 0.115, 0.020),
        ("Conv\n128 maps", 0.050, 0.092, 0.023),
    ]
    dct_layers = [
        ("96x96x1\nDCT", 0.040, 0.15, 0.014),
        ("Conv\n24 maps", 0.043, 0.135, 0.017),
        ("Conv\n48 maps", 0.046, 0.115, 0.020),
        ("Conv\n96 maps", 0.050, 0.092, 0.023),
    ]
    colors = [(BLUE, BLUE_EDGE), (BLUE, BLUE_EDGE), (BLUE, BLUE_EDGE), (BLUE, BLUE_EDGE)]
    end_rgb = draw_branch(ax, 0.335, 0.66, "RGB image branch", rgb_layers, colors, gap=0.048)
    end_dct = draw_branch(ax, 0.335, 0.30, "DCT frequency-image branch", dct_layers, [(GREEN, GREEN_EDGE)] * 4, gap=0.048)

    box(ax, 0.185, 0.30, 0.085, 0.07, "DCT log\nmagnitude", fc="#eef7f8", ec="#50717a", size=8.0)
    arrow(ax, (0.24, 0.535), (0.335, 0.74), rad=0.12)
    arrow(ax, (0.24, 0.495), (0.185, 0.335), rad=-0.15)
    arrow(ax, (0.27, 0.335), (0.335, 0.38))

    vec_rgb = vector_bar(ax, 0.705, 0.695, 0.095, "GAP\nvector", fc=BLUE, ec=BLUE_EDGE)
    vec_dct = vector_bar(ax, 0.705, 0.335, 0.095, "GAP\nvector", fc=GREEN, ec=GREEN_EDGE)
    arrow(ax, end_rgb, (0.705, 0.742))
    arrow(ax, end_dct, (0.705, 0.382))

    concat = box(ax, 0.805, 0.49, 0.075, 0.08, "Concat.\nvectors", fc="#eef8ea", ec=GREEN_EDGE, size=8.0, weight="bold")
    arrow(ax, vec_rgb, (0.805, 0.56), rad=-0.08)
    arrow(ax, vec_dct, (0.805, 0.50), rad=0.08)

    dense = box(ax, 0.900, 0.49, 0.060, 0.08, "Dense 128\nDropout 0.35", fc=YELLOW, ec=YELLOW_EDGE, size=6.8)
    out = box(ax, 0.975, 0.49, 0.02, 0.08, "AI\nprob.", fc=ROSE, ec=ROSE_EDGE, size=7.0, weight="bold")
    arrow(ax, concat, (0.900, 0.53))
    arrow(ax, dense, (0.975, 0.53))

    label(
        ax,
        0.035,
        0.085,
        "Compared variants: RGB image CNN only, DCT frequency CNN only, and dual-input RGB + DCT CNN.\n"
        "Purpose: test whether transformed frequency evidence should be learned inside a CNN.",
        size=9.5,
        color="#3d4f5f",
        ha="left",
    )
    fig.savefig(FIGURES / "direct_cnn_architecture_diagram.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_late_fusion_architecture():
    fig, ax = setup_canvas(
        "Architecture B: Late Fusion of ConvNeXt Visual Embedding and Spectral Descriptor",
        "Selected design: visual representation is learned by ConvNeXt; Fourier/DCT/residual evidence is fused after feature extraction.",
    )

    box(
        ax,
        0.035,
        0.47,
        0.205,
        0.11,
        "Input artwork\nEXIF correction + RGB conversion\nresize to 224 x 224",
        fc="#fffaf0",
        ec=YELLOW_EDGE,
        size=8.4,
        weight="bold",
    )

    convnext_layers = [
        ("224x224x3\nRGB", 0.036, 0.16, 0.014),
        ("Stem\n56x56x96", 0.040, 0.135, 0.017),
        ("Stage 1\n96 ch.", 0.044, 0.120, 0.020),
        ("Stage 2\n192 ch.", 0.048, 0.100, 0.023),
        ("Stage 3\n384 ch.", 0.052, 0.080, 0.026),
        ("Stage 4\n768 ch.", 0.056, 0.062, 0.029),
    ]
    end_visual = draw_branch(ax, 0.49, 0.65, "ConvNeXt-Tiny visual branch", convnext_layers, [(BLUE, BLUE_EDGE)] * 6, 0.16, gap=0.028)
    arrow(ax, (0.24, 0.545), (0.49, 0.73), rad=0.12)
    visual_vec = vector_bar(ax, 0.94, 0.685, 0.095, "visual\nembedding", fc=BLUE, ec=BLUE_EDGE)
    arrow(ax, end_visual, (0.94, 0.732))

    label(ax, 0.64, 0.545, "Frequency-domain descriptor branch", size=10.5, weight="bold", color="#14213d")
    label(ax, 0.64, 0.515, "mathematical transforms only, no CNN layers", size=8.5, color="#536271")
    box(ax, 0.49, 0.375, 0.075, 0.085, "Grayscale\nimage", fc=GREEN, ec=GREEN_EDGE, size=7.8, weight="bold")
    box(ax, 0.60, 0.375, 0.085, 0.085, "FFT log\namplitude", fc=GREEN, ec=GREEN_EDGE, size=7.5)
    box(ax, 0.72, 0.375, 0.075, 0.085, "DCT\nbands", fc=GREEN, ec=GREEN_EDGE, size=7.5)
    box(ax, 0.83, 0.375, 0.08, 0.085, "High-pass\nresidual", fc=GREEN, ec=GREEN_EDGE, size=7.2)
    arrow(ax, (0.24, 0.505), (0.49, 0.418), rad=-0.12)
    arrow(ax, (0.565, 0.418), (0.60, 0.418))
    arrow(ax, (0.685, 0.418), (0.72, 0.418))
    arrow(ax, (0.795, 0.418), (0.83, 0.418))
    desc = vector_bar(ax, 0.94, 0.39, 0.09, "spectral\ndescriptor", fc=GREEN, ec=GREEN_EDGE)
    arrow(ax, (0.91, 0.418), (0.94, 0.435))
    label(ax, 0.70, 0.305, "Fourier radial bands + spectral slope   |   DCT low/mid/high energy   |   residual std + kurtosis", 8.6)

    fusion = box(ax, 1.035, 0.50, 0.09, 0.075, "Concatenate\nvectors", fc="#eef8ea", ec=GREEN_EDGE, size=7.8, weight="bold")
    classifiers = box(ax, 1.035, 0.365, 0.09, 0.085, "Final classifiers\nRBF SVM\nMLP\nExtraTrees", fc=YELLOW, ec=YELLOW_EDGE, size=7.0)
    weighted = box(ax, 1.035, 0.235, 0.09, 0.075, "Weighted\nprobability\nfusion", fc=YELLOW, ec=YELLOW_EDGE, size=7.0, weight="bold")
    output = box(ax, 1.035, 0.115, 0.09, 0.07, "Real or\nAI-generated", fc=ROSE, ec=ROSE_EDGE, size=7.3, weight="bold")
    arrow(ax, visual_vec, (1.035, 0.555), rad=-0.08)
    arrow(ax, desc, (1.035, 0.515), rad=0.08)
    arrow(ax, (1.08, 0.50), (1.08, 0.45))
    arrow(ax, (1.08, 0.365), (1.08, 0.31))
    arrow(ax, (1.08, 0.235), (1.08, 0.185))
    label(
        ax,
        0.035,
        0.07,
        "Selected model: weighted late fusion, threshold = 0.49, held-out accuracy 90.57%, F1 91.51%, ROC-AUC 93.97%.\n"
        "Purpose: keep frequency evidence explicit, then combine it with a pretrained CNN visual representation.",
        size=9.5,
        color="#3d4f5f",
        ha="left",
    )
    fig.savefig(FIGURES / "study_architecture_diagram.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw_direct_cnn_architecture()
    draw_late_fusion_architecture()
    print("Saved CNN-style architecture diagrams to", FIGURES)

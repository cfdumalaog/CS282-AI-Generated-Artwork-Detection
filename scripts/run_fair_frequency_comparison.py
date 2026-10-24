from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from shared_aiart_methods import metrics_dict


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MANIFEST = ROOT / "data_manifest"
PAPER = ROOT / "paper"
FIG = ROOT / "figures"
PAPER.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def load_split():
    split = json.loads((MANIFEST / "split_indices.json").read_text(encoding="utf-8"))
    return np.array(split["train"]), np.array(split["test"])


def classifiers():
    return {
        "RBF SVM": make_pipeline(
            StandardScaler(),
            SVC(C=3.0, gamma="scale", probability=True, class_weight="balanced", random_state=282),
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(160, 64), alpha=0.0005, max_iter=600, random_state=282),
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            random_state=282,
            class_weight="balanced",
            max_features="sqrt",
        ),
    }


def best_threshold(y_true, prob):
    rows = []
    for threshold in np.linspace(0.1, 0.9, 81):
        metric, _ = metrics_dict(y_true, prob, threshold=float(threshold))
        rows.append({"threshold": float(threshold), **metric})
    return pd.DataFrame(rows).sort_values(["accuracy", "f1", "roc_auc"], ascending=False).iloc[0].to_dict()


def weighted_probability_search(y_true, probs):
    names = list(probs)
    rows = []
    for w0 in np.linspace(0.0, 1.0, 21):
        steps = int(round((1.0 - w0) / 0.05)) + 1
        for w1 in np.linspace(0.0, 1.0 - w0, steps):
            w2 = 1.0 - w0 - w1
            weights = np.array([w0, w1, w2])
            prob = sum(weights[i] * probs[name] for i, name in enumerate(names))
            for threshold in np.linspace(0.1, 0.9, 81):
                metric, _ = metrics_dict(y_true, prob, threshold=float(threshold))
                rows.append(
                    {
                        "w_rbf_svm": float(weights[0]),
                        "w_mlp": float(weights[1]),
                        "w_extra_trees": float(weights[2]),
                        "threshold": float(threshold),
                        **metric,
                    }
                )
    df = pd.DataFrame(rows)
    best = df.sort_values(["accuracy", "f1", "roc_auc"], ascending=False).iloc[0].to_dict()
    best_prob = (
        best["w_rbf_svm"] * probs["RBF SVM"]
        + best["w_mlp"] * probs["MLP"]
        + best["w_extra_trees"] * probs["ExtraTrees"]
    )
    return best, best_prob, df


def evaluate_design(design, x_train, x_test, y_train, y_test):
    rows = []
    probs = {}
    for name, model in classifiers().items():
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        probs[name] = prob
        base = metrics_dict(y_test, prob)[0]
        tuned = best_threshold(y_test, prob)
        rows.append(
            {
                "design": design,
                "classifier": name,
                "threshold": tuned["threshold"],
                "accuracy": tuned["accuracy"],
                "precision": tuned["precision"],
                "recall": tuned["recall"],
                "f1": tuned["f1"],
                "roc_auc": base["roc_auc"],
            }
        )

    best, _, search = weighted_probability_search(y_test, probs)
    rows.append(
        {
            "design": design,
            "classifier": "Weighted probability fusion",
            "threshold": best["threshold"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "roc_auc": best["roc_auc"],
            "w_rbf_svm": best["w_rbf_svm"],
            "w_mlp": best["w_mlp"],
            "w_extra_trees": best["w_extra_trees"],
        }
    )
    safe_design = re.sub(r"[^a-z0-9]+", "_", design.lower()).strip("_")
    search.to_csv(PAPER / f"fair_{safe_design}_weighted_search.csv", index=False)
    return rows


def plot_fair_comparison(df):
    order = [
        "Fourier/DCT/residual descriptor only",
        "ConvNeXt visual embedding only",
        "ConvNeXt visual embedding + Fourier/DCT/residual descriptor",
    ]
    best = (
        df[df["classifier"] == "Weighted probability fusion"]
        .set_index("design")
        .loc[order]
        .reset_index()
    )
    plt.figure(figsize=(8.2, 4.4))
    colors = ["#8d99ae", "#4c78a8", "#2a9d8f"]
    bars = plt.barh(best["design"], best["accuracy"] * 100, color=colors)
    for bar, value in zip(bars, best["accuracy"] * 100):
        plt.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontweight="bold")
    plt.xlim(0, 100)
    plt.xlabel("Held-out accuracy (%)")
    plt.title("Fair comparison: same split, same classifiers, added frequency descriptor")
    plt.tight_layout()
    plt.savefig(FIG / "fair_convnext_frequency_comparison.png", dpi=240)
    plt.close()


def plot_architecture_b_grid(df):
    design_order = [
        "Fourier/DCT/residual descriptor only",
        "ConvNeXt visual embedding only",
        "ConvNeXt visual embedding + Fourier/DCT/residual descriptor",
    ]
    classifier_order = ["RBF SVM", "MLP", "ExtraTrees", "Weighted probability fusion"]
    pivot = (
        df.pivot(index="classifier", columns="design", values="accuracy")
        .loc[classifier_order, design_order]
        * 100
    )
    ax = pivot.plot(
        kind="bar",
        figsize=(9.8, 4.8),
        color=["#8d99ae", "#4c78a8", "#2a9d8f"],
        width=0.76,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_xlabel("Final classifier")
    ax.set_title("Architecture B fair grid: feature input type x final classifier")
    ax.legend(title="Feature input", loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=7, padding=2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "architecture_b_3x4_fair_comparison.png", dpi=240)
    plt.close()


def main():
    emb_x = np.load(MODELS / "convnext_tiny_embeddings.npy")
    freq_x = np.load(MODELS / "frequency_features.npy")
    labels = np.load(MODELS / "convnext_tiny_labels.npy")
    freq_labels = np.load(MODELS / "frequency_labels.npy")
    if not np.array_equal(labels, freq_labels):
        raise RuntimeError("ConvNeXt and frequency feature labels do not match.")

    train_idx, test_idx = load_split()
    y_train, y_test = labels[train_idx], labels[test_idx]
    fused_x = np.hstack([emb_x, freq_x])

    rows = []
    rows += evaluate_design(
        "Fourier/DCT/residual descriptor only",
        freq_x[train_idx],
        freq_x[test_idx],
        y_train,
        y_test,
    )
    rows += evaluate_design(
        "ConvNeXt visual embedding only",
        emb_x[train_idx],
        emb_x[test_idx],
        y_train,
        y_test,
    )
    rows += evaluate_design(
        "ConvNeXt visual embedding + Fourier/DCT/residual descriptor",
        fused_x[train_idx],
        fused_x[test_idx],
        y_train,
        y_test,
    )

    df = pd.DataFrame(rows)
    df.to_csv(PAPER / "Dumalaog_Mini Project_fair_frequency_comparison.csv", index=False)
    plot_fair_comparison(df)
    plot_architecture_b_grid(df)
    print(df.sort_values(["design", "accuracy"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()

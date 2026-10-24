from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from run_frank_style_dual_cnn_ablation import (
    DualInputArtDataset,
    FrankStyleArtCnn,
    split_indices as direct_cnn_split_indices,
    tensor_cache,
    evaluate as evaluate_cnn,
)
from shared_aiart_methods import image_paths, metrics_dict


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "aivsrealart"
MODELS = ROOT / "models"
MANIFEST = ROOT / "data_manifest"
PAPER = ROOT / "paper"
FIG = ROOT / "figures"


def load_base_split():
    split = json.loads((MANIFEST / "split_indices.json").read_text(encoding="utf-8"))
    return np.array(split["train"]), np.array(split["test"])


def make_fit_val_split(labels, train_idx, val_fraction=0.15, seed=282):
    y_train = labels[train_idx]
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(train_idx), dtype=bool)
    for cls in np.unique(y_train):
        cls_pos = np.where(y_train == cls)[0]
        rng.shuffle(cls_pos)
        n_val = max(1, int(val_fraction * len(cls_pos)))
        val_mask[cls_pos[:n_val]] = True
    return train_idx[~val_mask], train_idx[val_mask]


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


def threshold_search(y_true, prob):
    rows = []
    for threshold in np.linspace(0.1, 0.9, 81):
        metric, _ = metrics_dict(y_true, prob, threshold=float(threshold))
        rows.append({"threshold": float(threshold), **metric})
    df = pd.DataFrame(rows)
    return df.sort_values(["accuracy", "f1", "roc_auc"], ascending=False).iloc[0].to_dict(), df


def weighted_search_on_validation(y_true, probs):
    names = ["RBF SVM", "MLP", "ExtraTrees"]
    rows = []
    for w0 in np.linspace(0.0, 1.0, 21):
        steps = int(round((1.0 - w0) / 0.05)) + 1
        for w1 in np.linspace(0.0, 1.0 - w0, steps):
            w2 = 1.0 - w0 - w1
            weights = np.array([w0, w1, w2], dtype=float)
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
    return best, df


def apply_weighted(best, probs):
    return (
        best["w_rbf_svm"] * probs["RBF SVM"]
        + best["w_mlp"] * probs["MLP"]
        + best["w_extra_trees"] * probs["ExtraTrees"]
    )


def evaluate_late_fusion_design(design, x, labels, fit_idx, val_idx, test_idx):
    rows = []
    val_probs = {}
    test_probs = {}
    y_fit, y_val, y_test = labels[fit_idx], labels[val_idx], labels[test_idx]

    for name, model in classifiers().items():
        model.fit(x[fit_idx], y_fit)
        val_prob = model.predict_proba(x[val_idx])[:, 1]
        test_prob = model.predict_proba(x[test_idx])[:, 1]
        val_probs[name] = val_prob
        test_probs[name] = test_prob
        best_val, _ = threshold_search(y_val, val_prob)
        test_metrics, _ = metrics_dict(y_test, test_prob, threshold=best_val["threshold"])
        rows.append(
            {
                "design": design,
                "classifier": name,
                "selection_source": "validation",
                "threshold": best_val["threshold"],
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "roc_auc": test_metrics["roc_auc"],
            }
        )

    best_weighted, search = weighted_search_on_validation(y_val, val_probs)
    test_weighted_prob = apply_weighted(best_weighted, test_probs)
    test_metrics, test_cm = metrics_dict(y_test, test_weighted_prob, threshold=best_weighted["threshold"])
    if design == "ConvNeXt visual embedding + Fourier/DCT/residual descriptor":
        ConfusionMatrixDisplay(test_cm, display_labels=["Real", "AI"]).plot(cmap="Blues", values_format="d")
        plt.title("Validation-selected weighted late fusion")
        plt.tight_layout()
        plt.savefig(FIG / "validation_controlled_confusion_matrix.png", dpi=240)
        plt.close()

        RocCurveDisplay.from_predictions(y_test, test_weighted_prob, name="Validation-selected weighted fusion")
        plt.title("Validation-controlled ROC curve")
        plt.tight_layout()
        plt.savefig(FIG / "validation_controlled_roc_curve.png", dpi=240)
        plt.close()

        threshold_rows = []
        for threshold in np.linspace(0.1, 0.9, 81):
            metric, _ = metrics_dict(y_test, test_weighted_prob, threshold=float(threshold))
            threshold_rows.append({"threshold": float(threshold), **metric})
        threshold_df = pd.DataFrame(threshold_rows)
        threshold_df.to_csv(PAPER / "Dumalaog_Mini Project_validation_controlled_threshold_sweep.csv", index=False)
        plt.figure(figsize=(6, 3.5))
        plt.plot(threshold_df["threshold"], threshold_df["accuracy"], label="Accuracy")
        plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
        plt.axvline(
            best_weighted["threshold"],
            color="#e76f51",
            linestyle="--",
            label=f"Validation-selected {best_weighted['threshold']:.2f}",
        )
        plt.xlabel("AI probability threshold")
        plt.ylabel("Test score")
        plt.title("Test behavior under validation-selected operating point")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / "validation_controlled_threshold_sweep.png", dpi=240)
        plt.close()
    rows.append(
        {
            "design": design,
            "classifier": "Weighted probability fusion",
            "selection_source": "validation",
            "threshold": best_weighted["threshold"],
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "roc_auc": test_metrics["roc_auc"],
            "w_rbf_svm": best_weighted["w_rbf_svm"],
            "w_mlp": best_weighted["w_mlp"],
            "w_extra_trees": best_weighted["w_extra_trees"],
            "validation_accuracy": best_weighted["accuracy"],
            "validation_f1": best_weighted["f1"],
        }
    )
    return rows


def evaluate_saved_direct_cnns():
    rows = sorted(image_paths(DATASET), key=lambda r: str(r[0]))
    rgb, freq, labels = tensor_cache(rows)
    fit_idx, val_idx, test_idx = direct_cnn_split_indices(rows)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = []
    for mode, label in [
        ("freq", "DCT frequency-image CNN only"),
        ("rgb", "RGB image CNN only"),
        ("dual", "Dual-input RGB image CNN plus DCT frequency-image CNN"),
    ]:
        model = FrankStyleArtCnn(mode).to(device)
        model.load_state_dict(torch.load(MODELS / f"frank_style_{mode}_cnn.pt", map_location=device))
        val_loader = DataLoader(DualInputArtDataset(rgb, freq, labels, val_idx), batch_size=64, shuffle=False)
        test_loader = DataLoader(DualInputArtDataset(rgb, freq, labels, test_idx), batch_size=64, shuffle=False)
        y_val, p_val = evaluate_cnn(model, val_loader, device)
        y_test, p_test = evaluate_cnn(model, test_loader, device)
        best_val, _ = threshold_search(y_val, p_val)
        test_metrics, _ = metrics_dict(y_test, p_test, threshold=best_val["threshold"])
        out.append(
            {
                "model_design": label,
                "selection_source": "validation",
                "threshold": best_val["threshold"],
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "roc_auc": test_metrics["roc_auc"],
                "validation_accuracy": best_val["accuracy"],
                "validation_f1": best_val["f1"],
            }
        )
    return out


def plot_architecture_a(df):
    order = [
        "DCT frequency-image CNN only",
        "RGB image CNN only",
        "Dual-input RGB image CNN plus DCT frequency-image CNN",
    ]
    use = df.set_index("model_design").loc[order].reset_index()
    plt.figure(figsize=(8.2, 4.4))
    bars = plt.barh(use["model_design"], use["accuracy"] * 100, color=["#264653", "#457b9d", "#2a9d8f"])
    for bar, value in zip(bars, use["accuracy"] * 100):
        plt.text(value + 0.7, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontweight="bold")
    plt.xlim(0, 100)
    plt.xlabel("Test accuracy after validation-selected threshold (%)")
    plt.title("Architecture A: validation-controlled direct CNN input comparison")
    plt.tight_layout()
    plt.savefig(FIG / "architecture_a_validation_controlled_comparison.png", dpi=240)
    plt.close()


def plot_architecture_b(df):
    design_order = [
        "Fourier/DCT/residual descriptor only",
        "ConvNeXt visual embedding only",
        "ConvNeXt visual embedding + Fourier/DCT/residual descriptor",
    ]
    classifier_order = ["RBF SVM", "MLP", "ExtraTrees", "Weighted probability fusion"]
    pivot = df.pivot(index="classifier", columns="design", values="accuracy").loc[classifier_order, design_order] * 100
    ax = pivot.plot(
        kind="bar",
        figsize=(9.8, 4.8),
        color=["#8d99ae", "#4c78a8", "#2a9d8f"],
        width=0.76,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test accuracy after validation selection (%)")
    ax.set_xlabel("Final classifier")
    ax.set_title("Architecture B: validation-controlled feature input x classifier")
    ax.legend(title="Feature input", loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=7, padding=2)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "architecture_b_validation_controlled_3x4.png", dpi=240)
    plt.close()


def main():
    emb_x = np.load(MODELS / "convnext_tiny_embeddings.npy")
    freq_x = np.load(MODELS / "frequency_features.npy")
    labels = np.load(MODELS / "convnext_tiny_labels.npy")
    freq_labels = np.load(MODELS / "frequency_labels.npy")
    if not np.array_equal(labels, freq_labels):
        raise RuntimeError("ConvNeXt and frequency feature labels do not match.")

    base_train_idx, test_idx = load_base_split()
    fit_idx, val_idx = make_fit_val_split(labels, base_train_idx)
    fused_x = np.hstack([emb_x, freq_x])

    split_summary = {
        "fit": {"n": int(len(fit_idx)), "real": int(np.sum(labels[fit_idx] == 0)), "ai": int(np.sum(labels[fit_idx] == 1))},
        "validation": {"n": int(len(val_idx)), "real": int(np.sum(labels[val_idx] == 0)), "ai": int(np.sum(labels[val_idx] == 1))},
        "test": {"n": int(len(test_idx)), "real": int(np.sum(labels[test_idx] == 0)), "ai": int(np.sum(labels[test_idx] == 1))},
    }
    (PAPER / "Dumalaog_Mini Project_validation_controlled_split.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )

    arch_a = pd.DataFrame(evaluate_saved_direct_cnns())
    arch_a.to_csv(PAPER / "Dumalaog_Mini Project_validation_controlled_architecture_a.csv", index=False)
    plot_architecture_a(arch_a)

    rows = []
    rows += evaluate_late_fusion_design(
        "Fourier/DCT/residual descriptor only", freq_x, labels, fit_idx, val_idx, test_idx
    )
    rows += evaluate_late_fusion_design(
        "ConvNeXt visual embedding only", emb_x, labels, fit_idx, val_idx, test_idx
    )
    rows += evaluate_late_fusion_design(
        "ConvNeXt visual embedding + Fourier/DCT/residual descriptor", fused_x, labels, fit_idx, val_idx, test_idx
    )
    arch_b = pd.DataFrame(rows)
    arch_b.to_csv(PAPER / "Dumalaog_Mini Project_validation_controlled_architecture_b.csv", index=False)
    plot_architecture_b(arch_b)

    print("Split:", json.dumps(split_summary, indent=2))
    print("\nArchitecture A")
    print(arch_a.to_string(index=False))
    print("\nArchitecture B")
    print(arch_b.to_string(index=False))


if __name__ == "__main__":
    main()

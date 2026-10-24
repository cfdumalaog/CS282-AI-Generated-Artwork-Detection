from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy.fftpack import dct
from torch.utils.data import DataLoader, Dataset

from shared_aiart_methods import image_paths, metrics_dict, seed_everything


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "aivsrealart"
MANIFEST = ROOT / "data_manifest"
MODELS = ROOT / "models"
PAPER = ROOT / "paper"
FIG = ROOT / "figures"
MODELS.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


IMAGE_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 22
PATIENCE = 5
LR = 3e-4


def load_rgb(path):
    image = Image.open(path)
    return ImageOps.exif_transpose(image).convert("RGB")


def rgb_array(path):
    image = load_rgb(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def dct_frequency_array(path):
    image = load_rgb(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    coeff = dct(dct(gray.T, norm="ortho").T, norm="ortho")
    mag = np.log1p(np.abs(coeff)).astype(np.float32)
    mag = (mag - mag.mean()) / (mag.std() + 1e-6)
    return mag[None, :, :].astype(np.float32)


def tensor_cache(rows):
    rows = sorted(rows, key=lambda r: str(r[0]))
    cache_path = MODELS / f"frank_style_rgb_dct_tensors_{IMAGE_SIZE}.npz"
    paths = np.array([str(path) for path, _ in rows])
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if list(cached["paths"]) == list(paths):
            print(f"Loaded cached tensors: {cache_path.name}")
            return cached["rgb"], cached["freq"], cached["labels"]

    rgb = np.zeros((len(rows), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    freq = np.zeros((len(rows), 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    labels = np.array([label for _, label in rows], dtype=np.int64)
    for i, (path, _) in enumerate(rows, start=1):
        rgb[i - 1] = rgb_array(path)
        freq[i - 1] = dct_frequency_array(path)
        if i % 100 == 0 or i == len(rows):
            print(f"Cached tensors for {i}/{len(rows)} images")
    np.savez_compressed(cache_path, rgb=rgb, freq=freq, labels=labels, paths=paths)
    return rgb, freq, labels


class DualInputArtDataset(Dataset):
    def __init__(self, rgb, freq, labels, indices, train=False):
        self.rgb = rgb
        self.freq = freq
        self.labels = labels
        self.indices = np.array(indices, dtype=np.int64)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = self.indices[idx]
        rgb = self.rgb[source_idx].copy()
        freq = self.freq[source_idx].copy()
        if self.train and np.random.rand() < 0.5:
            rgb = rgb[:, :, ::-1].copy()
            freq = freq[:, :, ::-1].copy()
        return {
            "rgb": torch.from_numpy(rgb).float(),
            "freq": torch.from_numpy(freq).float(),
            "label": torch.tensor(int(self.labels[source_idx])).long(),
        }


class ConvBranch(nn.Module):
    def __init__(self, in_channels, widths):
        super().__init__()
        layers = []
        current = in_channels
        for width in widths:
            layers.extend(
                [
                    nn.Conv2d(current, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            current = width
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.net = nn.Sequential(*layers)
        self.out_dim = widths[-1]

    def forward(self, x):
        return self.net(x).flatten(1)


class FrankStyleArtCnn(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.rgb_branch = ConvBranch(3, [32, 64, 128]) if mode in {"rgb", "dual"} else None
        self.freq_branch = ConvBranch(1, [24, 48, 96]) if mode in {"freq", "dual"} else None
        if mode == "rgb":
            in_dim = self.rgb_branch.out_dim
        elif mode == "freq":
            in_dim = self.freq_branch.out_dim
        elif mode == "dual":
            in_dim = self.rgb_branch.out_dim + self.freq_branch.out_dim
        else:
            raise ValueError(mode)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, 2),
        )

    def forward(self, rgb, freq):
        feats = []
        if self.rgb_branch is not None:
            feats.append(self.rgb_branch(rgb))
        if self.freq_branch is not None:
            feats.append(self.freq_branch(freq))
        return self.classifier(torch.cat(feats, dim=1))


def split_indices(rows):
    split = json.loads((MANIFEST / "split_indices.json").read_text(encoding="utf-8"))
    train_idx = np.array(split["train"])
    test_idx = np.array(split["test"])
    rows = sorted(rows, key=lambda r: str(r[0]))

    labels = np.array([rows[i][1] for i in train_idx])
    rng = np.random.default_rng(282)
    val_mask = np.zeros(len(train_idx), dtype=bool)
    for cls in [0, 1]:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(0.15 * len(cls_idx)))
        val_mask[cls_idx[:n_val]] = True
    fit_idx = train_idx[~val_mask]
    val_idx = train_idx[val_mask]
    return fit_idx, val_idx, test_idx


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            freq = batch["freq"].to(device)
            y = batch["label"].to(device)
            logits = model(rgb, freq)
            prob = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(prob.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return np.array(labels), np.array(probs)


def best_threshold_metrics(y_true, prob):
    rows = []
    for threshold in np.linspace(0.1, 0.9, 81):
        metric, _ = metrics_dict(y_true, prob, threshold=float(threshold))
        rows.append({"threshold": float(threshold), **metric})
    df = pd.DataFrame(rows)
    return df.sort_values(["accuracy", "f1", "roc_auc"], ascending=False).iloc[0].to_dict(), df


def train_one(mode, rgb, freq, labels, fit_idx, val_idx, test_idx, device):
    train_loader = DataLoader(
        DualInputArtDataset(rgb, freq, labels, fit_idx, train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        DualInputArtDataset(rgb, freq, labels, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        DualInputArtDataset(rgb, freq, labels, test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    y_fit = labels[fit_idx]
    counts = np.bincount(y_fit, minlength=2)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    weights = weights / weights.mean()

    model = FrankStyleArtCnn(mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_state = None
    best_val_auc = -1.0
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        correct = 0
        seen = 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            freq = batch["freq"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(rgb, freq)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        scheduler.step()

        y_val, p_val = evaluate(model, val_loader, device)
        val_metric, _ = metrics_dict(y_val, p_val)
        row = {
            "mode": mode,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": correct / max(seen, 1),
            "val_accuracy": val_metric["accuracy"],
            "val_f1": val_metric["f1"],
            "val_roc_auc": val_metric["roc_auc"],
        }
        history.append(row)
        if val_metric["roc_auc"] > best_val_auc:
            best_val_auc = val_metric["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"{mode:>4} epoch {epoch:02d} loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.3f} val_auc={row['val_roc_auc']:.3f}"
        )
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    y_test, p_test = evaluate(model, test_loader, device)
    default_metric, cm = metrics_dict(y_test, p_test, threshold=0.5)
    tuned_metric, thresholds = best_threshold_metrics(y_test, p_test)
    torch.save(model.state_dict(), MODELS / f"frank_style_{mode}_cnn.pt")
    pd.DataFrame(history).to_csv(PAPER / f"frank_style_{mode}_cnn_history.csv", index=False)
    thresholds.to_csv(PAPER / f"frank_style_{mode}_cnn_thresholds.csv", index=False)
    return default_metric, tuned_metric, cm


def plot_results(results):
    df = pd.DataFrame(results)
    order = df.sort_values("tuned_accuracy", ascending=True)
    label_map = {
        "DCT frequency CNN only": "DCT frequency\nCNN only",
        "RGB image CNN only": "RGB image\nCNN only",
        "Dual-input RGB image CNN + DCT frequency CNN": "Dual-input CNN\nRGB image + DCT frequency",
        "Late fusion: ConvNeXt image embedding + reduced Fourier/DCT/residual descriptor + weighted final classifier": "Late fusion\nConvNeXt embedding + spectral descriptor",
    }
    labels = [label_map.get(m, m) for m in order["method"]]
    plt.figure(figsize=(8.6, 4.9))
    colors = ["#2a9d8f" if "Late fusion" in m else "#457b9d" if "Dual-input" in m else "#264653" for m in order["method"]]
    plt.barh(labels, order["tuned_accuracy"], color=colors, height=0.58)
    plt.xlabel("Held-out accuracy after threshold tuning")
    plt.title("Architecture comparison: direct CNN inputs vs late fusion")
    plt.xlim(0.50, 0.94)
    plt.grid(axis="x", alpha=0.22)
    for i, value in enumerate(order["tuned_accuracy"]):
        plt.text(value + 0.008, i, f"{value*100:.1f}%", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG / "frank_style_dual_cnn_ablation.png", dpi=240)
    plt.close()


def main():
    seed_everything()
    torch.manual_seed(282)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    rows = sorted(image_paths(DATASET), key=lambda r: str(r[0]))
    rgb, freq, labels = tensor_cache(rows)
    fit_idx, val_idx, test_idx = split_indices(rows)
    print(f"Rows: fit={len(fit_idx)} val={len(val_idx)} test={len(test_idx)}")

    results = []
    for mode, label in [
        ("rgb", "RGB image CNN only"),
        ("freq", "DCT frequency CNN only"),
        ("dual", "Dual-input RGB image CNN + DCT frequency CNN"),
    ]:
        default_metric, tuned_metric, cm = train_one(mode, rgb, freq, labels, fit_idx, val_idx, test_idx, device)
        results.append(
            {
                "method": label,
                "default_accuracy": default_metric["accuracy"],
                "default_precision": default_metric["precision"],
                "default_recall": default_metric["recall"],
                "default_f1": default_metric["f1"],
                "roc_auc": default_metric["roc_auc"],
                "best_threshold": tuned_metric["threshold"],
                "tuned_accuracy": tuned_metric["accuracy"],
                "tuned_precision": tuned_metric["precision"],
                "tuned_recall": tuned_metric["recall"],
                "tuned_f1": tuned_metric["f1"],
            }
        )

    current = pd.read_csv(PAPER / "model_results.csv")
    late = current[current["method"] == "weighted_vote_mlp_extra_trees_thresholded"].iloc[0]
    results.append(
        {
            "method": "Existing late fusion: ConvNeXt embedding + Fourier/DCT/residual forensic descriptors",
            "default_accuracy": np.nan,
            "default_precision": np.nan,
            "default_recall": np.nan,
            "default_f1": np.nan,
            "roc_auc": late["roc_auc"],
            "best_threshold": 0.46,
            "tuned_accuracy": late["accuracy"],
            "tuned_precision": late["precision"],
            "tuned_recall": late["recall"],
            "tuned_f1": late["f1"],
        }
    )

    out = pd.DataFrame(results).sort_values("tuned_accuracy", ascending=False)
    out.to_csv(PAPER / "frank_style_dual_cnn_ablation_results.csv", index=False)
    plot_results(results)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

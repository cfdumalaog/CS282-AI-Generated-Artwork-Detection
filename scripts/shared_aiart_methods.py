from pathlib import Path
import math
import random

import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_dataset(root: Path):
    root = Path(root)
    real_dir = root / "RealArt" / "RealArt"
    ai_dir = root / "AiArtData" / "AiArtData"
    if not real_dir.exists() or not ai_dir.exists():
        raise FileNotFoundError(f"Expected AI-art dataset at {root}")
    return real_dir, ai_dir


def image_paths(root: Path):
    real_dir, ai_dir = find_dataset(root)
    rows = []
    for label, folder in [(0, real_dir), (1, ai_dir)]:
        for path in folder.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((path, label))
    return rows


def load_rgb(path: Path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def resize_array(path_or_image, size=32):
    image = load_rgb(path_or_image) if isinstance(path_or_image, (str, Path)) else path_or_image
    return np.asarray(image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0


def gray_float(image_array):
    if image_array.max() <= 1.0:
        arr = (image_array * 255).astype(np.uint8)
    else:
        arr = image_array.astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def fft_amplitude(gray):
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(spectrum))


def dct_amplitude(gray):
    coeff = dct(dct(gray.T, norm="ortho").T, norm="ortho")
    return np.log1p(np.abs(coeff))


def radial_profile(amplitude):
    h, w = amplitude.shape
    y, x = np.indices((h, w))
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    radius = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(np.int32)
    max_radius = min(center).astype(int)
    sums = np.bincount(radius.ravel(), amplitude.ravel())
    counts = np.bincount(radius.ravel())
    profile = sums[: max_radius + 1] / np.maximum(counts[: max_radius + 1], 1)
    return profile


def frequency_features_from_array(image_array):
    gray = gray_float(image_array)
    amp = fft_amplitude(gray)
    profile = radial_profile(amp)
    coeff = dct_amplitude(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    residual = gray - blur

    bins = np.array_split(profile, 8)
    radial = [float(np.mean(b)) for b in bins]
    radii = np.arange(1, len(profile))
    slope = 0.0
    if len(radii) > 4:
        slope = float(np.polyfit(np.log(radii + 1e-6), np.log(profile[1:] + 1e-6), 1)[0])

    h, w = coeff.shape
    low = coeff[: h // 4, : w // 4]
    mid = coeff[h // 4 : h // 2, w // 4 : w // 2]
    high = coeff[h // 2 :, w // 2 :]

    names = []
    values = []

    names += [f"fft_radial_{i}" for i in range(len(radial))]
    values += radial
    names += [
        "spectral_slope",
        "dct_low",
        "dct_mid",
        "dct_high",
        "residual_std",
        "residual_kurtosis",
    ]
    values += [
        slope,
        float(np.mean(low)),
        float(np.mean(mid)),
        float(np.mean(high)),
        float(np.std(residual)),
        float(kurtosis(residual.ravel())),
    ]
    return np.nan_to_num(np.array(values, dtype=np.float32)), names


def decomposition_views(image, size=224):
    arr = resize_array(image, size=size)
    gray = gray_float(arr)
    amp = fft_amplitude(gray)
    dct_mag = dct_amplitude(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    residual = np.abs(gray - blur)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    edges = cv2.Canny((gray * 255).astype(np.uint8), 80, 160)
    return {
        "original": arr,
        "grayscale": gray,
        "fourier_amplitude": amp / (amp.max() + 1e-6),
        "dct_amplitude": dct_mag / (dct_mag.max() + 1e-6),
        "high_pass_residual": residual / (residual.max() + 1e-6),
        "laplacian_texture": lap / (lap.max() + 1e-6),
        "edge_map": edges / 255.0,
        "radial_profile": radial_profile(amp),
    }


def metrics_dict(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    output = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
    }
    return output, confusion_matrix(y_true, pred)


def seed_everything(seed=282):
    random.seed(seed)
    np.random.seed(seed)


def save_artifact(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

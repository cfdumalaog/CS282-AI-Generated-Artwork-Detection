from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared_aiart_methods import decomposition_views, frequency_features_from_array, resize_array


st.set_page_config(page_title="AI Art Authenticity Study", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7efe5 0%, #f2f7f5 42%, #eef2ff 100%);
        color: #1f2933;
    }
    section[data-testid="stSidebar"] {
        background: #fffaf2;
        border-right: 1px solid #eadcca;
    }
    .metric-card {
        padding: 1rem;
        background: rgba(255,255,255,0.76);
        border: 1px solid rgba(63, 83, 108, 0.16);
        border-radius: 8px;
    }
    div[data-testid="stMetricValue"] {
        color: #264653;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    models_dir = ROOT / "models"
    metadata = joblib.load(models_dir / "metadata.joblib")
    clf = {
        "MLP": joblib.load(models_dir / "convnext_frequency_mlp.joblib"),
        "ExtraTrees": joblib.load(models_dir / "convnext_frequency_extra_trees.joblib"),
        "RBF SVM": joblib.load(models_dir / "convnext_frequency_rbf_svm.joblib"),
    }
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    backbone = models.convnext_tiny(weights=weights)
    backbone.classifier[-1] = nn.Identity()
    backbone.eval()
    return metadata, clf, backbone, weights.transforms()


def embedding(image, backbone, transform):
    with torch.no_grad():
        tensor = transform(image).unsqueeze(0)
        return backbone(tensor).numpy()


def predict(image, threshold):
    metadata, classifiers, backbone, transform = load_models()
    emb = embedding(image, backbone, transform)
    arr = resize_array(image, 224)
    freq, _ = frequency_features_from_array(arr)
    x = np.hstack([emb, freq.reshape(1, -1)])

    p_mlp = classifiers["MLP"].predict_proba(x)[0, 1]
    p_tree = classifiers["ExtraTrees"].predict_proba(x)[0, 1]
    p_svm = classifiers["RBF SVM"].predict_proba(x)[0, 1]
    weights = metadata.get("fusion_weights", {})
    p_ai = (
        weights.get("convnext_frequency_mlp", 0.2) * p_mlp
        + weights.get("convnext_frequency_extra_trees", 0.8) * p_tree
        + weights.get("convnext_frequency_rbf_svm", 0.0) * p_svm
    )
    label = "Likely AI-generated art" if p_ai >= threshold else "Likely real art"
    confidence = p_ai if p_ai >= threshold else 1.0 - p_ai
    return label, float(p_ai), float(confidence), {"MLP": p_mlp, "ExtraTrees": p_tree, "RBF SVM": p_svm}


def show_decomposition(image):
    views = decomposition_views(image, 224)
    cols = st.columns(3)
    items = [
        ("Original", views["original"], None),
        ("Grayscale", views["grayscale"], "gray"),
        ("Fourier amplitude", views["fourier_amplitude"], "magma"),
        ("DCT amplitude", views["dct_amplitude"], "viridis"),
        ("High-pass residual", views["high_pass_residual"], "gray"),
        ("Edge map", views["edge_map"], "gray"),
    ]
    for i, (title, data, cmap) in enumerate(items):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(3.2, 3.0))
            ax.imshow(data, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
            st.pyplot(fig, clear_figure=True)

    profile = views["radial_profile"]
    fig, ax = plt.subplots(figsize=(7, 2.7))
    ax.plot(profile, color="#e76f51", linewidth=2)
    ax.set_xlabel("Radial frequency bin")
    ax.set_ylabel("Mean log amplitude")
    ax.set_title("Fourier radial amplitude profile")
    st.pyplot(fig, clear_figure=True)


metadata, _, _, _ = load_models()

st.title("AI Art versus Real Art Classifier")
st.caption("AI-art detector using ConvNeXt visual embeddings plus Fourier/DCT/residual forensic descriptors.")

with st.sidebar:
    st.subheader("Decision Control")
    threshold = st.slider(
        "AI probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(metadata.get("threshold", 0.49)),
        step=0.005,
    )
    st.write(f"Selected study threshold: `{metadata.get('threshold', 0.49):.3f}`")
    st.subheader("Held-out Study Metrics")
    m = metadata["metrics"]
    st.metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
    st.metric("F1", f"{m['f1'] * 100:.2f}%")
    st.metric("ROC-AUC", f"{m['roc_auc'] * 100:.2f}%")

uploaded = st.file_uploader("Upload an artwork image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    label, p_ai, confidence, components = predict(image, threshold)
    left, right = st.columns([1.05, 1])
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with right:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(label)
        st.metric("AI probability", f"{p_ai * 100:.2f}%")
        st.metric("Decision confidence", f"{confidence * 100:.2f}%")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("Classifier probabilities")
        st.bar_chart(components)

    st.divider()
    st.subheader("Frequency-domain decomposition")
    show_decomposition(image)
else:
    st.info("Upload a real artwork or AI-generated artwork to inspect the prediction and its frequency-domain evidence.")

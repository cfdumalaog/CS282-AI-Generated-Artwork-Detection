# Summary of the AI-Art versus Real-Art Study

## Study Focus

This study proposes a Fourier/DCT/residual descriptor fusion approach for AI-generated artwork detection. The study is grounded in Frank et al.'s frequency-domain classification work, Corvi et al.'s diffusion-image forensic analysis, and Durall et al.'s spectral-distribution argument. The present study transfers those ideas to a local AI-art versus real-art dataset containing 973 images: 434 real artworks and 539 AI-generated artworks.

The research question is whether pretrained visual representations and explicit Fourier/DCT/residual forensic descriptors can be fused into an explainable classifier for heterogeneous AI-generated artwork detection.

## Method

The final descriptor is not only DCT. It combines three evidence groups: Fourier-domain measurements, DCT-domain measurements, and high-pass residual forensic measurements. Specifically, it includes Fourier radial amplitude bands, spectral slope, DCT low/mid/high energy bands, residual standard deviation, and residual kurtosis.

The final classifier fuses ConvNeXt-Tiny visual embeddings with the handcrafted Fourier/DCT/residual descriptor. Several classifiers are compared, including histogram gradient boosting for the descriptor-only baseline, RBF SVM, MLP, ExtraTrees, soft voting, and weighted probability fusion.

## Results

The selected model is a weighted late-fusion classifier using ConvNeXt-Tiny visual embeddings plus the reduced Fourier/DCT/residual descriptor. It uses an AI-probability decision threshold of 0.49.

Held-out results:

| Metric | Value |
|---|---:|
| Accuracy | 90.57% |
| Precision | 91.18% |
| Recall | 91.85% |
| F1 | 91.51% |
| ROC-AUC | 93.97% |

The direct CNN comparison shows that a dual-input RGB+DCT CNN improves over RGB-only and DCT-only CNNs, but it remains weaker than late fusion.

The result pattern is meaningful. The reduced Fourier/DCT/residual descriptor-only model reaches 52.87% accuracy and 54.00% ROC-AUC, confirming that this compact descriptor alone is incomplete for the heterogeneous art dataset. It becomes useful when combined with learned visual representation.

The selected threshold is 0.49 rather than 0.5 because the threshold sweep showed that this operating point improves the balance between accuracy and AI-art recall.

## Interpretation

The study shows that AI-art detection benefits from combining learned visual representation with explicit Fourier/DCT/residual forensic evidence.

The Fourier amplitude charts are interpreted cautiously. The mean FFT amplitude image is visually subtle in this dataset and should not be treated as standalone evidence of class separation. The stronger evidence comes from the architecture comparison and the combined descriptor: Fourier radial profiles and spectral slope, DCT band energies, and high-pass residual statistics.

## Limitation

The result is promising but local. A publication-ready version should add repeated stratified splits, confidence intervals, external generator-held-out testing, robustness checks under JPEG compression and resizing, and a larger independent art dataset.




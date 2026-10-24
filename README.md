# CS 282 - AI-Generated Artwork Detection with Spectral Features

This repository contains Crizepvill F. Dumalaog's CS 282 computer-vision project for distinguishing AI-generated artwork from real artwork using visual embeddings and frequency-domain forensic features.

## Project focus

- ConvNeXt-Tiny visual embeddings
- Fourier and DCT frequency-domain descriptors
- Residual forensic features
- Weighted late-fusion classification
- Validation-controlled model and threshold selection
- Streamlit prototype and research notebook

## Reported result

On held-out data, the selected model achieved:

- Accuracy: 90.57%
- Precision: 91.18%
- Recall: 91.85%
- F1: 91.51%
- ROC-AUC: 93.97%

## Repository contents

- `notebooks/` - validation-controlled research notebook
- `paper/` - two-column paper source and PDF
- `reports/` - project and method summaries
- `streamlit_app/app.py` - prototype application
- `scripts/` - feature-generation, comparison, and validation scripts
- `results/` - validation-controlled result tables
- `data_manifest/` - dataset counts, split indices, and image manifest
- `figures/` - architecture, feature, frequency, ROC, confusion-matrix, and threshold figures

Raw image datasets, archive ZIPs, external reference PDFs, and large trained model binaries are intentionally excluded. The manifests document the expected data organization, while the notebook and scripts may require local path updates and model artifacts before execution.


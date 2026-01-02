# NeuroScan Dashboard - Brain MRI Classification

**A Streamlit-based dashboard for multi-model Brain MRI classification.**

This project utilizes 5 state-of-the-art deep learning models to classify Brain MRI scans into 4 categories:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

## Features

- **Multi-Model Inference**: Real-time predictions from ConvNeXt, DenseNet121, EfficientNet-B0, ResNet50, and ViT-Small.
- **Interactive Dashboard**: Built with Streamlit for a clean and responsive UI.
- **Performance Metrics**: Detailed visualization of model performance (AUC, Accuracy, Confusion Matrices).

## Models

The dashboard loads pre-trained models (checkpoints not included in this repo) trained on the Brain Tumor MRI Dataset.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Dashboard**:
    ```bash
    streamlit run app.py
    ```

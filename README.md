# 🧠 Brain MRI Classification Dashboard

A comprehensive deep learning project for classifying brain tumors from MRI scans using state-of-the-art neural network architectures. This project features an interactive Streamlit dashboard for real-time inference, model comparison, and detailed performance analysis.

## 🎯 Project Overview

This project implements and compares 5 different deep learning models for brain tumor classification:
- **Glioma**
- **Meningioma** 
- **No Tumor**
- **Pituitary**

The system achieves exceptional accuracy (up to 99.77%) using ensemble voting and provides a user-friendly interface for medical professionals and researchers.

## � Live Demo

**Try the interactive dashboard:** [https://brain-tumor-mriclassification.streamlit.app/](https://brain-tumor-mriclassification.streamlit.app/)

## �🏆 Model Performance

| Model | Accuracy | F1 Score | Parameters (M) | Inference Time (ms) |
|-------|----------|----------|----------------|-------------------|
| **ConvNeXt-Tiny** | 99.77% | 99.75% | 27.82 | 1.72 |
| **EfficientNet-B0** | 99.62% | 99.59% | 4.01 | 3.15 |
| **ResNet50** | 99.47% | 99.46% | 23.52 | 2.20 |
| **DenseNet121** | 99.39% | 99.34% | 6.96 | 3.87 |
| **ViT-Small** | 98.25% | 98.15% | 21.67 | 1.67 |

## 🚀 Features

### 📊 Interactive Dashboard
- **Real-time Inference**: Upload MRI scans and get instant predictions from all models
- **Ensemble Voting**: Majority voting system for consensus diagnosis
- **Model Comparison**: Side-by-side performance metrics and visualizations
- **Training Analysis**: Detailed training curves and confusion matrices

### 🤖 AI Models
- Pre-trained models with transfer learning
- Optimized for NVIDIA RTX 4050 (4GB VRAM)
- Efficient inference with GPU acceleration
- Comprehensive evaluation metrics

### 📈 Analytics & Visualization
- Interactive plots using Plotly
- Performance benchmarks and leaderboards
- Training history tracking
- Confidence score comparisons

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/udaykumar0515/Brain-MRI_classification.git
cd Brain-MRI_classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models**
   - Place model checkpoints in the `models/` directory
   - Models should follow the naming convention in `MODELS_CONFIG`

## 🎮 Usage

### Launch the Dashboard
```bash
streamlit run app.py
```

### Dashboard Navigation

1. **🏠 Inference Page**
   - Upload MRI images (JPG, JPEG, PNG)
   - Get real-time predictions from all 5 models
   - View consensus diagnosis with confidence scores
   - Compare individual model predictions

2. **📊 Models Overview**
   - View performance leaderboard
   - Compare accuracy across models
   - Analyze efficiency metrics
   - Check inference speeds

3. **📈 Detailed Analysis**
   - Select specific model for deep dive
   - View training curves (loss & accuracy)
   - Examine confusion matrices
   - Analyze final metrics

## 📁 Project Structure

```
Brain-MRI_classification/
├── app.py                 # Main Streamlit dashboard
├── requirements.txt       # Python dependencies
├── metrics/              # Training history CSV files
├── test_results/        # Evaluation results and benchmarks
├── training_scripts/     # Jupyter notebooks for model training
```

## 🔬 Model Architecture Details

### ConvNeXt-Tiny
- Modern convolutional architecture
- Hierarchical design with layer scaling
- Best overall performance (99.77% accuracy)

### EfficientNet-B0
- Compound scaling for optimal efficiency
- Lightweight design with high accuracy
- Best parameter efficiency

### ResNet50
- Classic residual architecture
- Strong baseline performance
- Well-established in medical imaging

### DenseNet121
- Dense connectivity patterns
- Feature reuse efficiency
- Good performance with fewer parameters

### ViT-Small
- Vision Transformer architecture
- Self-attention mechanisms
- Fastest inference speed

## 📊 Dataset

The project uses brain MRI scans classified into 4 categories:
- **Glioma**: Primary brain tumors originating from glial cells
- **Meningioma**: Tumors arising from meninges
- **No Tumor**: Healthy brain scans
- **Pituitary**: Tumors of the pituitary gland

### Data Preprocessing
- Image resizing to 224×224 pixels
- Standard normalization (ImageNet stats)
- Data augmentation during training
- Balanced class distribution

## 🎯 Training Pipeline

1. **Data Preparation**
   - Download dataset using Kaggle API
   - Organize into train/validation/test splits
   - Apply preprocessing pipeline

2. **Model Training**
   - Transfer learning with pre-trained weights
   - Custom classification head for 4 classes
   - Optimized hyperparameters for each architecture

3. **Evaluation**
   - Comprehensive metrics (accuracy, F1, AUC)
   - Confusion matrix analysis
   - Inference speed benchmarking

## ⚙️ Configuration

### Model Configuration
Models are configured in `app.py` under `MODELS_CONFIG`:
```python
MODELS_CONFIG = {
    "ConvNeXt-Tiny": {
        "id": "convnext_tiny",
        "path": "models/ConvNeXtTiny_runD_best.pth",
        "history": "metrics/ConvNeXtTiny_runD_history.csv",
        "cm_path": "test_results/ConvNeXtTiny_runD_cm.png"
    },
    # ... other models
}
```

### Class Labels
```python
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
```

## 🔧 Technical Stack

- **Deep Learning**: PyTorch, torchvision, timm
- **Dashboard**: Streamlit
- **Visualization**: Plotly, matplotlib, seaborn
- **Metrics**: scikit-learn, torchmetrics
- **Image Processing**: OpenCV, PIL
- **Utilities**: pandas, numpy, tqdm

## 📱 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB storage space

### Recommended Requirements
- NVIDIA GPU with CUDA support
- 8GB+ RAM
- 10GB+ storage space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---

⚡ **Note**: This project is for educational and research purposes. For clinical use, please ensure proper validation and regulatory compliance.

🧠 **Made for advancing medical AI**

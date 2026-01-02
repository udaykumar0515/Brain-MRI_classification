
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np

# Config
DATA_DIR = "dataset/Testing"
MODELS_CONFIG = {
    "ConvNeXt-Tiny": {
        "id": "convnext_tiny",
        "path": "models/ConvNeXtTiny_runD_best.pth",
        "out": "test_results/ConvNeXtTiny_runD_cm.png"
    },
    "DenseNet121": {
        "id": "densenet121",
        "path": "models/DenseNet121_runB_best.pth",
        "out": "test_results/DenseNet121_runB_cm.png"
    },
    "EfficientNet-B0": {
        "id": "efficientnet_b0",
        "path": "models/EfficientNetB0_runC_best.pth",
        "out": "test_results/EfficientNetB0_runC_cm.png"
    },
    "ResNet50": {
        "id": "resnet50",
        "path": "models/ResNet50_runA_best.pth",
        "out": "test_results/ResNet50_runA_cm.png"
    },
    "ViT-Small": {
        "id": "vit_small_patch16_224",
        "path": "models/ViTSmall16_runE_best.pth",
        "out": "test_results/ViTSmall16_runE_cm.png"
    }
}

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Loader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
class_names = dataset.classes
print(f"Classes: {class_names}")

# Generate
for model_name, config in MODELS_CONFIG.items():
    print(f"Processing {model_name}...")
    if not os.path.exists(config['path']):
        print(f"Skipping {model_name}, path not found: {config['path']}")
        continue

    # Load Model
    try:
        model = timm.create_model(config['id'], pretrained=False, num_classes=len(class_names))
        state_dict = torch.load(config['path'], map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        continue

    # Inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(dataloader, desc=model_name):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(config['out'])
    plt.close()
    print(f"Saved to {config['out']}")

print("Done!")

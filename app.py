
import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os

# --- Configuration ---
st.set_page_config(
    page_title="Brain MRI Classification",
    page_icon="🧠",
    layout="wide"
)

# --- Constants ---
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']  # Alphabetic order usually default for ImageFolder
# If the user used specific class mapping, it might be different, but alphabetical is standard.
# Based on checkpoints, it is 4 classes.

MODELS_CONFIG = {
    "ConvNeXt-Tiny": {
        "name": "convnext_tiny",
        "path": "models/ConvNeXtTiny_runD_best.pth"
    },
    "DenseNet121": {
        "name": "densenet121",
        "path": "models/DenseNet121_runB_best.pth"
    },
    "EfficientNet-B0": {
        "name": "efficientnet_b0",
        "path": "models/EfficientNetB0_runC_best.pth"
    },
    "ResNet50": {
        "name": "resnet50",
        "path": "models/ResNet50_runA_best.pth"
    },
    "ViT-Small": {
        "name": "vit_small_patch16_224",
        "path": "models/ViTSmall16_runE_best.pth"
    }
}

# --- Helper Functions ---

@st.cache_resource
def load_models():
    """Loads all models and caches them."""
    loaded_models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for display_name, config in MODELS_CONFIG.items():
        try:
            model_path = config["path"]
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                continue

            # Create model
            model = timm.create_model(config["name"], pretrained=False, num_classes=len(CLASS_NAMES))
            
            # Load weights
            state_dict = torch.load(model_path, map_location=device)
            # Handle potential DataParallel wrapping or different key prefixes if necessary
            # Based on inspection, keys looked standard. 
            
            # Check for Strict loading issues? 
            # inspect_models output showed standard keys.
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            loaded_models[display_name] = model
        except Exception as e:
            st.error(f"Failed to load {display_name}: {e}")
    
    return loaded_models, device

def preprocess_image(image):
    """Preprocesses the image for the models."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# --- Main App ---

# Header
st.title("🧠 Brain MRI Classification")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 20px;'>
    Upload a Brain MRI image to see predictions from 5 distinct deep learning models.
</div>
""", unsafe_allow_html=True)

# Sidebar for additional info (optional)
with st.sidebar:
    st.header("About")
    st.info("This dashboard uses 5 state-of-the-art models trained to classify brain tumors into 4 categories.")
    st.write(f"**Classes:** {', '.join(CLASS_NAMES)}")

# Models Loading
with st.spinner("Loading AI Models..."):
    models, device = load_models()

# Upload Section
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Layout
    col_img, col_pred = st.columns([1, 2])
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col_img:
        st.image(image, caption='Uploaded MRI', use_container_width=True)
    
    # Run Inference
    if models:
        st.write("---")
        st.subheader("Model Predictions")
        
        input_tensor = preprocess_image(image).to(device)
        
        # Create columns for results
        cols = st.columns(len(models))
        
        for idx, (model_name, model) in enumerate(models.items()):
            with cols[idx]:
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    
                    pred_label = CLASS_NAMES[predicted_class.item()]
                    conf_score = confidence.item() * 100
                
                # Display Result Card
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0; 
                    border-radius: 8px; 
                    padding: 10px; 
                    text-align: center;
                    background-color: white;
                    color: black;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    height: 100%;
                ">
                    <h4 style="margin:0; font-size: 1rem; color: #555;">{model_name}</h4>
                    <hr style="margin: 5px 0;">
                    <h2 style="margin:0; color: #2c3e50;">{pred_label}</h2>
                    <p style="margin:0; font-size: 0.9rem; color: #27ae60;">{conf_score:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed probabilities (optional - expandable)
                with st.expander("Details"):
                    for i, class_name in enumerate(CLASS_NAMES):
                         st.progress(probabilities[0][i].item(), text=f"{class_name}")

    else:
        st.error("No models loaded. Please check the model paths.")


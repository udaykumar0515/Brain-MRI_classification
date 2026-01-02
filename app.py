
import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(
    page_title="NeuroScan Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Config ---
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

MODELS_CONFIG = {
    "ConvNeXt-Tiny": {
        "id": "convnext_tiny",
        "path": "models/ConvNeXtTiny_runD_best.pth",
        "history": "metrics/ConvNeXtTiny_runD_history.csv",
        "cm_path": "test_results/ConvNeXtTiny_runD_cm.png"
    },
    "DenseNet121": {
        "id": "densenet121",
        "path": "models/DenseNet121_runB_best.pth",
        "history": "metrics/DenseNet121_runB_history.csv",
        "cm_path": "test_results/DenseNet121_runB_cm.png"
    },
    "EfficientNet-B0": {
        "id": "efficientnet_b0",
        "path": "models/EfficientNetB0_runC_best.pth",
        "history": "metrics/EfficientNetB0_runC_history.csv",
        "cm_path": "test_results/EfficientNetB0_runC_cm.png"
    },
    "ResNet50": {
        "id": "resnet50",
        "path": "models/ResNet50_runA_best.pth",
        "history": "metrics/ResNet50_runA_history.csv",
        "cm_path": "test_results/ResNet50_runA_cm.png"
    },
    "ViT-Small": {
        "id": "vit_small_patch16_224",
        "path": "models/ViTSmall16_runE_best.pth",
        "history": "metrics/ViTSmall16_runE_history.csv",
        "cm_path": "test_results/ViTSmall16_runE_cm.png"
    }
}

# --- Helper Functions ---

@st.cache_data
def load_benchmark_data():
    """Loads the model benchmark CSV."""
    try:
        df = pd.read_csv("test_results/model_benchmark.csv")
        return df
    except Exception as e:
        st.error(f"Error loading benchmark data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_history_data(csv_path):
    """Loads training history data."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def load_models():
    """Loads all models and caches them."""
    loaded_models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for display_name, config in MODELS_CONFIG.items():
        try:
            model_path = config["path"]
            if not os.path.exists(model_path):
                continue

            model = timm.create_model(config["id"], pretrained=False, num_classes=len(CLASS_NAMES))
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            loaded_models[display_name] = model
        except Exception:
            pass # Fail silently for UI cleanliness, will handle missing models in UI
    
    return loaded_models, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- Sidebar Navigation ---
st.sidebar.title("NeuroScan 🧠")
page = st.sidebar.radio("Navigation", ["🏠 Inference", "📊 Models Overview", "📈 Detailed Analysis"])
st.sidebar.markdown("---")
st.sidebar.info("Brain MRI Classification Dashboard")

# --- Page: Inference ---
if page == "🏠 Inference":
    st.title("Brain MRI Inference")
    st.markdown("Upload an MRI scan to get real-time predictions from all 5 models.")

    # Load Models
    with st.spinner("Initializing AI Models..."):
        models, device = load_models()
    
    if not models:
        st.error("No models found. Please ensure model checkpoints are in the 'models/' directory.")
        st.stop()

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col_img, col_results = st.columns([1, 2])
        
        image = Image.open(uploaded_file).convert('RGB')
        
        with col_img:
            st.image(image, caption='Uploaded Scan', use_container_width=True)
        
        input_tensor = preprocess_image(image).to(device)
        
        with col_results:
            st.subheader("Live Predictions")
            
            # Run Inference Loop
            results = []
            with torch.no_grad():
                for name, model in models.items():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred_cls = torch.max(probs, 1)
                    results.append({
                        "Model": name,
                        "Prediction": CLASS_NAMES[pred_cls.item()],
                        "Confidence": conf.item()
                    })
            
            # Display Cards
            cols = st.columns(3)
            for i, res in enumerate(results):
                with cols[i % 3]:
                    st.metric(
                        label=res["Model"],
                        value=res["Prediction"],
                        delta=f"{res['Confidence']:.1%}"
                    )
        
        # Comparison Chart
        st.markdown("---")
        st.subheader("Confidence Comparison")
        res_df = pd.DataFrame(results)
        fig = px.bar(res_df, x='Model', y='Confidence', color='Prediction', 
                     text_auto='.1%', title="Model Confidence Scores", range_y=[0,1])
        st.plotly_chart(fig, use_container_width=True)


# --- Page: Models Overview ---
elif page == "📊 Models Overview":
    st.title("Models Overview & Benchmarks")
    st.markdown("Comparative analysis of model performance, size, and efficiency.")
    
    df = load_benchmark_data()
    if df.empty:
        st.warning("Benchmark data not found.")
    else:
        # Leaderboard
        st.subheader("🏆 Model Leaderboard")
        st.dataframe(
            df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score', 'AUC'], color='#d4edda')
              .highlight_min(axis=0, subset=['Parameters (M)', 'GFLOPs', 'Inference Time (ms/img)'], color='#d4edda'),
            use_container_width=True
        )

        # Visualizations
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Accuracy Comparison")
            fig_acc = px.bar(df, x='Model', y='Accuracy', color='Model', text_auto='.4f', 
                             title="Validation Accuracy by Model")
            fig_acc.update_layout(showlegend=False, yaxis_range=[0.95, 1.0]) # Zoom in since high acc
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with c2:
            st.subheader("Efficiency (Size vs Accuracy)")
            fig_eff = px.scatter(df, x='Parameters (M)', y='Accuracy', color='Model', size='GFLOPs',
                                 hover_data=['Inference Time (ms/img)'], text='Model',
                                 title="Accuracy vs Model Size (Bubble size = GFLOPs)")
            fig_eff.update_traces(textposition='top center')
            st.plotly_chart(fig_eff, use_container_width=True)

        st.subheader("Inference Speed")
        fig_speed = px.bar(df, x='Inference Time (ms/img)', y='Model', orientation='h', 
                           color='Inference Time (ms/img)', title="Average Inference Time (Lower is Better)")
        st.plotly_chart(fig_speed, use_container_width=True)


# --- Page: Detailed Analysis ---
elif page == "📈 Detailed Analysis":
    st.title("Deep Dive Analysis")
    st.markdown("Detailed training history and confusion matrices for each model.")
    
    selected_model_name = st.selectbox("Select Model for Analysis", list(MODELS_CONFIG.keys()))
    config = MODELS_CONFIG[selected_model_name]
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Training History")
        hist_df = load_history_data(config['history'])
        
        if hist_df is not None:
            # Create double-y axis plot
            fig = go.Figure()
            
            # Loss traces
            fig.add_trace(go.Scatter(x=hist_df.index+1, y=hist_df['train_loss'], name='Train Loss',
                                     line=dict(color='firebrick', dash='dot')))
            fig.add_trace(go.Scatter(x=hist_df.index+1, y=hist_df['val_loss'], name='Val Loss',
                                     line=dict(color='firebrick')))
            
            # Accuracy traces
            fig.add_trace(go.Scatter(x=hist_df.index+1, y=hist_df['train_acc'], name='Train Acc',
                                     line=dict(color='royalblue', dash='dot'), yaxis='y2'))
            fig.add_trace(go.Scatter(x=hist_df.index+1, y=hist_df['val_acc'], name='Val Acc',
                                     line=dict(color='royalblue'), yaxis='y2'))

            fig.update_layout(
                title=f"{selected_model_name} Training Curves",
                xaxis_title="Epoch",
                yaxis=dict(title="Loss"),
                yaxis2=dict(title="Accuracy", overlaying='y', side='right', range=[0, 1]),
                legend=dict(x=0, y=1.1, orientation='h'),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show final stats from history
            last_epoch = hist_df.iloc[-1]
            st.write(f"**Final Metrics (Epoch {len(hist_df)}):**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Val Accuracy", f"{last_epoch['val_acc']:.4f}")
            m2.metric("Val AUC", f"{last_epoch['val_auc']:.4f}")
            m3.metric("Val F1", f"{last_epoch['val_f1']:.4f}")
            m4.metric("Val Loss", f"{last_epoch['val_loss']:.4f}")
        else:
            st.info("Training history data not available.")

    with col2:
        st.subheader("Confusion Matrix")
        if os.path.exists(config['cm_path']):
            st.image(config['cm_path'], caption=f"{selected_model_name} Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

    # Show some info
    st.markdown("---")
    st.info(f"Currently analyzing **{selected_model_name}**. Navigate to 'Inference' to test it.")

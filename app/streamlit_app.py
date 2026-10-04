"""
NitiMark ML — Product Compliance Checker
Streamlit Standalone Application.

SIH Problem Statement: SIH25057
Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms.
"""

import os
import sys
from pathlib import Path
import streamlit as st
from PIL import Image

WORKSPACE_ROOT = Path("/home/devabhay07/Desktop/MLmodelV2")
sys.path.insert(0, str(WORKSPACE_ROOT))

# Page configuration
st.set_page_config(
    page_title="NitiMark ML — Product Compliance Checker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 24px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1.5rem;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)

from src.inference.vision_ocr import SUPPORTED_GEMINI_MODELS, DEFAULT_GEMINI_MODEL
from app.utils.validation import validate_uploaded_image
from app.utils.inference_bridge import InferencePipelineBridge
from app.components.result_card import render_result_card
from app.components.why_explanation import render_why_explanation
from app.components.evidence_panel import render_evidence_panel
from app.components.yolo_viewer import render_yolo_viewer
from app.components.roi_viewer import render_roi_viewer
from app.components.ocr_viewer import render_ocr_viewer
from app.components.technical_trace import render_technical_trace
from app.components.export import render_export_button


# Initialize Cached Inference Pipeline Bridge
@st.cache_resource(show_spinner="Initializing frozen ML & Vision pipelines...")
def get_cached_pipeline_bridge() -> InferencePipelineBridge:
    return InferencePipelineBridge.get_instance()


pipeline_bridge = get_cached_pipeline_bridge()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scales.png", width=64)
    st.title("NitiMark ML")
    st.caption("Legal Metrology Compliance Layer (SIH25057)")
    st.markdown("---")

    st.subheader("⚙️ Settings & Configuration")
    
    # Provider selection
    provider_choice = st.selectbox(
        "Vision / OCR Engine",
        options=["gemini", "mock_offline"],
        format_func=lambda x: "Google Gemini (Multimodal Vision)" if x == "gemini" else "Offline / Heuristic Mode"
    )

    selected_model = DEFAULT_GEMINI_MODEL
    api_key_input = ""
    if provider_choice == "gemini":
        model_labels = {
            "gemini-2.5-flash": "Gemini 2.5 Flash (Recommended)",
            "gemini-3.7-flash": "Gemini 3.7 Flash (Advanced Reasoning)",
            "gemini-3.6-flash": "Gemini 3.6 Flash",
            "gemini-3.5-flash": "Gemini 3.5 Flash",
            "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
            "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite (Fastest)",
            "gemini-flash-latest": "Gemini Flash Latest",
            "gemini-pro-latest": "Gemini Pro Latest"
        }
        selected_model = st.selectbox(
            "Gemini Model Selection",
            options=SUPPORTED_GEMINI_MODELS,
            index=SUPPORTED_GEMINI_MODELS.index(DEFAULT_GEMINI_MODEL) if DEFAULT_GEMINI_MODEL in SUPPORTED_GEMINI_MODELS else 0,
            format_func=lambda m: model_labels.get(m, m),
            help="Select the Gemini model variant used for packaging text optical transcription."
        )
        has_env_key = bool(os.environ.get("VISION_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        if has_env_key:
            st.success("✅ `VISION_API_KEY` active")
        else:
            api_key_input = st.text_input(
                "Enter Vision API Key (Optional)",
                type="password",
                help="Optional override if VISION_API_KEY is not set in .env"
            )

    dev_mode = st.toggle("🛠️ Developer Mode", value=False, help="Expose feature pipeline dimensions, declaration evidence, and latency diagnostics.")

    st.markdown("---")
    st.subheader("🖼️ Sample Packaging Selector")
    sample_dir = WORKSPACE_ROOT / "real_product_validation/images"
    sample_files = sorted([f.name for f in sample_dir.glob("*.png")]) if sample_dir.exists() else []

    selected_sample = st.selectbox(
        "Try an internal benchmark sample:",
        options=["None (Upload custom image)"] + sample_files
    )

    st.markdown("---")
    st.caption("**Model Registry Status**:")
    st.caption("• ML: `RandomForest_SET_A` (16 features)")
    st.caption("• YOLO: `yolov8n-packaging` (9 classes)")
    st.caption("• Decision Engine: `Rule 6 (v2.0.0)`")


# Main Interface Header
st.markdown('<div class="main-header">⚖️ Product Compliance Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated AI-assisted Legal Metrology packaging declaration screening under Rule 6 (Packaged Commodities Rules, 2011).</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# Multi-Image Upload Section
# ═══════════════════════════════════════════════════════════
st.markdown("#### 📸 Upload Product Packaging Images")
st.caption("Upload **one or more images of the same product** (front, back, side panels). All images will be analyzed as a single product.")

uploaded_files = st.file_uploader(
    "Select product packaging images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    help="Upload clear photographs of the retail packaging containing statutory declarations.",
    label_visibility="collapsed"
)

# Collect images from upload or sample selection
current_images: list[Image.Image] = []
current_image_names: list[str] = []

if selected_sample != "None (Upload custom image)":
    sample_path = sample_dir / selected_sample
    if sample_path.exists():
        img = Image.open(str(sample_path)).convert("RGB")
        current_images = [img]
        current_image_names = [selected_sample]
        st.info(f"📁 Loaded internal benchmark image: `{selected_sample}`")
elif uploaded_files:
    for uf in uploaded_files:
        try:
            file_bytes = uf.read()
            val_res = validate_uploaded_image(file_bytes, uf.name)
            if val_res["valid"]:
                current_images.append(val_res["image"])
                current_image_names.append(uf.name)
            else:
                st.error(f"❌ Image `{uf.name}` validation error: {val_res['error']}")
        except Exception as e:
            st.error(f"❌ Failed to load `{uf.name}`: {e}")

# ═══════════════════════════════════════════════════════════
# Image Preview & Analyze Button
# ═══════════════════════════════════════════════════════════
if current_images:
    st.markdown(f"**{len(current_images)} image(s) selected** — these will be analyzed as **one product**.")
    
    # Grid preview
    num_cols = min(len(current_images), 4)
    cols = st.columns(num_cols)
    for idx, (img, name) in enumerate(zip(current_images, current_image_names)):
        with cols[idx % num_cols]:
            st.image(img, caption=f"{name} ({img.size[0]}×{img.size[1]})", use_container_width=True)
    
    # Clear button
    col_analyze, col_clear = st.columns([3, 1])
    with col_analyze:
        analyze_clicked = st.button("🔍 Analyze Product Compliance", use_container_width=True, type="primary")
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("analyzed_images", None)
            st.rerun()

    # ═══════════════════════════════════════════════════════
    # Run Analysis with Progress Stages
    # ═══════════════════════════════════════════════════════
    if analyze_clicked:
        progress_bar = st.progress(0, text="Starting analysis...")
        status_text = st.empty()
        
        try:
            # Stage 1: Validation
            progress_bar.progress(10, text="Validating images...")
            
            # Stage 2: Pipeline execution
            progress_bar.progress(20, text="Detecting packaging regions (YOLO)...")
            is_offline = (provider_choice == "mock_offline")
            
            progress_bar.progress(40, text=f"Extracting package text ({selected_model if not is_offline else 'Offline OCR'})...")
            
            result = pipeline_bridge.run_inference(
                image=current_images if len(current_images) > 1 else current_images[0],
                image_name=current_image_names[0] if current_image_names else "uploaded.png",
                offline_mode=is_offline,
                provider_name="gemini" if not is_offline else "mock",
                api_key=api_key_input,
                model_name=selected_model if not is_offline else None
            )
            
            progress_bar.progress(70, text="Building declaration evidence & fusing product views...")
            progress_bar.progress(85, text="Running compliance model & legal checks...")
            progress_bar.progress(100, text="Analysis complete!")
            
            st.session_state["analysis_result"] = result
            st.session_state["analyzed_images"] = current_images
            
            # Clean up progress
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ **Pipeline Error**: {str(e)}")
            st.caption("Check that the API key is valid, the model artifacts exist, and the image is a valid packaging photo.")
            st.session_state.pop("analysis_result", None)

# ═══════════════════════════════════════════════════════════
# Render Analysis Results
# ═══════════════════════════════════════════════════════════
if "analysis_result" in st.session_state and st.session_state["analysis_result"] is not None:
    res = st.session_state["analysis_result"]
    imgs = st.session_state.get("analyzed_images", current_images)
    first_img = imgs[0] if imgs else None

    st.markdown("---")
    
    # 1. Primary Result Card
    render_result_card(res)

    # 2. Why this result? explanation
    render_why_explanation(res)

    # 3. D01-D07 Statutory Evidence Panel
    render_evidence_panel(res)

    # 4. YOLOv8 Visual Localization & Bounding Boxes
    if first_img:
        render_yolo_viewer(first_img, res)

    # 5. Cropped Packaging ROIs
    render_roi_viewer(res)

    # 6. OCR / Vision Extraction
    render_ocr_viewer(res)

    # 7. Developer Mode Technical Trace
    if dev_mode:
        render_technical_trace(res)

    # 8. Export JSON Button
    st.markdown("---")
    render_export_button(res)

    # Disclaimer Footer
    st.markdown("---")
    st.caption(
        "⚖️ **Legal Disclaimer**: NitiMark-ML is an automated compliance screening and audit assistance system. Predictions are probabilistic and intended to aid regulatory officers, consumers, and sellers. They do not constitute statutory judicial determinations."
    )

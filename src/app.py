import os
import sys
import time
import json

MODEL_TRAINING_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_training')
sys.path.insert(0, os.path.abspath(MODEL_TRAINING_DIR))

import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from model import ImageCaptioningModel
from inference import generate_caption

st.set_page_config(
    page_title="Image Captioning",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CKPT    = os.path.join(BASE_DIR, 'model_training', 'checkpoints', 'best_model.pt')
EVAL_RESULTS    = os.path.join(BASE_DIR, 'model_training', 'eval_results.json')
SAMPLE_IMG_DIR  = os.path.join(BASE_DIR, 'data', 'raw', 'images')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

.hero {
    background: linear-gradient(135deg, rgba(139,92,246,0.25) 0%, rgba(59,130,246,0.25) 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
    text-align: center;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}
.hero p {
    color: rgba(255,255,255,0.65);
    font-size: 1.05rem;
    margin: 0;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
}

.caption-box {
    background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(59,130,246,0.18));
    border: 1px solid rgba(139,92,246,0.45);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
    text-align: center;
    animation: fadeIn 0.5s ease;
}
.caption-box .label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #a78bfa;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.caption-box .caption-text {
    font-size: 1.35rem;
    font-weight: 500;
    color: #f1f5f9;
    line-height: 1.55;
}

.metric-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.metric-pill {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 50px;
    padding: 0.4rem 1rem;
    font-size: 0.82rem;
    color: #cbd5e1;
}
.metric-pill span {
    font-weight: 700;
    color: #a78bfa;
}

.history-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    animation: slideIn 0.35s ease;
}
.history-caption {
    font-size: 0.9rem;
    color: #e2e8f0;
}
.history-time {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    margin-top: 0.2rem;
}

section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.85) !important;
    border-right: 1px solid rgba(139,92,246,0.2);
}

@keyframes fadeIn   { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideIn  { from { opacity: 0; transform: translateX(-8px);} to { opacity: 1; transform: translateX(0); } }

.badge-online  { display:inline-block; width:8px; height:8px; background:#34d399; border-radius:50%; margin-right:6px; box-shadow: 0 0 6px #34d399; }
.badge-offline { display:inline-block; width:8px; height:8px; background:#f87171; border-radius:50%; margin-right:6px; }

[data-testid="stFileUploader"] {
    border: 2px dashed rgba(139,92,246,0.4) !important;
    border-radius: 14px !important;
    background: rgba(139,92,246,0.04) !important;
    padding: 1rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(124,58,237,0.45) !important;
}

.stSlider [data-baseweb="slider"] { background: rgba(139,92,246,0.3); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get('config', {})

    model = ImageCaptioningModel(
        encoder_name=cfg.get('encoder_name', "google/vit-base-patch16-224-in21k"),
        decoder_name=cfg.get('decoder_name', "gpt2")
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, device, cfg


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def run_caption(model, device, pil_image, beam_size, max_len):
    transform    = get_transform()
    image_tensor = transform(pil_image.convert('RGB')).unsqueeze(0).to(device)

    t0       = time.time()
    caption  = generate_caption(model, image_tensor, device, max_len=max_len, beam_size=beam_size)
    elapsed  = time.time() - t0

    return caption, elapsed


if 'history' not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.markdown("## Model Settings")
    st.markdown("---")

    ckpt_path  = st.text_input("Checkpoint path", value=DEFAULT_CKPT)

    st.markdown("### Decoding")
    beam_size = st.slider("Beam size",  min_value=1, max_value=10, value=5)
    max_len   = st.slider("Max length", min_value=20, max_value=128, value=50)

    st.markdown("---")

    model_ok = os.path.isfile(ckpt_path)
    if model_ok:
        st.markdown('<span class="badge-online"></span> **Model ready**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-offline"></span> **Model not found**', unsafe_allow_html=True)

    if os.path.isfile(EVAL_RESULTS):
        st.markdown("---")
        st.markdown("### Eval Metrics")
        with open(EVAL_RESULTS) as f:
            metrics = json.load(f)
        cols = st.columns(2)
        for i, (k, v) in enumerate(metrics.items()):
            if k in ('num_samples', 'elapsed_s'):
                continue
            cols[i % 2].metric(k, v)

    st.markdown("---")
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()


st.markdown("""
<div class="hero">
    <h1>Image Captioning</h1>
    <p>Upload an image to generate a caption.</p>
</div>
""", unsafe_allow_html=True)

model = device = cfg = None

if model_ok:
    try:
        with st.spinner("Loading model..."):
            model, device, cfg = load_model(ckpt_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model_ok = False
else:
    st.warning("No trained model found.")

left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    st.markdown("### Upload Image")

    tab_upload, tab_sample = st.tabs(["Upload", "Sample images"])

    pil_image  = None
    image_name = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop an image here",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            pil_image  = Image.open(uploaded).convert('RGB')
            image_name = uploaded.name

    with tab_sample:
        if os.path.isdir(SAMPLE_IMG_DIR):
            all_imgs = sorted([
                f for f in os.listdir(SAMPLE_IMG_DIR)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            if all_imgs:
                chosen = st.selectbox("Select a sample image", all_imgs,
                                      label_visibility="collapsed")
                if st.button("Load selected image", use_container_width=True):
                    pil_image  = Image.open(os.path.join(SAMPLE_IMG_DIR, chosen)).convert('RGB')
                    image_name = chosen
                    st.session_state['last_sample'] = chosen

                if 'last_sample' in st.session_state and pil_image is None:
                    try:
                        pil_image  = Image.open(os.path.join(SAMPLE_IMG_DIR,
                                                              st.session_state['last_sample'])).convert('RGB')
                        image_name = st.session_state['last_sample']
                    except Exception:
                        pass
            else:
                st.info("No images found in data directory.")
        else:
            st.info("Dataset image directory not found.")

    if pil_image is not None:
        st.image(pil_image, caption=image_name, use_container_width=True)

        st.markdown("")
        gen_btn = st.button(
            "Generate Caption",
            use_container_width=True,
            disabled=(not model_ok),
        )

        if gen_btn:
            if model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner("Generating..."):
                    caption, elapsed = run_caption(
                        model, device, pil_image,
                        beam_size=beam_size, max_len=max_len,
                    )

                st.session_state['last_caption']  = caption
                st.session_state['last_elapsed']  = elapsed
                st.session_state['last_img_name'] = image_name

                st.session_state.history.insert(0, {
                    'caption':   caption,
                    'filename':  image_name,
                    'time':      time.strftime('%H:%M:%S'),
                    'beam_size': beam_size,
                    'elapsed':   elapsed,
                })

        if st.session_state.get('last_caption') and st.session_state.get('last_img_name') == image_name:
            cap     = st.session_state['last_caption']
            elapsed = st.session_state['last_elapsed']

            st.markdown(f"""
<div class="caption-box">
    <div class="label">Caption</div>
    <div class="caption-text">"{cap}"</div>
</div>
""", unsafe_allow_html=True)

            info_cols = st.columns(3)
            info_cols[0].metric("Inference", f"{elapsed:.2f}s")
            info_cols[1].metric("Beam size", beam_size)
            info_cols[2].metric("Words",     len(cap.split()))

    else:
        st.markdown("""
<div class="glass-card" style="text-align:center; padding: 3rem 1rem; color: rgba(255,255,255,0.35);">
    <div style="font-size:3rem;"></div>
    <div style="margin-top:0.75rem; font-size:0.95rem;">Upload an image or select a sample</div>
</div>
""", unsafe_allow_html=True)


with right_col:
    st.markdown("### Model Info")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if cfg:
        device_str = 'CUDA' if torch.cuda.is_available() else 'CPU'
        encoder = cfg.get('encoder_name', 'ViT')
        decoder = cfg.get('decoder_name', 'GPT-2')
        st.markdown(f"""
<div class="metric-row">
  <div class="metric-pill">Encoder <span>{encoder}</span></div>
  <div class="metric-pill">Decoder <span>{decoder}</span></div>
  <div class="metric-pill">Device <span>{device_str}</span></div>
</div>
""", unsafe_allow_html=True)
    elif model_ok:
        st.caption("Model loaded")
    else:
        st.caption("No model loaded")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### History")

    if not st.session_state.history:
        st.markdown("""
<div class="glass-card" style="text-align:center; color: rgba(255,255,255,0.3); padding: 2rem 1rem; font-size:0.9rem;">
    No history yet.
</div>
""", unsafe_allow_html=True)
    else:
        for item in st.session_state.history[:10]:
            st.markdown(f"""
<div class="history-item">
  <div style="font-size:1.6rem;"></div>
  <div style="flex:1; min-width:0;">
    <div class="history-caption">"{item['caption']}"</div>
    <div class="history-time">
      {item['filename']} | beam={item['beam_size']} | {item['elapsed']:.2f}s | {item['time']}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
IMG_WIDTH, IMG_HEIGHT = 150, 150
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "gender_image_classifier.h5")
st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="🧑",
    layout="centered",
)
st.markdown(
    """
    <style>
    #dev-badge-wrap {
        position: fixed;
        top: 12px;
        right: 16px;
        z-index: 9999;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #dev-badge-wrap .dev-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6dd5ed, #2193b0);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        cursor: default;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        transition: border-radius 0.2s ease;
    }
    #dev-badge-wrap .dev-card {
        max-height: 0;
        overflow: hidden;
        opacity: 0;
        width: 240px;
        margin-top: 8px;
        background: #10222c;
        border: 1px solid #2193b0;
        border-radius: 12px;
        padding: 0 14px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        transition: max-height 0.25s ease, opacity 0.25s ease, padding 0.25s ease;
        color: #e6f6fb;
        font-size: 12px;
    }
    #dev-badge-wrap:hover .dev-card {
        max-height: 220px;
        opacity: 1;
        padding: 14px;
    }
    #dev-badge-wrap .dev-card h4 {
        margin: 0 0 2px 0;
        font-size: 15px;
        color: #ffffff;
    }
    #dev-badge-wrap .dev-card .role {
        color: #6dd5ed;
        font-size: 11px;
        margin-bottom: 8px;
    }
    #dev-badge-wrap .dev-card .edu {
        opacity: 0.8;
        line-height: 1.5;
        margin-bottom: 10px;
    }
    #dev-badge-wrap .dev-card a {
        color: #6dd5ed;
        text-decoration: none;
        display: block;
        margin-bottom: 4px;
    }
    #dev-badge-wrap .dev-card a:hover {
        text-decoration: underline;
    }
    </style>

    <div id="dev-badge-wrap">
        <div class="dev-circle">👤</div>
        <div class="dev-card">
            <h4>Aditya Agarwal</h4>
            <div class="role">Data Science / ML Enthusiast</div>
            <div class="edu">
                🎓 B.Tech, Computer Science Engineering<br/>
                Shri Ramswaroop Memorial College of Engineering &amp; Management, Lucknow
            </div>
            <a href="mailto:aasblko@gmail.com">📧 Email</a>
            <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank">💼 LinkedIn</a>
            <a href="https://github.com/DragonWarrior9842" target="_blank">🐙 GitHub</a>
            <a href="https://www.instagram.com/adityaagarwal67/" target="_blank">🌐 Instagram</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
@st.cache_resource
def get_model():
    if not os.path.isfile(MODEL_PATH):
        return None, f"Model file not found at '{MODEL_PATH}'."
    try:
        model = load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
    x = np.array(pil_img, dtype=np.float32)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x
st.title("🧑 Male vs Female Classifier 👩")
st.write("Upload a face image to get a prediction.")
model, load_error = get_model()
if load_error:
    st.error(
        "Could not load the model.\n\n"
        f"Details: {load_error}\n\n"
        "Make sure `gender_image_classifier.h5` is in the same folder as `app.py`."
    )
    st.stop()
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Classifying..."):
            x = preprocess_image(img)
            predictions = model.predict(x)
            pred = float(predictions[0][0])
        if pred >= 0.5:
            label = "Male 🧑"
            confidence = pred
        else:
            label = "Female 👩"
            confidence = 1 - pred
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        st.progress(min(max(confidence, 0.0), 1.0))
        with st.expander("Raw model output"):
            st.write("predictions:", predictions.tolist())
            st.write("predictions.shape:", predictions.shape)
            st.caption("Score close to 0 → Female, score close to 1 → Male.")
    except Exception as e:
        st.error(f"Something went wrong while processing the image: {e}")
else:
    st.info("👆 Upload a JPG or PNG image to get a prediction.")
st.markdown("---")
st.caption(
    "Model: CNN (3× Conv2D + MaxPooling, Dense(128) + Dropout, sigmoid output), "
    "150×150 RGB input, trained for binary male/female classification."
)
st.caption(
    "⚠️ Note: gender-classification models can be biased by their training data "
    "and may not perform equally well across all ages, ethnicities, or presentations. "
    "Predictions should be treated as approximate, not authoritative."
)

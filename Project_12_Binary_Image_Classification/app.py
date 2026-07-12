import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
# ---------------------------------------------------------
# Config — must match what the model was trained on
# ---------------------------------------------------------
IMG_WIDTH, IMG_HEIGHT = 150, 150
# Dynamically get the folder where app.py lives, then attach the model filename
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "binary_image_classifier.h5")
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

# ---------------------------------------------------------
# 👨‍💻 ABOUT THE DEVELOPER — Native Streamlit Popover
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Space+Mono:wght@400;700&display=swap');

    #dev-popover-content {
        font-family: 'Fredoka', sans-serif;
        text-align: center;
        padding: 4px 2px;
    }

    #dev-popover-content .paw-avatar {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        margin: 0 auto 8px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        background: linear-gradient(135deg, #ffb26b, #ff7e5f);
        box-shadow: 0 0 0 3px rgba(255,255,255,0.15);
    }

    #dev-popover-content h4 {
        font-family: 'Fredoka', sans-serif;
        font-weight: 700;
        font-size: 17px;
        margin: 2px 0 2px 0;
        color: inherit;
    }

    #dev-popover-content .role {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: #ff7e5f;
        margin-bottom: 8px;
    }

    #dev-popover-content .edu {
        font-family: 'Space Mono', monospace;
        font-size: 10.5px;
        line-height: 1.55;
        opacity: 0.75;
        margin-bottom: 10px;
    }

    #dev-popover-content .paw-links {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
    }

    #dev-popover-content .paw-links a {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        text-decoration: none;
        background: #ff7e5f;
        color: white !important;
        padding: 5px 10px;
        border-radius: 14px;
        display: inline-block;
        transition: transform 0.15s ease;
    }

    #dev-popover-content .paw-links a:hover {
        transform: scale(1.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    with st.popover("🐾 About the Developer", use_container_width=True):
        st.markdown(
            """
            <div id="dev-popover-content">
                <div class="paw-avatar">🐾</div>
                <h4>Aditya Agarwal</h4>
                <div class="role">// Data Science &amp; ML Enthusiast</div>
                <div class="edu">
                    🎓 B.Tech, Computer Science Engineering<br/>
                    Shri Ramswaroop Memorial College of<br/>
                    Engineering &amp; Management, Lucknow
                </div>
                <div class="paw-links">
                    <a href="https://github.com/DragonWarrior9842" target="_blank">🐙 GitHub</a>
                    <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank">💼 LinkedIn</a>
                    <a href="mailto:aasblko@gmail.com">✉️ Email</a>
                    <a href="https://www.instagram.com/adityaagarwal67/" target="_blank">🌐 Instagram</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    if not os.path.isfile(MODEL_PATH):
        return None, f"Model file not found at '{MODEL_PATH}'."
    try:
        model = load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)
# ---------------------------------------------------------
# Preprocessing (matches training: resize -> RGB -> /255)
# ---------------------------------------------------------
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")  # handles grayscale/PNG safely
    pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
    x = np.array(pil_img, dtype=np.float32)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)  # add batch dimension
    return x
# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image of a cat or a dog to get a prediction.")
model, load_error = get_model()
if load_error:
    st.error(
        "Could not load the model.\n\n"
        f"Details: {load_error}\n\n"
        "Make sure `binary_image_classifier.h5` is in the same folder as `app.py`."
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
        # class_mode='binary' with flow_from_directory assigns classes
        # alphabetically -> cats=0, dogs=1
        if pred >= 0.5:
            label = "Dog 🐶"
            confidence = pred
        else:
            label = "Cat 🐱"
            confidence = 1 - pred
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        st.progress(min(max(confidence, 0.0), 1.0))
        with st.expander("Raw model output"):
            st.write("predictions:", predictions.tolist())
            st.write("predictions.shape:", predictions.shape)
            st.caption("Score close to 0 → Cat, score close to 1 → Dog.")
    except Exception as e:
        st.error(f"Something went wrong while processing the image: {e}")
else:
    st.info("👆 Upload a JPG or PNG image to get a prediction.")
st.markdown("---")
st.caption(
    "Model: CNN (3× Conv2D + MaxPooling, Dense(128), sigmoid output), "
    "150×150 RGB input, trained for binary cat/dog classification."
)

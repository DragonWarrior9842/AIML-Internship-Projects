import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
IMG_WIDTH, IMG_HEIGHT = 224, 224
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "covid_xray_classifier.h5")
st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🫁", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
    #dev-badge-toggle { display: none; }
    #dev-clip {
        position: fixed; top: 0; left: 50%; transform: translateX(-50%); z-index: 9999;
        width: 46px; height: 20px; background: #0f766e; border-radius: 0 0 8px 8px;
        cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #f0fdfa;
    }
    #dev-id-card {
        position: fixed; top: -420px; left: 50%; transform: translateX(-50%);
        width: 260px; background: #ffffff; border-radius: 4px 4px 14px 14px;
        box-shadow: 0 14px 40px rgba(15,23,42,0.35); z-index: 9998;
        transition: top 0.45s cubic-bezier(0.34, 1.2, 0.64, 1); overflow: hidden;
        border: 1px solid #ccfbf1;
    }
    #dev-badge-toggle:checked ~ #dev-id-card { top: 18px; }
    #dev-id-card .strip { background: #0f766e; height: 10px; width: 100%; }
    #dev-id-card .body { padding: 18px 20px 20px 20px; text-align: center; }
    #dev-id-card .photo {
        width: 58px; height: 58px; border-radius: 50%; margin: 0 auto 10px auto;
        background: #ccfbf1; display: flex; align-items: center; justify-content: center;
        font-size: 26px; border: 3px solid #0f766e;
    }
    #dev-id-card h4 {
        font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; font-size: 17px;
        color: #134e4a; margin: 0 0 2px 0;
    }
    #dev-id-card .role {
        font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #0f766e;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;
    }
    #dev-id-card .edu {
        font-family: 'IBM Plex Sans', sans-serif; font-size: 11px; color: #475569;
        line-height: 1.5; margin-bottom: 12px; padding-bottom: 12px;
        border-bottom: 1px dashed #99f6e4;
    }
    #dev-id-card .links a {
        font-family: 'IBM Plex Sans', sans-serif; display: inline-block; font-size: 11px;
        color: #0f766e; text-decoration: none; margin: 0 6px; font-weight: 500;
    }
    #dev-id-card .links a:hover { color: #134e4a; text-decoration: underline; }
    #dev-id-card .id-no {
        font-family: 'IBM Plex Mono', monospace; font-size: 9px; color: #94a3b8;
        margin-top: 10px; letter-spacing: 0.5px;
    }
    </style>
    <input type="checkbox" id="dev-badge-toggle" />
    <label id="dev-clip" for="dev-badge-toggle">ID ▾</label>
    <div id="dev-id-card">
        <div class="strip"></div>
        <div class="body">
            <div class="photo">🩺</div>
            <h4>Aditya Agarwal</h4>
            <div class="role">Data Science / ML Enthusiast</div>
            <div class="edu">
                B.Tech, Computer Science Engineering<br/>
                Shri Ramswaroop Memorial College of Engineering &amp; Management, Lucknow
            </div>
            <div class="links">
                <a href="mailto:aasblko@gmail.com">Email</a>
                <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank">LinkedIn</a>
                <a href="https://github.com/DragonWarrior9842" target="_blank">GitHub</a>
                <a href="https://www.instagram.com/adityaagarwal67/" target="_blank">Instagram</a>
            </div>
            <div class="id-no">DEV-ID · APP: CHEST-XRAY-CLF</div>
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
st.title("🫁 Chest X-ray Classifier")
st.write("Upload a chest X-ray image to classify it as COVID19, NORMAL, or PNEUMONIA.")
model, load_error = get_model()
if load_error:
    st.error(
        "Could not load the model.\n\n"
        f"Details: {load_error}\n\n"
        "Make sure `covid_xray_classifier.h5` is in the same folder as `app.py`."
    )
    st.stop()
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-ray", use_column_width=True)
        with st.spinner("Classifying..."):
            x = preprocess_image(img)
            predictions = model.predict(x)[0]
        predicted_index = int(np.argmax(predictions))
        label = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])
        if label == "COVID19":
            st.error(f"Prediction: **{label}**")
        elif label == "PNEUMONIA":
            st.warning(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        st.markdown("#### Class probabilities")
        for cls_name, prob in sorted(
                zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True
        ):
            st.write(f"{cls_name}: {prob * 100:.2f}%")
            st.progress(min(max(float(prob), 0.0), 1.0))
        with st.expander("Raw model output"):
            st.write("predictions:", predictions.tolist())
            st.write("class order:", CLASS_NAMES)
    except Exception as e:
        st.error(f"Something went wrong while processing the image: {e}")
else:
    st.info("👆 Upload a JPG or PNG chest X-ray image to get a prediction.")
st.markdown("---")
st.caption(
    "Model: CNN (4× Conv2D + MaxPooling, Dense(128) + Dropout, softmax output), "
    "224×224 RGB input, trained for 3-class chest X-ray classification "
    "(COVID19 / NORMAL / PNEUMONIA)."
)
st.caption(
    "⚠️ This tool is for educational/demonstration purposes only and is **not** "
    "a substitute for professional medical diagnosis. Predictions may be "
    "inaccurate — always consult a qualified healthcare provider for any "
    "medical concerns."
)

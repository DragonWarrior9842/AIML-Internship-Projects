import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Config — must match what the model was trained on
# ---------------------------------------------------------
IMG_WIDTH, IMG_HEIGHT = 224, 224
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]  # same order used during training

# Dynamically get the folder where app.py lives, then attach the model filename
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "covid_xray_classifier.h5")

st.set_page_config(
    page_title="Chest X-ray Classifier",
    page_icon="🫁",
    layout="centered",
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
    pil_img = pil_img.convert("RGB")  # handles grayscale X-rays / PNG safely
    pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
    x = np.array(pil_img, dtype=np.float32)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)  # add batch dimension
    return x


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
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
            predictions = model.predict(x)[0]  # array of 3 probabilities

        predicted_index = int(np.argmax(predictions))
        label = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])

        # Color-code the headline result for quick visual triage
        if label == "COVID19":
            st.error(f"Prediction: **{label}**")
        elif label == "PNEUMONIA":
            st.warning(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")

        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        # Show probability breakdown for all three classes, not just the winner
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
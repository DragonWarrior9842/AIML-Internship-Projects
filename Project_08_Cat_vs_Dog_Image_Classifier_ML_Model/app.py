import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "cat_dog_model.pkl"))
IMG_SIZE = 64
tab_classifier, tab_about = st.tabs(["🐾 Classifier", "👤 About the Developer"])
with tab_classifier:
    st.title("🐱 Cat vs Dog Image Classifier")
    st.write("Upload an image to predict whether it is a Cat or Dog.")
    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)
        resized = image.resize((IMG_SIZE, IMG_SIZE))
        resized = np.array(resized)
        resized = resized.flatten()
        prediction = model.predict([resized])[0]
        probability = model.predict_proba([resized])[0]
        if prediction == 0:
            st.success("🐱 Prediction: CAT")
        else:
            st.success("🐶 Prediction: DOG")
        st.subheader("Prediction Confidence")
        st.write(f"🐱 Cat Probability: **{probability[0] * 100:.2f}%**")
        st.write(f"🐶 Dog Probability: **{probability[1] * 100:.2f}%**")
with tab_about:
    st.header("About the Developer")
    st.markdown("### Aditya Agarwal")
    st.write("Data Science / ML Enthusiast")
    st.write(
        "Currently a student of Shri Ramswaroop Memorial College of "
        "Engineering and Management, Lucknow, pursuing a Bachelor of "
        "Technology in Computer Science Engineering."
    )
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("📧 [Email](mailto:aasblko@gmail.com)")
        st.markdown("💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/)")
    with c2:
        st.markdown("🐙 [GitHub](https://github.com/DragonWarrior9842)")
        st.markdown("🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)")
    st.markdown("---")
    st.caption("This classifier was built as part of a machine learning practice project.")

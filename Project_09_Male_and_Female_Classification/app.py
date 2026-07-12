import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "male_and_female.pkl"))
st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="♂️♀️",
    layout="centered"
)
IMG_SIZE = 64
st.title("Male(♂️) vs Female(♀️) Image Classifier")
st.write("Upload an image to predict whether it is a Male or Female.")
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
        st.success("♂️ Prediction: Male")
    else:
        st.success("♀️ Prediction: Female")
    st.subheader("Prediction Confidence")
    st.write(f"♂️ Male Probability: **{probability[0] * 100:.2f}%**")
    st.write(f"♀️ Female Probability: **{probability[1] * 100:.2f}%**")
st.markdown("---")
with st.expander("👤 About the Developer"):
    st.markdown("**Aditya Agarwal**")
    st.caption("Data Science / ML Enthusiast")
    st.write(
        "B.Tech CSE student at Shri Ramswaroop Memorial College of "
        "Engineering and Management, Lucknow."
    )
    st.markdown(
        "📧 [Email](mailto:aasblko@gmail.com) &nbsp;·&nbsp; "
        "💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) &nbsp;·&nbsp; "
        "🐙 [GitHub](https://github.com/DragonWarrior9842) &nbsp;·&nbsp; "
        "🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)"
    )

Live Preview:- https://aiml-internship-projects-yq4ebsidacdyxfv5r4wpqs.streamlit.app/

# 🐱 Cat vs Dog Classifier 🐶

A Streamlit app that classifies an uploaded image as a **Cat** or **Dog** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras, showing the predicted class and confidence score.

---

## ✨ Features

- **Upload an image** (JPG/JPEG/PNG) and get an instant Cat/Dog prediction.
- **Confidence score** with a progress bar.
- **Raw model output** expander — inspect the exact sigmoid output value and prediction shape.
- **"About the Developer" popover** in the sidebar — a playful, paw-themed card (Fredoka + Space Mono fonts, orange gradient avatar) with pill-shaped contact links.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [TensorFlow / Keras](https://www.tensorflow.org/) — CNN model loading & inference
- [Pillow (PIL)](https://python-pillow.org/) — image loading, RGB conversion, resizing
- [NumPy](https://numpy.org/) — image preprocessing

---

## 📂 Required Files

The app expects a trained Keras model file named **`binary_image_classifier.h5`** in the same folder as `app.py`. Per the model summary shown in the app, it's a:

> CNN (3× Conv2D + MaxPooling, Dense(128), sigmoid output), 150×150 RGB input, trained for binary cat/dog classification.

- Output **< 0.5** → predicted **Cat 🐱**
- Output **≥ 0.5** → predicted **Dog 🐶**

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_12_CNN_Cat_Dog_Classifier   # adjust folder name if different

# install dependencies
pip install streamlit tensorflow pillow numpy

# make sure binary_image_classifier.h5 is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_12_CNN_Cat_Dog_Classifier/
├── app.py                        # Streamlit app (this project)
├── binary_image_classifier.h5    # Trained Keras CNN model
└── requirements.txt              # Python dependencies
```

---

## 📖 How It Works

1. The uploaded image is converted to RGB, resized to **150×150**, and normalized to `[0, 1]`.
2. A batch dimension is added and the image is passed through the CNN.
3. The model outputs a single sigmoid probability — values near `0` indicate **Cat**, values near `1` indicate **Dog**.
4. The predicted label and confidence are displayed, with the raw output available for inspection.

> **Note:** this is a different, deep-learning-based (CNN) cat/dog classifier from any earlier flattened-pixel scikit-learn version in this repo — expect noticeably better accuracy and robustness to lighting/framing here.

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science & ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

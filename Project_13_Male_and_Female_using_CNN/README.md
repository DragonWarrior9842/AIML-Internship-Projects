Live Preview:- https://aiml-internship-projects-xprausjxoxtmrark3scapk.streamlit.app/

# 🧑 Male vs Female Classifier 👩

A Streamlit app that classifies an uploaded face image as **Male** or **Female** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras, showing the predicted class and confidence score.

---

## ✨ Features

- **Upload a face image** (JPG/JPEG/PNG) and get an instant Male/Female prediction.
- **Confidence score** with a progress bar.
- **Raw model output** expander — inspect the exact sigmoid output value and prediction shape.
- **Hover-to-reveal developer badge** — a small circular avatar fixed in the top-right corner that expands into a contact card on hover (cyan/teal gradient, dark card).
- **Bias disclaimer** — the app explicitly notes that gender-classification models can be biased by their training data and may not perform equally well across ages, ethnicities, or presentations.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [TensorFlow / Keras](https://www.tensorflow.org/) — CNN model loading & inference
- [Pillow (PIL)](https://python-pillow.org/) — image loading, RGB conversion, resizing
- [NumPy](https://numpy.org/) — image preprocessing

---

## 📂 Required Files

The app expects a trained Keras model file named **`gender_image_classifier.h5`** in the same folder as `app.py`. Per the model summary shown in the app, it's a:

> CNN (3× Conv2D + MaxPooling, Dense(128) + Dropout, sigmoid output), 150×150 RGB input, trained for binary male/female classification.

- Output **< 0.5** → predicted **Female 👩**
- Output **≥ 0.5** → predicted **Male 🧑**

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_13_CNN_Gender_Classifier   # adjust folder name if different

# install dependencies
pip install streamlit tensorflow pillow numpy

# make sure gender_image_classifier.h5 is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_13_CNN_Gender_Classifier/
├── app.py                        # Streamlit app (this project)
├── gender_image_classifier.h5    # Trained Keras CNN model
└── requirements.txt              # Python dependencies
```

---

## 📖 How It Works

1. The uploaded image is converted to RGB, resized to **150×150**, and normalized to `[0, 1]`.
2. A batch dimension is added and the image is passed through the CNN.
3. The model outputs a single sigmoid probability — values near `0` indicate **Female**, values near `1` indicate **Male**.
4. The predicted label and confidence are displayed, with the raw output available for inspection.

> **Note:** this is a different, deep-learning-based (CNN) gender classifier from the earlier flattened-pixel scikit-learn version and the eye-image-only classifier elsewhere in this repo — keep the model filenames distinct so they aren't mixed up when deploying.

---

## ⚠️ Limitations

This tool is for educational/demonstration purposes only. Gender-classification models trained on limited or imbalanced datasets can be biased, and predictions may not generalize well across different ages, ethnicities, lighting conditions, or gender presentations. Treat all outputs as approximate, not authoritative.

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

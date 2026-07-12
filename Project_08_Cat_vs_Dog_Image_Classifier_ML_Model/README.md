Live Preview:- https://aiml-internship-projects-s7n7xvyp4hwlxmfdhkcnbm.streamlit.app/

# 🐱 Cat vs Dog Image Classifier

A Streamlit app that classifies an uploaded image as a **Cat** or **Dog** using a pre-trained scikit-learn model, showing the predicted class along with confidence probabilities for both classes.

---

## ✨ Features

- **Upload an image** (JPG/JPEG/PNG) and get an instant Cat/Dog prediction.
- **Prediction confidence** — shows the probability for both Cat and Dog classes.
- **Two tabs:**
  - 🐾 **Classifier** — the main upload-and-predict interface
  - 👤 **About the Developer** — contact details

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pillow (PIL)](https://python-pillow.org/) — image loading, RGB conversion, resizing
- [NumPy](https://numpy.org/) — image-to-array flattening
- [joblib](https://joblib.readthedocs.io/) — loading the pre-trained model

---

## 📂 Required Files

The app expects a pre-trained model file named **`cat_dog_model.pkl`** in the same folder as `app.py`. It must be a scikit-learn-compatible classifier (supporting `.predict()` and `.predict_proba()`) trained on **64×64 RGB images flattened to a 1D array** (`64 × 64 × 3 = 12,288` features), with class labels:

- `0` → Cat
- `1` → Dog

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_08_Cat_Dog_Classifier   # adjust folder name if different

# install dependencies
pip install streamlit numpy pillow joblib scikit-learn

# make sure cat_dog_model.pkl is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_08_Cat_Dog_Classifier/
├── app.py               # Streamlit app (this project)
├── cat_dog_model.pkl     # Pre-trained scikit-learn classifier
└── requirements.txt      # Python dependencies
```

---

## 📖 How It Works

1. The uploaded image is converted to RGB and resized to **64×64**.
2. The resized image is flattened into a single 1D feature vector.
3. The pre-trained model predicts the class (`0` = Cat, `1` = Dog) and returns class probabilities.
4. The result and confidence breakdown are displayed instantly.

> **Note:** since prediction relies on a simple flattened-pixel representation rather than a CNN's learned features, accuracy will vary a lot based on image framing, lighting, and background compared to a deep-learning-based classifier.

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

Live Preview:- https://aiml-internship-projects-6b6mptmzpsy2zjhj25fzsm.streamlit.app/

# ♂️♀️ Male vs Female Image Classifier

A Streamlit app that classifies an uploaded image as **Male** or **Female** using a pre-trained scikit-learn model, showing the predicted class along with confidence probabilities for both classes.

---

## ✨ Features

- **Upload an image** (JPG/JPEG/PNG) and get an instant Male/Female prediction.
- **Prediction confidence** — shows the probability for both Male and Female classes.
- **About the Developer** expander with contact details.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pillow (PIL)](https://python-pillow.org/) — image loading, RGB conversion, resizing
- [NumPy](https://numpy.org/) — image-to-array flattening
- [joblib](https://joblib.readthedocs.io/) — loading the pre-trained model

---

## 📂 Required Files

The app expects a pre-trained model file named **`male_and_female.pkl`** in the same folder as `app.py`. It must be a scikit-learn-compatible classifier (supporting `.predict()` and `.predict_proba()`) trained on **64×64 RGB images flattened to a 1D array** (`64 × 64 × 3 = 12,288` features), with class labels:

- `0` → Male
- `1` → Female

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_09_Male_Female_Classifier   # adjust folder name if different

# install dependencies
pip install streamlit numpy pillow joblib scikit-learn

# make sure male_and_female.pkl is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_09_Male_Female_Classifier/
├── app.py                  # Streamlit app (this project)
├── male_and_female.pkl     # Pre-trained scikit-learn classifier
└── requirements.txt        # Python dependencies
```

---

## 📖 How It Works

1. The uploaded image is converted to RGB and resized to **64×64**.
2. The resized image is flattened into a single 1D feature vector.
3. The pre-trained model predicts the class (`0` = Male, `1` = Female) and returns class probabilities.
4. The result and confidence breakdown are displayed instantly.

> **Note:** since prediction relies on a simple flattened-pixel representation rather than a CNN's learned features, accuracy will vary a lot based on image framing, lighting, and background compared to a deep-learning-based classifier. As with any demographic classifier, predictions are statistical estimates only and shouldn't be treated as authoritative.

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

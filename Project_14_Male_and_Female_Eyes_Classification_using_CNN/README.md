Live Preview:- https://aiml-internship-projects-yuypv49xsboqj3gtawkhsk.streamlit.app/

# 👁️ Male vs Female Eye Classifier

A simple Streamlit web app that predicts whether an uploaded close-up **eye image** belongs to a **Male** or **Female**, using a trained Convolutional Neural Network (CNN).

---

## 🚀 Demo

Upload a close-up image of an eye, and the app will:
- Display the uploaded image
- Predict the gender (Male/Female) with a confidence score
- Show the raw model output for transparency

---

## 🧠 Model Details

- **Architecture:** CNN with 3× `Conv2D` + `MaxPooling` layers, followed by a `Dense(128)` layer with `Dropout`, and a final sigmoid output layer
- **Input size:** 150×150 RGB images
- **Output:** Binary classification
  - Score close to **0** → Female
  - Score close to **1** → Male
- **Training data:** [Kaggle 'eyes-rtte' dataset](https://www.kaggle.com/)

> ⚠️ **Disclaimer:** This model was trained on a specific dataset and may not generalize well to eyes with different lighting, angles, makeup, or image quality. Predictions should be treated as approximate, not authoritative.

---

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── eye_gender_classifier.h5        # Trained Keras model (required)
└── README.md
```

> **Note:** The `eye_gender_classifier.h5` model file must be placed in the same directory as `app.py` for the app to work.

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DragonWarrior9842/<repo-name>.git
   cd <repo-name>
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit numpy pillow tensorflow
   ```

   Or, if you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the model file**

   Place `eye_gender_classifier.h5` in the project's root directory (same folder as `app.py`).

---

## ▶️ Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) in your browser.

**Steps in the app:**
1. Upload a `.jpg`, `.jpeg`, or `.png` image of an eye.
2. Wait for the model to classify the image.
3. View the predicted label, confidence score, and raw model output.

---

## 📦 Dependencies

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/)

---

## 👨‍💻 Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
🎓 B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

- 📧 Email: aasblko@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/)
- 🐙 [GitHub](https://github.com/DragonWarrior9842)
- 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute it as per your needs (add a specific license, e.g. MIT, if desired).

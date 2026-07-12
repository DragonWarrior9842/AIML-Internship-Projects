Live Preview:- https://chest-xray-disease-detection-by-aditya-agarwal.streamlit.app/

# 🫁 Chest X-ray Classifier

A Streamlit web app that classifies chest X-ray images into **COVID19**, **NORMAL**, or **PNEUMONIA** categories using a trained Convolutional Neural Network (CNN).

> ⚠️ **This tool is for educational/demonstration purposes only** and is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for any medical concerns.

---

## 🚀 Demo

Upload a chest X-ray image, and the app will:
- Display the uploaded X-ray
- Predict the class (COVID19 / NORMAL / PNEUMONIA) with a confidence score
- Show a full breakdown of class probabilities with visual progress bars
- Provide the raw model output for transparency

---

## 🧠 Model Details

- **Architecture:** CNN with 4× `Conv2D` + `MaxPooling` layers, followed by a `Dense(128)` layer with `Dropout`, and a final softmax output layer
- **Input size:** 224×224 RGB images
- **Output:** 3-class classification — `COVID19`, `NORMAL`, `PNEUMONIA`
- **Result styling:** Color-coded predictions in the UI (error for COVID19, warning for PNEUMONIA, success for NORMAL)

---

## 📁 Project Structure

```
├── app.py                       # Main Streamlit application
├── covid_xray_classifier.h5     # Trained Keras model (required)
└── README.md
```

> **Note:** The `covid_xray_classifier.h5` model file must be placed in the same directory as `app.py` for the app to work.

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

   Place `covid_xray_classifier.h5` in the project's root directory (same folder as `app.py`).

---

## ▶️ Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) in your browser.

**Steps in the app:**
1. Upload a `.jpg`, `.jpeg`, or `.png` chest X-ray image.
2. Wait for the model to classify the image.
3. View the predicted label, confidence score, full class probability breakdown, and raw model output.

---

## 📦 Dependencies

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/)

---

## ⚠️ Disclaimer

This model was trained on a specific dataset for demonstration purposes and has not been clinically validated. It should **never** be used as a basis for real medical decisions. Predictions may be inaccurate, and image quality, positioning, and equipment differences can significantly affect results.

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

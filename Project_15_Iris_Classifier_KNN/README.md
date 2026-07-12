Live Preview:- https://aiml-internship-projects-4zm2hqkzgzkrxwunjsfnfs.streamlit.app/

# 🌸 KNN — Iris Flower Classifier

An interactive Streamlit app that recreates a full K-Nearest Neighbors (KNN) case study on the classic **Iris flower dataset** — covering data exploration, visualization, model training, evaluation, and live predictions, all in one dashboard.

---

## 🚀 Features

- **🏠 Overview** — Dataset structure, summary statistics, null checks, and species distribution
- **📊 EDA** — Feature distribution histograms, per-species comparisons, and interactive pairwise scatter plots
- **🌡️ Correlation** — Correlation matrix table and heatmap across all features
- **🔵 t-SNE** — 2D t-SNE projection to visualize cluster separation between species
- **🤖 Model & Evaluation** — Train/test split details, live accuracy, confusion matrix, and an accuracy-vs-k curve
- **🔮 Try It Yourself** — Enter custom flower measurements and get a real-time species prediction with class probabilities

All model settings (`k` neighbors, test size, random state) are adjustable live from the sidebar, and the model retrains instantly.

---

## 🧠 Model Details

- **Algorithm:** K-Nearest Neighbors (`sklearn.neighbors.KNeighborsClassifier`)
- **Dataset:** Classic Iris dataset (`sklearn.datasets.load_iris`) — 3 species: *setosa*, *versicolor*, *virginica*
- **Features used:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Adjustable hyperparameters:** number of neighbors (`k`), test set size, random state
- **Evaluation:** Accuracy score, confusion matrix, accuracy-vs-k comparison curve

---

## 📁 Project Structure

```
├── app.py          # Main Streamlit application
└── README.md
```

> No external dataset or model file is required — the Iris dataset is loaded directly from `scikit-learn`, and the KNN model is trained live within the app.

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
   pip install streamlit numpy pandas matplotlib seaborn scikit-learn
   ```

   Or, if you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) in your browser.

**In the app:**
1. Adjust `k` (neighbors), test set size, and random state from the sidebar.
2. Explore the dataset and visualizations across the tabs.
3. Head to **"Try It Yourself"** to input custom flower measurements and get a live prediction.

---

## 📦 Dependencies

- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

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

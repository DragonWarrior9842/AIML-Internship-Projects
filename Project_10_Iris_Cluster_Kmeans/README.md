Live Preview:- https://aiml-internship-projects-djxf5lttfkg4uywcgnmgjl.streamlit.app/

# 🌸 K-Means Clustering — Iris Petal Length & Width

An interactive app reproducing a classic K-Means clustering exercise on the Iris dataset — clustering flowers by **petal length and width only**, exploring the effect of feature scaling, and using an elbow plot to pick the optimal number of clusters (k).

---

## ✨ Features

- **Elbow Plot** — computes and plots Sum of Squared Error (SSE) across a configurable range of k values, with the currently selected k marked, to help identify the "elbow" point (typically k = 3 for this dataset).
- **Adjustable clustering settings** (sidebar):
  - Toggle **MinMax Scaling** on/off to see whether preprocessing changes the resulting clusters
  - **Number of clusters (k)** slider (2–9)
  - **Max k to test** in the elbow plot (3–15)
- **Clustering visualization** — scatter plot of petal length vs. width, colored by cluster, with centroids marked (automatically un-scaled back to original units for display when scaling is on).
- **Cluster sizes & sample table** — quick counts and a preview of the clustered data.
- **Ground-truth comparison** — an expandable cross-tab comparing the K-Means cluster assignments against the true Iris species labels, to see how well unsupervised clustering recovers the known species groupings.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [Matplotlib](https://matplotlib.org/) — elbow plot & cluster scatter plot
- [scikit-learn](https://scikit-learn.org/) — `KMeans`, `MinMaxScaler`, and the bundled `load_iris()` dataset

---

## 📂 Dataset

No external file needed — the app loads the classic **Iris dataset** directly from `sklearn.datasets.load_iris()`, using only the `petal length (cm)` and `petal width (cm)` features (sepal features are dropped, as per the exercise instructions).

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_10_KMeans_Iris_Clustering   # adjust folder name if different

# install dependencies
pip install streamlit pandas numpy matplotlib scikit-learn

# run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_10_KMeans_Iris_Clustering/
├── app.py               # Streamlit app (this project) — no external data file needed
└── requirements.txt      # Python dependencies
```

---

## 📖 How It Works

1. **Load** the Iris dataset and keep only the petal length/width columns.
2. **Optionally scale** the features to [0, 1] with `MinMaxScaler`, since K-Means is distance-based and sensitive to feature scale.
3. **Elbow method** — fit K-Means for k = 1 to the chosen max, and plot inertia (SSE) vs. k to visually identify diminishing returns.
4. **Fit K-Means** at the selected k and plot the resulting clusters with centroids.
5. **Compare** the unsupervised clusters against the true species labels (for reference only — K-Means never sees these labels during training).

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

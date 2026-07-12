Live Preview:- https://aiml-internship-projects-e9qlzuknswtct3nedwuava.streamlit.app/

# 📊 K-Means Clustering — Interactive Tutorial

An interactive two-part K-Means clustering tutorial: **customer segmentation by age & income**, plus the classic **Iris petal-clustering exercise** — both with side-by-side raw vs. scaled comparisons and elbow plots for picking the optimal number of clusters.

Based on the "Introduction to K Means Clustering" tutorial.

---

## ✨ Features

### 👥 Tab 1 — Customer Income Clustering
- Loads `income.csv` and plots the raw **Age vs. Income** scatter.
- **Raw vs. Scaled clustering, side by side** — since Age and Income sit on very different scales (years vs. dollars), the app runs K-Means on both the raw data and MinMax-scaled data so you can visually compare how much scaling improves cluster shape.
- **Elbow plot** (on scaled data) with adjustable max-k, to help choose the right number of clusters.
- Expandable raw data and clustered-data tables.

### 🌸 Tab 2 — Iris Petal Clustering (Exercise)
- Clusters iris flowers using only **petal length** and **petal width** (sepal features dropped, per the exercise).
- Toggle **MinMax Scaling** on/off.
- **Elbow plot** with adjustable max-k.
- Cluster scatter plot with centroids (un-scaled back to original units for display).
- Cluster sizes, sample data table, and a **ground-truth comparison** — a cross-tab of predicted clusters vs. true Iris species, to see how well unsupervised clustering recovers the known species groupings.

### Extras
- A collapsible **"Meet the Developer"** drawer pinned to the bottom of the page — a dark, editorial-style panel (Playfair Display + IBM Plex Mono) with skill chips and contact links, tucked out of the way until clicked open.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [Matplotlib](https://matplotlib.org/) — scatter plots & elbow plots
- [scikit-learn](https://scikit-learn.org/) — `KMeans`, `MinMaxScaler`, and the bundled `load_iris()` dataset

---

## 📂 Dataset

- **`income.csv`** — required in the same folder as `app.py`, with `Age` and `Income($)` columns:
  ```
  Name,Age,Income($)
  Rob,27,70000
  Michael,29,90000
  ...
  ```
- **Iris dataset** — loaded automatically from `sklearn.datasets.load_iris()`, no file needed.

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_11_KMeans_Tutorial   # adjust folder name if different

# install dependencies
pip install streamlit pandas numpy matplotlib scikit-learn

# make sure income.csv is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_11_KMeans_Tutorial/
├── app.py            # Streamlit app (this project)
├── income.csv         # Customer age/income dataset
└── requirements.txt   # Python dependencies
```

---

## 📖 How It Works

1. **Customer data:** load `Age`/`Income($)`, fit K-Means once on the raw values and once on MinMax-scaled values, and compare the two cluster layouts side by side.
2. **Elbow method:** for both datasets, fit K-Means across a range of k and plot inertia (SSE) vs. k to visually spot the point of diminishing returns.
3. **Iris exercise:** repeat the same scale-vs-raw and elbow-plot workflow on the Iris petal features, then cross-check the resulting clusters against the known species labels (for reference only — K-Means never sees these labels during training).

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data & ML
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

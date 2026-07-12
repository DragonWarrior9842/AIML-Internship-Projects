Live Review:- https://aiml-internship-projects-5vjmfs3wvtmkeg3eje3y3l.streamlit.app/
# 📱 Google Play Store — Data Visualisation Case Study

An interactive Streamlit dashboard that walks through a full exploratory data analysis (EDA) case study on the Google Play Store apps dataset — from raw-data cleaning and sanity checks to a wide range of visualisations (Matplotlib, Seaborn, and Plotly) exploring what separates a well-performing app from the rest.

**🔗 Live demo:** [aiml-internship-projects-5vjmfs3wvtmkeg3eje3y3l.streamlit.app](https://aiml-internship-projects-5vjmfs3wvtmkeg3eje3y3l.streamlit.app/)

---

## 🧭 Problem Statement

- Does a higher **size** or **price** necessarily mean an app performs better than others?
- Does a higher number of **installs** give a clear picture of which app will have a better rating?

---

## ✨ Features

- **Interactive filters** — filter the entire dashboard by Category, Content Rating, and Free/Paid type from the sidebar.
- **Data cleaning pipeline** with a full transparency log, including:
  - Dropping rows with a null `Rating` (the target variable)
  - Fixing shifted/garbage rows and incorrect data types (`Price`, `Reviews`, `Installs`)
  - Imputing missing `Android Ver` / `Current Ver` with the mode
  - Sanity checks (e.g. `Reviews > Installs`, free apps priced `> 0`)
  - Outlier removal on `Price`, `Reviews`, `Installs`
  - Dropping under-represented `Content Rating` categories
- **8 dashboard tabs:**
  | Tab | What it shows |
  |---|---|
  | 🏠 Overview | Problem statement, cleaned data sample, summary statistics |
  | 🧹 Data Cleaning | Step-by-step cleaning log + before/after boxplots |
  | 📊 Distributions | Histograms and an interactive Seaborn `histplot` (adjustable bins, color, style) |
  | 🧩 Categorical Charts | Pie/bar charts, aggregated bar plots, boxplots by Content Rating and Genre |
  | 🔵 Scatter / Pair Plots | Size vs. Rating scatter, interactive joint plots, pair plots |
  | 🌡️ Heat Map | Pivot-table heatmap of Rating by Content Rating × Size bucket |
  | 📈 Time & Stacked Charts | Monthly average rating (Matplotlib + Plotly), stacked install charts |
  | 🗂️ Raw Data | Filtered data table with CSV download |
- **CSV upload support** — bring your own `googleplaystore_v2.csv`, or the app auto-locates a bundled copy.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data cleaning & manipulation
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) — static visualisations
- [Plotly Express](https://plotly.com/python/plotly-express/) — interactive charts

---

## 📂 Dataset

The app expects a CSV named **`googleplaystore_v2.csv`** in the same folder as `app.py` (or uploaded via the sidebar), with columns including:

`App, Category, Rating, Reviews, Size, Installs, Type, Price, Content Rating, Genres, Last Updated, Current Ver, Android Ver`

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_01_Google_Play_Store

# install dependencies
pip install streamlit pandas numpy matplotlib seaborn plotly

# make sure googleplaystore_v2.csv is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_01_Google_Play_Store/
├── app.py                     # Streamlit app (this project)
├── googleplaystore_v2.csv     # Dataset (upload via sidebar if not present)
└── requirements.txt           # Python dependencies
```

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

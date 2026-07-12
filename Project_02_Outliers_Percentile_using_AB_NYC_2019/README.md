Live Preview:- https://aiml-internship-projects-wue9c8jwwuk8lhu5gknmxi.streamlit.app/

# :cityscape: NYC Airbnb — Percentile-Based Outlier Detection

An interactive Streamlit app that walks through removing outliers from a numeric column using the **percentile method**, applied to the `AB_NYC_2019` (New York City Airbnb Open Data) dataset. Adjust the cutoff percentiles live and see the before/after distributions, thresholds, and row counts update instantly.

---

## ✨ Features

- **Upload your own CSV** or use the bundled `AB_NYC_2019.csv` automatically.
- **Pick any numeric column** to clean (defaults to `price` if present).
- **Before-cleaning view:**
  - Summary statistics (`describe()`)
  - Histogram of the raw distribution
  - Boxplot of the raw distribution
- **Adjustable percentile cutoffs** — separate sliders for the lower (0–10%) and upper (90–100%) tail, with live validation (lower must be less than upper).
- **Threshold metrics** — shows the exact min/max cutoff values for the selected percentiles.
- **Inspect removed rows** — expandable tables showing exactly which rows fall below the min threshold and above the max threshold.
- **After-cleaning view:**
  - Rows before vs. after vs. removed count
  - Random sample of the cleaned data
  - Side-by-side "before vs. after" summary statistics
  - Cleaned histogram and boxplot
- **Download the cleaned CSV** directly from the app.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data loading & percentile calculations
- [Plotly Express](https://plotly.com/python/plotly-express/) — interactive histograms & boxplots

---

## 📂 Dataset

The app expects a CSV named **`AB_NYC_2019.csv`** in the same folder as `app.py` (or uploaded via the sidebar) — the standard [NYC Airbnb Open Data](http://insideairbnb.com/get-the-data.html) columns, including `price`, `minimum_nights`, `number_of_reviews`, `availability_365`, etc. Any numeric column in the file can be selected for outlier cleaning.

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_02_NYC_Airbnb   # adjust folder name if different

# install dependencies
pip install streamlit pandas numpy plotly

# make sure AB_NYC_2019.csv is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_02_NYC_Airbnb/
├── app.py               # Streamlit app (this project)
├── AB_NYC_2019.csv      # Dataset (upload via sidebar if not present)
└── requirements.txt     # Python dependencies
```

---

## 📖 How the Percentile Method Works

The percentile method removes the most extreme low and high values by cutting off a small percentage from each tail of a column's distribution:

1. Compute the value at the **lower percentile** (e.g. 1st percentile) and the **upper percentile** (e.g. 99.9th percentile).
2. Keep only rows where the column's value falls strictly between those two thresholds.
3. Compare the "before" and "after" distributions to confirm extreme outliers were removed without over-trimming the data.

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

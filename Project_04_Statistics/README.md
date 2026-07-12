Live Preview:- https://aiml-internship-projects-huyrsuvirjc468cwlf9b2i.streamlit.app/

# 📊 Statistics with Python — Interactive Course Companion

An interactive Streamlit app covering descriptive statistics, probability distributions, the Central Limit Theorem, one- and two-sample hypothesis tests, ANOVA, and chi-square tests — built as a hands-on companion to a statistics course, letting you plug in your own data or the bundled datasets and see every calculation, chart, and conclusion update live.

---

## ✨ Sections

The app is organized into 7 sections, selectable from the sidebar:

### 1. Descriptive Statistics
Mean, median, mode, range, variance, standard deviation, IQR, and outlier fences (1.5×IQR rule) on any numeric column, with a histogram + KDE and a boxplot.

### 2. Probability Distributions
Interactive **Binomial**, **Poisson**, and **Normal** distribution explorers:
- Adjustable parameters (n/p, λ, μ/σ) via sliders
- Mean, variance, std dev
- On-demand probability lookups (`P(X = k)`, `P(X ≤ k)`, `P(X > k)`)
- PMF/CDF bar charts and a Normal PDF plot with a probability-between-bounds calculator

### 3. Central Limit Theorem
Simulates drawing many samples from a non-normal population (uniform die-roll or bimodal) and visualizes how the distribution of **sample means** approaches normality as sample size grows, with population vs. sample-mean statistics side by side.

### 4. One-Sample Tests
- Mean (Z-test, known σ)
- Mean (t-test)
- Proportion (exact binomial test)
- Variance (Chi-square test)

Each includes hypothesis setup, alternative-hypothesis selection, significance level (α), visualizations, and a plain-language reject/fail-to-reject conclusion.

### 5. Two-Sample Tests
- Two Means (Z-test)
- Two Means (t-test, equal variance)
- Two Means (Welch's t-test, unequal variance)
- Paired t-test (with an editable before/after data table)
- Two Proportions (Wald / score / Agresti-Caffo methods)
- Two Variances (Bartlett & Levene tests)

### 6. ANOVA
One-way ANOVA to test whether 3+ group means are equal, with group selection, boxplot comparison, F-statistic/p-value, and a per-group summary table.

### 7. Chi-Square Tests
- **Goodness of Fit** — compare observed counts to expected proportions (editable data table)
- **Contingency Table (Independence)** — cross-tabulate two categorical variables, view expected counts under independence, and a heatmap of the contingency table

---

## 📂 Bundled Datasets

The app ships with (and lets you swap in your own CSV for) the following:

| File | Used in |
|---|---|
| `Machine1.csv` | One-sample mean tests |
| `Two_Machines.csv` | Two-sample mean/variance tests |
| `Piece_Dim.csv` | ANOVA |
| `Smokers.csv` | One-sample proportion test |
| `bottle_caps.csv` | General exploration |
| `Perfume_Volumes.csv` | General exploration |
| `titanic.csv` | Descriptive statistics, chi-square independence |

Every section also supports **"Upload my own CSV"** as an alternative to the bundled files.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) — visualizations
- [SciPy](https://scipy.org/) (`scipy.stats`) — distributions & hypothesis tests
- [statsmodels](https://www.statsmodels.org/) — two-sample Z-tests and proportion tests

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_04_Statistics_with_Python   # adjust folder name if different

# install dependencies
pip install streamlit pandas numpy matplotlib seaborn scipy statsmodels

# make sure the bundled CSVs are in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_04_Statistics_with_Python/
├── app.py                   # Streamlit app (this project)
├── Machine1.csv
├── Perfume_Volumes.csv
├── Piece_Dim.csv
├── Smokers.csv
├── Two_Machines.csv
├── bottle_caps.csv
├── titanic.csv
└── requirements.txt          # Python dependencies
```

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

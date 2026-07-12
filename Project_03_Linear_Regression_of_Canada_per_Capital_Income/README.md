Live Preview:- https://aiml-internship-projects-3yjpxperibkqorp4jtrcw6.streamlit.app/

# :chart_with_upwards_trend: Canada Per Capita Income Predictor

A simple linear regression app that trains on historical Canadian per-capita income data and predicts future (or in-range) values — with an interactive fitted trend line, single-year predictions, and batch forecasting for a range of years.

---

## ✨ Features

- **Upload your own CSV** or use the bundled `canada_per_capita_income.csv` automatically.
- **Step 1 — Model fit:** trains a `scikit-learn` `LinearRegression` model on `year → income`, and displays:
  - Slope (income per year)
  - Intercept
  - R² score
  - An interactive Plotly scatter + fitted regression line over the historical data
- **Step 2 — Single-year prediction:** pick any year (even beyond the dataset's range) and get a predicted per-capita income, plotted as a highlighted point on the regression line. The app flags when a prediction is an **extrapolation** beyond the last year of training data.
- **Step 3 — Batch predictions:** choose a "from" and "to" year to generate a table of predicted income for every year in that range, downloadable as CSV.
- **About the Developer dialog** — click the "👤 Developer" button in the top-right for a modal with contact details.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework (including `st.dialog` for the developer modal)
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data loading & array handling
- [scikit-learn](https://scikit-learn.org/) — `LinearRegression` model
- [Plotly](https://plotly.com/python/) — interactive charts

---

## 📂 Dataset

The app expects a CSV named **`canada_per_capita_income.csv`** in the same folder as `app.py` (or uploaded via the sidebar), with two columns (renamed internally to `year` and `income`):

```
year,per capita income (US$)
1970,3399.299037
1971,3768.297935
...
```

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_03_Linear_Regression_of_Canada_per_Capital_Income

# install dependencies
pip install streamlit pandas numpy scikit-learn plotly

# make sure canada_per_capita_income.csv is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_03_Linear_Regression_of_Canada_per_Capital_Income/
├── app.py                          # Streamlit app (this project)
├── canada_per_capita_income.csv    # Dataset (upload via sidebar if not present)
└── requirements.txt                # Python dependencies
```

---

## 📖 How It Works

1. **Load & sort** the historical `year` / `income` data.
2. **Fit** a linear regression model (`income = slope × year + intercept`).
3. **Predict** income for any requested year by evaluating the fitted line — predictions beyond the training data's max year are extrapolations, and the app calls this out explicitly.
4. **Visualize** both the historical fit and any individual/batch predictions on interactive Plotly charts.

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

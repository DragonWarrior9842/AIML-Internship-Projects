Live Preview:- https://aiml-internship-projects-brxmpkhxwgnis6umauftgq.streamlit.app/

# 🏠 House Price Prediction using Linear Regression

A minimal Streamlit app that trains a simple linear regression model on house area vs. price, then lets you enter a house's area to get a predicted price — along with the underlying model's coefficient, intercept, and R² score.

---

## ✨ Features

- **Loads and displays** the bundled `houseprice.csv` dataset in a table.
- **Trains a `scikit-learn` `LinearRegression`** model (`price ~ area`) on app load.
- **Interactive prediction** — enter an area (100–10,000, step 50) and click **"Predicted Price"** to get an instant price estimate.
- **Model transparency** — displays the regression coefficient, intercept, and R² score so you can see exactly how the prediction is calculated.
- **About the Developer** panel in the sidebar.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pandas](https://pandas.pydata.org/) — data loading
- [scikit-learn](https://scikit-learn.org/) — `LinearRegression` model

---

## 📂 Dataset

The app expects a CSV named **`houseprice.csv`** in the same folder as `app.py`, with an `area` column (or similar single feature column) and a `price` column (the target):

```
area,price
5000,750000
7000,980000
...
```

> **Note:** the model currently trains on **all** non-`price` columns in the CSV (`x = df.drop("price", axis=1)`), but the UI only collects a single `area` input for prediction. If `houseprice.csv` has more than one feature column, the app will error out on `reg.predict([[area]])` since the model expects a matching number of features. Keep the dataset to a single `area` feature column, or extend the UI to collect all feature values used in training.

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_05_House_Price_Prediction   # adjust folder name if different

# install dependencies
pip install streamlit pandas scikit-learn

# make sure houseprice.csv is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_05_House_Price_Prediction/
├── app.py              # Streamlit app (this project)
├── houseprice.csv       # Dataset
└── requirements.txt     # Python dependencies
```

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

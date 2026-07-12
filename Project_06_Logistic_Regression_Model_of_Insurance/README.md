Live Preview:- https://aiml-internship-projects-2blhm9rdkcgqec8ekkum89.streamlit.app/

# 🛡️ Insurance Prediction using Logistic Regression

A simple ML demo that predicts whether someone is likely to buy insurance, based only on their **age** — enter an age, click a button, and get a Yes/No prediction from a logistic regression model.

---

## ✨ Features

- **Loads and displays** the bundled `insurance_data.csv` dataset in a table.
- **Age vs. Insurance Uptake scatter chart** — quick visual of the raw relationship in the data.
- **Trains a `scikit-learn` `LogisticRegression`** model (`bought_insurance ~ age`) on app load, and reports training accuracy.
- **Interactive prediction** — enter an age (3–60) and click **"Suggestion"** to get a "Yes, you need to take insurance" / "No, you don't need to take insurance" result.
- **Educational section** — explains what logistic regression is (classification vs. regression) and shows the sigmoid function formula rendered with LaTeX.
- **Custom-styled buttons/success boxes** and a header image (`insurance.jpg`).
- **About the Developer** expander in the sidebar.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pandas](https://pandas.pydata.org/) — data loading
- [scikit-learn](https://scikit-learn.org/) — `LogisticRegression` model

---

## 📂 Required Files

Both files must sit in the same folder as `app.py`:

| File | Purpose |
|---|---|
| `insurance_data.csv` | Training data, with `age` and `bought_insurance` (0/1) columns |
| `insurance.jpg` | Header image displayed next to the title, and used as the browser tab icon |

```
age,bought_insurance
22,0
25,0
47,1
52,1
...
```

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_06_Insurance_Prediction   # adjust folder name if different

# install dependencies
pip install streamlit pandas scikit-learn

# make sure insurance_data.csv and insurance.jpg are in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_06_Insurance_Prediction/
├── app.py               # Streamlit app (this project)
├── insurance_data.csv   # Training dataset
├── insurance.jpg         # Header image / page icon
└── requirements.txt      # Python dependencies
```

---

## 📖 The Method: Logistic Regression

Unlike linear regression (which predicts continuous numbers like prices), **logistic regression** predicts a category — here, whether someone bought insurance (`1`) or didn't (`0`) — by passing a linear combination of inputs through the sigmoid function:

```
f(x) = 1 / (1 + e^(-x))
```

This squashes any input into a value between 0 and 1, interpreted as the probability of the positive class.

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

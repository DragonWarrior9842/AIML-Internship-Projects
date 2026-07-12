Live Preview:- https://aiml-internship-projects-mgdyfrta4ftjpaewv27rcy.streamlit.app/

# 💼 HR Employee Retention Predictor

An interactive exploratory analysis and logistic regression model predicting whether an employee is likely to leave the company, built on the HR Analytics dataset from Kaggle — walking through EDA, feature selection, model training, evaluation, and live "what-if" predictions for a hypothetical employee.

---

## ✨ Features

### Step 1 — Data Exploration
- Loads `HR_comma_sep.csv` (or your own upload) and summarizes total / left / retained employee counts.
- Groups all columns by attrition (`left`) to compare averages, surfacing patterns like:
  - Lower **satisfaction level** among employees who left
  - Higher **average monthly hours** among employees who left
  - Employees **promoted in the last 5 years** are more likely to stay
- Interactive Plotly histograms: **salary vs. retention** and **department vs. retention**.

### Step 2 — Feature Selection & Encoding
Uses `satisfaction_level`, `average_montly_hours`, `promotion_last_5years`, and one-hot encoded `salary` as model inputs (department is excluded, matching the original exercise's approach since it showed only a minor effect).

### Step 3 — Model Training & Evaluation
- Adjustable **test set size** and **random seed** sliders.
- Trains a `scikit-learn` `LogisticRegression` model, cached for performance.
- Reports train/test row counts and **test accuracy**.
- **Confusion matrix** heatmap (Plotly).
- **Model coefficients** bar chart, showing which features push predictions toward "left" vs. "stayed."

### Step 4 — Predict for a Hypothetical Employee
Adjust satisfaction level, average monthly hours, promotion status, and salary level via sliders/selectors to get a live "Likely to LEAVE / STAY" prediction with probability.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — app framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [scikit-learn](https://scikit-learn.org/) — `LogisticRegression`, train/test split, accuracy & confusion matrix
- [Plotly Express](https://plotly.com/python/plotly-express/) — interactive charts

---

## 📂 Dataset

The app expects a CSV named **`HR_comma_sep.csv`** in the same folder as `app.py` (or uploaded via the sidebar) — the standard Kaggle **HR Analytics** dataset, with columns including:

`satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, left, promotion_last_5years, Department, salary`

---

## 🚀 Running Locally

```bash
# clone the repo
git clone https://github.com/DragonWarrior9842/AIML-Internship-Projects.git
cd AIML-Internship-Projects/Project_07_HR_Employee_Retention   # adjust folder name if different

# install dependencies
pip install streamlit pandas numpy scikit-learn plotly

# make sure HR_comma_sep.csv is in this folder, then run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
Project_07_HR_Employee_Retention/
├── app.py                # Streamlit app (this project)
├── HR_comma_sep.csv      # Dataset (upload via sidebar if not present)
└── requirements.txt      # Python dependencies
```

---

## 👨‍💻 About the Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

📧 [Email](mailto:aasblko@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/) · 🐙 [GitHub](https://github.com/DragonWarrior9842) · 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Employee Retention Predictor",
    page_icon=":briefcase:",
    layout="wide",
)

st.title(":briefcase: HR Employee Retention Predictor")
st.caption(
    "Exploratory analysis and a logistic regression model predicting "
    "whether an employee is likely to leave the company, based on the "
    "HR Analytics dataset from Kaggle."
)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(APP_DIR, "HR_comma_sep.csv")


@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    source = uploaded if uploaded is not None else DEFAULT_PATH
    try:
        df = load_data(source)
        if uploaded is None:
            st.caption(f"Using bundled dataset: `{os.path.basename(DEFAULT_PATH)}`")
    except FileNotFoundError:
        st.error(
            f"Couldn't find HR_comma_sep.csv at:\n\n`{DEFAULT_PATH}`\n\n"
            "Make sure the CSV is in the same folder as app.py, "
            "or upload a CSV above to continue."
        )
        st.stop()

st.success(f"Loaded **{df.shape[0]:,} employees** and **{df.shape[1]} columns**.")

with st.expander("Preview raw data", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------------------------------
# Exploration
# --------------------------------------------------------------------------
st.subheader("Step 1 - Data exploration")

left_df = df[df.left == 1]
retained_df = df[df.left == 0]

e1, e2, e3 = st.columns(3)
e1.metric("Total employees", f"{df.shape[0]:,}")
e2.metric("Left the company", f"{left_df.shape[0]:,}")
e3.metric("Retained", f"{retained_df.shape[0]:,}")

st.write("**Average numbers for all columns, grouped by attrition (`left`):**")
st.dataframe(df.groupby("left").mean(numeric_only=True), use_container_width=True)

st.markdown(
    """
    From this table, a few patterns tend to stand out:
    - **Satisfaction level** is noticeably lower for employees who left.
    - **Average monthly hours** tend to be higher for employees who left.
    - Employees who were **promoted in the last 5 years** are more likely to stay.
    """
)

ec1, ec2 = st.columns(2)
with ec1:
    fig_salary = px.histogram(
        df, x="salary", color="left", barmode="group",
        category_orders={"salary": ["low", "medium", "high"]},
        title="Impact of salary on employee retention",
        labels={"left": "Left (1) / Retained (0)"},
    )
    st.plotly_chart(fig_salary, use_container_width=True)

with ec2:
    fig_dept = px.histogram(
        df, x="Department", color="left", barmode="group",
        title="Department-wise employee retention",
        labels={"left": "Left (1) / Retained (0)"},
    )
    fig_dept.update_xaxes(tickangle=45)
    st.plotly_chart(fig_dept, use_container_width=True)

st.caption(
    "Employees on higher salaries are less likely to leave. Department shows "
    "some effect but not a dominant one, so it's left out of the model below "
    "(matching the original exercise's approach)."
)

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
st.subheader("Step 2 - Feature selection & encoding")

st.write(
    "Based on the exploration above, the model uses these independent "
    "variables: **satisfaction_level**, **average_montly_hours**, "
    "**promotion_last_5years**, and **salary** (one-hot encoded)."
)

FEATURES = ["satisfaction_level", "average_montly_hours", "promotion_last_5years", "salary"]
subdf = df[FEATURES].copy()

salary_dummies = pd.get_dummies(subdf.salary, prefix="salary", drop_first=True)
X = pd.concat([subdf.drop("salary", axis="columns"), salary_dummies], axis="columns")
y = df["left"]

with st.expander("Preview encoded features (X)", expanded=False):
    st.dataframe(X.head(10), use_container_width=True)

# --------------------------------------------------------------------------
# Train/test split & model training
# --------------------------------------------------------------------------
st.subheader("Step 3 - Train the model")

test_size = st.slider("Test set size", 0.1, 0.5, 0.2, step=0.05)
random_state = st.number_input("Random seed", value=42, step=1)


@st.cache_resource
def train_model(X_data, y_data, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


model, X_train, X_test, y_train, y_test = train_model(X, y, test_size, random_state)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("Train rows", f"{X_train.shape[0]:,}")
m2.metric("Test rows", f"{X_test.shape[0]:,}")
m3.metric("Test accuracy", f"{accuracy:.2%}")

cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=["Predicted: Retained", "Predicted: Left"],
    y=["Actual: Retained", "Actual: Left"],
    color_continuous_scale="Blues",
    title="Confusion matrix",
)
st.plotly_chart(fig_cm, use_container_width=True)

coef_df = pd.DataFrame(
    {"feature": X.columns, "coefficient": model.coef_[0]}
).sort_values("coefficient")
fig_coef = px.bar(
    coef_df, x="coefficient", y="feature", orientation="h",
    title="Model coefficients (logistic regression)",
)
st.plotly_chart(fig_coef, use_container_width=True)

# --------------------------------------------------------------------------
# Predict for a hypothetical employee
# --------------------------------------------------------------------------
st.subheader("Step 4 - Predict for a hypothetical employee")

p1, p2, p3, p4 = st.columns(4)
with p1:
    satisfaction_level = st.slider("Satisfaction level", 0.0, 1.0, 0.5, step=0.01)
with p2:
    average_montly_hours = st.slider("Average monthly hours", 50, 350, 200, step=1)
with p3:
    promotion_last_5years = st.selectbox(
        "Promoted in last 5 years?", options=[0, 1], format_func=lambda x: "Yes" if x else "No"
    )
with p4:
    salary = st.selectbox("Salary level", options=["low", "medium", "high"])

input_row = {
    "satisfaction_level": satisfaction_level,
    "average_montly_hours": average_montly_hours,
    "promotion_last_5years": promotion_last_5years,
}
for col in [c for c in X.columns if c.startswith("salary_")]:
    input_row[col] = 1 if col == f"salary_{salary}" else 0

input_df = pd.DataFrame([input_row])[X.columns]

pred_class = model.predict(input_df)[0]
pred_proba = model.predict_proba(input_df)[0]

r1, r2 = st.columns(2)
r1.metric(
    "Prediction",
    "Likely to LEAVE" if pred_class == 1 else "Likely to STAY",
)
r2.metric("Probability of leaving", f"{pred_proba[1]:.1%}")

st.progress(float(pred_proba[1]))

st.caption(
    "Model: scikit-learn LogisticRegression, trained on satisfaction_level, "
    "average_montly_hours, promotion_last_5years, and one-hot encoded salary. "
    "Built with Streamlit."
)
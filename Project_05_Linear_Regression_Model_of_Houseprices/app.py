import pandas as pd
import streamlit as sl
from sklearn.linear_model import LinearRegression
import os

sl.set_page_config(page_title="House Price Prediction using Linear Regression", layout='centered')

# --------------------------------------------------------------------------
# Developer credit - small floating badge, top-right corner
# --------------------------------------------------------------------------
sl.markdown(
    """
    <style>
    .dev-badge {
        position: fixed;
        top: 12px;
        right: 16px;
        background-color: #1a1c24;
        border: 1px solid #333844;
        border-radius: 999px;
        padding: 6px 14px;
        font-size: 0.78rem;
        color: #cfcfcf;
        z-index: 999;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    .dev-badge a {
        color: #cfcfcf;
        text-decoration: none;
        margin-left: 8px;
    }
    .dev-badge a:hover {
        color: #4C7DF5;
    }
    </style>

    <div class="dev-badge">
        👨‍💻 Aditya Agarwal
        <a href="mailto:aasblko@gmail.com" title="Email">📧</a>
        <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank" title="LinkedIn">💼</a>
        <a href="https://github.com/DragonWarrior9842" target="_blank" title="GitHub">🐙</a>
        <a href="https://www.instagram.com/adityaagarwal67/" target="_blank" title="Instagram">🌐</a>
    </div>
    """,
    unsafe_allow_html=True,
)

sl.title("House Price Prediction using Linear Regression")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "houseprice.csv"))

sl.subheader("Data of the Prices is given below")
sl.dataframe(df)

x = df.drop("price", axis=1)
y = df["price"]
reg = LinearRegression()
reg.fit(x, y)

sl.subheader("Enter the input for Prediction of the Price")
area = sl.number_input("Enter the Area of the House", min_value=100, max_value=10000, value=5000, step=50)
if sl.button("Predicted Price"):
    prediction = reg.predict([[area]])
    sl.success(f"The Predicted Price of the House is {prediction[0]:.2f}")

sl.subheader("Model Details")
sl.write("Regression Coefficient: ", reg.coef_[0])
sl.write("Regression Intercept: ", reg.intercept_)
sl.write("R-squared: ", reg.score(x, y))

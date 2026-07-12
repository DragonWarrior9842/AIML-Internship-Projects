import pandas as pd
import streamlit as sl
from sklearn.linear_model import LinearRegression
import os

sl.set_page_config(page_title="House Price Prediction using Linear Regression", layout='centered')

# --------------------------------------------------------------------------
# Developer credit - sidebar
# --------------------------------------------------------------------------
with sl.sidebar:
    sl.header("About the Developer")
    sl.write("**Aditya Agarwal**")
    sl.write("Data Science / ML Enthusiast")
    sl.write(
        "Currently a student of Shri Ramswaroop Memorial College of "
        "Engineering and Management, Lucknow, pursuing a Bachelor of "
        "Technology in Computer Science Engineering."
    )

    sl.markdown("---")

    sl.markdown("📧 [Email](mailto:aasblko@gmail.com)")
    sl.markdown("💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/)")
    sl.markdown("🐙 [GitHub](https://github.com/DragonWarrior9842)")
    sl.markdown("🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)")

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

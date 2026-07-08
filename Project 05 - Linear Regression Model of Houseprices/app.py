import pandas as pd
import streamlit as sl
from sklearn.linear_model import LinearRegression
sl.set_page_config(page_title="House Price Prediction using Linear Regression", layout='centered')
sl.title("House Price Prediction using Linear Regression")
df=pd.read_csv("houseprice.csv")
sl.subheader("Data of the Prices is given below")
sl.dataframe(df)
x=df.drop("price", axis=1)
y=df["price"]
reg=LinearRegression()
reg.fit(x,y)
sl.subheader("Enter the input for Prediction of the Price")
area=sl.number_input("Enter the Area of the House", min_value=100, max_value=10000, value=5000, step=50)
if sl.button("Predicted Price"):
  prediction=reg.predict([[area]])
  sl.success(f"The Predicted Price of the House is {prediction[0]:.2f}")
sl.subheader("Model Details")
sl.write("Regression Coefficient: ", reg.coef_[0])
sl.write("Regression Intercept: ", reg.intercept_)
sl.write("R-squared: ", reg.score(x,y))
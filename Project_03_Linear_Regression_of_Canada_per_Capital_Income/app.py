import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Canada Per Capita Income Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# --------------------------------------------------------------------------
# Developer credit - popup dialog, opened from a button in the top corner
# --------------------------------------------------------------------------
@st.dialog("About the Developer")
def show_developer_info():
    st.markdown("### Aditya Agarwal")
    st.write("Data Science / ML Enthusiast")
    st.write(
        "Currently a student of Shri Ramswaroop Memorial College of "
        "Engineering and Management, Lucknow, pursuing a Bachelor of "
        "Technology in Computer Science Engineering."
    )
    st.markdown("---")
    st.markdown("📧 [Email](mailto:aasblko@gmail.com)")
    st.markdown("💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/)")
    st.markdown("🐙 [GitHub](https://github.com/DragonWarrior9842)")
    st.markdown("🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)")


title_col, button_col = st.columns([6, 1])
with title_col:
    st.title(":chart_with_upwards_trend: Canada Per Capita Income Predictor")
with button_col:
    st.write("")
    if st.button("👤 Developer"):
        show_developer_info()

st.caption(
    "A simple linear regression model trained on historical per-capita "
    "income data, used to predict future values (e.g. year 2020)."
)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(APP_DIR, "canada_per_capita_income.csv")


@st.cache_data
def load_data(file) -> pd.DataFrame:
    data = pd.read_csv(file)
    data.columns = ["year", "income"]
    return data.sort_values("year").reset_index(drop=True)


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
            f"Couldn't find canada_per_capita_income.csv at:\n\n`{DEFAULT_PATH}`\n\n"
            "Make sure the CSV is in the same folder as app.py, "
            "or upload a CSV above to continue."
        )
        st.stop()

st.success(
    f"Loaded **{df.shape[0]} years** of data, "
    f"from **{int(df.year.min())}** to **{int(df.year.max())}**."
)

with st.expander("Preview raw data", expanded=False):
    st.dataframe(df, use_container_width=True)


# --------------------------------------------------------------------------
# Train the model
# --------------------------------------------------------------------------
@st.cache_resource
def train_model(data: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(data[["year"]], data["income"])
    return model


model = train_model(df)

st.subheader("Step 1 - Model fit")

c1, c2, c3 = st.columns(3)
c1.metric("Slope (income / year)", f"{model.coef_[0]:,.2f}")
c2.metric("Intercept", f"{model.intercept_:,.2f}")
r2 = model.score(df[["year"]], df["income"])
c3.metric("R-squared", f"{r2:.4f}")

fig_fit = go.Figure()
fig_fit.add_trace(
    go.Scatter(
        x=df.year, y=df.income, mode="markers", name="Historical data",
        marker=dict(size=8),
    )
)
line_years = np.linspace(df.year.min(), df.year.max(), 100)
line_pred = model.predict(line_years.reshape(-1, 1))
fig_fit.add_trace(
    go.Scatter(x=line_years, y=line_pred, mode="lines", name="Regression line")
)
fig_fit.update_layout(
    title="Per Capita Income vs Year - Historical Data & Fitted Line",
    xaxis_title="Year",
    yaxis_title="Per capita income (US$)",
)
st.plotly_chart(fig_fit, use_container_width=True)

# --------------------------------------------------------------------------
# Predict for a chosen year
# --------------------------------------------------------------------------
st.subheader("Step 2 - Predict per capita income for a year")

default_year = 2020
year_input = st.number_input(
    "Year to predict",
    min_value=int(df.year.min()),
    max_value=int(df.year.max()) + 100,
    value=default_year,
    step=1,
)

prediction = model.predict([[year_input]])[0]

st.metric(
    label=f"Predicted per capita income for {int(year_input)}",
    value=f"US$ {prediction:,.2f}",
)

if year_input > df.year.max():
    st.info(
        f"{int(year_input)} is beyond the last year in the training data "
        f"({int(df.year.max())}), so this is an extrapolation from the "
        "linear trend, not an interpolation."
    )

fig_pred = go.Figure()
fig_pred.add_trace(
    go.Scatter(
        x=df.year, y=df.income, mode="markers", name="Historical data",
        marker=dict(size=8),
    )
)
extended_years = np.linspace(df.year.min(), max(df.year.max(), year_input), 100)
extended_pred = model.predict(extended_years.reshape(-1, 1))
fig_pred.add_trace(
    go.Scatter(x=extended_years, y=extended_pred, mode="lines", name="Regression line")
)
fig_pred.add_trace(
    go.Scatter(
        x=[year_input], y=[prediction], mode="markers", name=f"Prediction ({int(year_input)})",
        marker=dict(size=14, symbol="star", color="red"),
    )
)
fig_pred.update_layout(
    title=f"Prediction for {int(year_input)}",
    xaxis_title="Year",
    yaxis_title="Per capita income (US$)",
)
st.plotly_chart(fig_pred, use_container_width=True)

# --------------------------------------------------------------------------
# Batch predictions
# --------------------------------------------------------------------------
st.subheader("Step 3 - Batch predictions for a range of years")

r1, r2 = st.columns(2)
with r1:
    start_year = st.number_input("From year", value=int(df.year.max()) + 1, step=1)
with r2:
    end_year = st.number_input("To year", value=int(df.year.max()) + 10, step=1)

if start_year > end_year:
    st.error("'From year' must be less than or equal to 'To year'.")
else:
    years_range = np.arange(start_year, end_year + 1)
    preds_range = model.predict(years_range.reshape(-1, 1))
    result_df = pd.DataFrame(
        {"year": years_range, "predicted_income": np.round(preds_range, 2)}
    )
    st.dataframe(result_df, use_container_width=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions CSV",
        data=csv_bytes,
        file_name="income_predictions.csv",
        mime="text/csv",
    )

st.caption(
    "Model: scikit-learn LinearRegression, trained on year -> per capita income. "
    "Built with Streamlit."
)

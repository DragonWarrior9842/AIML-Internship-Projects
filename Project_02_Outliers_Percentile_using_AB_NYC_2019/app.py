import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import os
st.set_page_config(
    page_title="NYC Airbnb - Outlier Detection",
    page_icon=":cityscape:",
    layout="wide",
)
st.title(":cityscape: NYC Airbnb - Percentile-Based Outlier Detection")
st.caption(
    "An interactive walkthrough of removing outliers from a numeric column "
    "using the percentile method, based on the AB_NYC_2019 dataset."
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(BASE_DIR, "AB_NYC_2019.csv")
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    source = uploaded if uploaded is not None else DEFAULT_PATH
    try:
        df = load_data(source)
    except FileNotFoundError:
        st.error(
            f"Couldn't find `{DEFAULT_PATH}` next to app.py. "
            "Upload a CSV above to continue."
        )
        st.stop()
st.success(f"Loaded **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.")
with st.expander("Preview raw data", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_col = "price" if "price" in numeric_cols else numeric_cols[0]
with st.sidebar:
    st.header("2. Column")
    column = st.selectbox(
        "Numeric column to clean",
        options=numeric_cols,
        index=numeric_cols.index(default_col),
    )
col1, col2 = st.columns([1, 2])
with col1:
    st.write(df[column].describe())
with col2:
    fig_before = px.histogram(
        df, x=column, nbins=50, title=f"Distribution of {column} (raw)"
    )
    st.plotly_chart(fig_before, use_container_width=True)
fig_box_before = px.box(df, x=column, title=f"Boxplot of {column} (raw)")
st.plotly_chart(fig_box_before, use_container_width=True)
st.write(
    "The percentile method removes the most extreme low and high values "
    "by cutting off a small percentage from each tail of the distribution."
)
pcol1, pcol2 = st.columns(2)
with pcol1:
    low_pct = st.slider("Lower percentile cutoff", 0.0, 10.0, 1.0, step=0.1) / 100
with pcol2:
    high_pct = st.slider("Upper percentile cutoff", 90.0, 100.0, 99.9, step=0.1) / 100
if low_pct >= high_pct:
    st.error("Lower percentile must be smaller than the upper percentile.")
    st.stop()
min_threshold, max_threshold = df[column].quantile([low_pct, high_pct])
m1, m2 = st.columns(2)
m1.metric(f"Min threshold ({low_pct:.3%})", f"{min_threshold:,.2f}")
m2.metric(f"Max threshold ({high_pct:.3%})", f"{max_threshold:,.2f}")
with st.expander(f"Rows below the min threshold ({column} < {min_threshold:,.2f})"):
    st.dataframe(df[df[column] < min_threshold], use_container_width=True)
with st.expander(f"Rows above the max threshold ({column} > {max_threshold:,.2f})"):
    st.dataframe(df[df[column] > max_threshold], use_container_width=True)
df2 = df[(df[column] > min_threshold) & (df[column] < max_threshold)]
r1, r2, r3 = st.columns(3)
r1.metric("Rows before", f"{df.shape[0]:,}")
r2.metric("Rows after", f"{df2.shape[0]:,}")
r3.metric("Rows removed", f"{df.shape[0] - df2.shape[0]:,}")
st.write("Random sample from the cleaned data:")
st.dataframe(df2.sample(min(5, len(df2))), use_container_width=True)
st.write(f"`{column}` summary statistics - before vs. after:")
compare = pd.concat(
    [df[column].describe(), df2[column].describe()],
    axis=1,
    keys=["before", "after"],
)
st.dataframe(compare, use_container_width=True)
vcol1, vcol2 = st.columns(2)
with vcol1:
    fig_after_hist = px.histogram(
        df2, x=column, nbins=50, title=f"Distribution of {column} (cleaned)"
    )
    st.plotly_chart(fig_after_hist, use_container_width=True)
with vcol2:
    fig_after_box = px.box(df2, x=column, title=f"Boxplot of {column} (cleaned)")
    st.plotly_chart(fig_after_box, use_container_width=True)
csv_bytes = df2.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download cleaned CSV",
    data=csv_bytes,
    file_name="AB_NYC_2019_cleaned.csv",
    mime="text/csv",
)
st.caption(
    "Built with Streamlit - Percentile-based outlier removal on the "
    "AB_NYC_2019 (New York City Airbnb Open Data) dataset."
)
st.markdown(
    """
    <style>
    .dev-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        border-top: 1px solid #262730;
        padding: 8px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: #b0b0b0;
        z-index: 999;
    }
    .dev-footer a {
        color: #b0b0b0;
        text-decoration: none;
        margin-left: 14px;
    }
    .dev-footer a:hover {
        color: #4C7DF5;
    }
    .block-container {
        padding-bottom: 4rem;
    }
    </style>

    <div class="dev-footer">
        <div>Built by <strong>Aditya Agarwal</strong> &middot; Data Science / ML Enthusiast</div>
        <div>
            <a href="mailto:aasblko@gmail.com">Email</a>
            <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank">LinkedIn</a>
            <a href="https://github.com/DragonWarrior9842" target="_blank">GitHub</a>
            <a href="https://www.instagram.com/adityaagarwal67/" target="_blank">Instagram</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

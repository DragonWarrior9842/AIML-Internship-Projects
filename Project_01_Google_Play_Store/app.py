import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
st.set_page_config(
    page_title="Google Play Store Case Study",
    page_icon="📱",
    layout="wide",
)
sns.set_style("whitegrid")
def clean_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return np.nan
@st.cache_data
def load_and_clean(file):
    log = []
    inp0 = pd.read_csv(file)
    log.append(f"Raw data loaded: {inp0.shape[0]} rows, {inp0.shape[1]} columns")
    inp1 = inp0[~inp0.Rating.isnull()].copy()
    log.append(f"Dropped rows with null Rating -> {inp1.shape[0]} rows")
    shifted_mask = inp1['Android Ver'].isnull() & (inp1.Category == "1.9")
    inp1 = inp1[~shifted_mask]
    log.append(f"Dropped shifted/garbage row(s) -> {inp1.shape[0]} rows")
    inp1['Android Ver'] = inp1['Android Ver'].fillna(inp1['Android Ver'].mode()[0])
    inp1['Current Ver'] = inp1['Current Ver'].fillna(inp1['Current Ver'].mode()[0])
    inp1['Price'] = inp1['Price'].apply(clean_price)
    before = inp1.shape[0]
    inp1 = inp1[~inp1['Price'].isna()]
    if before - inp1.shape[0]:
        log.append(f"Dropped rows with unparseable Price -> removed {before - inp1.shape[0]} rows")
    inp1['Reviews'] = pd.to_numeric(inp1['Reviews'], errors="coerce")
    before = inp1.shape[0]
    inp1 = inp1[~inp1['Reviews'].isna()]
    if before - inp1.shape[0]:
        log.append(f"Dropped rows with unparseable Reviews -> removed {before - inp1.shape[0]} rows")
    inp1['Reviews'] = inp1['Reviews'].astype("int64")
    def clean_installs(val):
        try:
            return int(str(val).replace(",", "").replace("+", "").strip())
        except ValueError:
            return np.nan
    inp1['Installs'] = inp1['Installs'].apply(clean_installs)
    before = inp1.shape[0]
    inp1 = inp1[~inp1['Installs'].isna()]
    if before - inp1.shape[0]:
        log.append(f"Dropped rows with unparseable Installs -> removed {before - inp1.shape[0]} rows")
    inp1['Installs'] = inp1['Installs'].astype("int64")
    before = inp1.shape[0]
    inp1 = inp1[inp1.Reviews <= inp1.Installs]
    log.append(f"Sanity check (Reviews <= Installs): removed {before - inp1.shape[0]} rows")
    bad_free = inp1[(inp1.Type == "Free") & (inp1.Price > 0)]
    if len(bad_free):
        inp1 = inp1[~((inp1.Type == "Free") & (inp1.Price > 0))]
        log.append(f"Sanity check (Free apps priced > 0): removed {len(bad_free)} rows")
    before = inp1.shape[0]
    inp1 = inp1[inp1.Price < 200]
    inp1 = inp1[inp1.Price <= 30]
    log.append(f"Price outliers removed (kept Price <= 30): removed {before - inp1.shape[0]} rows")
    before = inp1.shape[0]
    inp1 = inp1[inp1.Reviews <= 1000000]
    log.append(f"Review outliers removed (kept Reviews <= 1,000,000): removed {before - inp1.shape[0]} rows")
    before = inp1.shape[0]
    inp1 = inp1[inp1.Installs <= 100000000]
    log.append(f"Install outliers removed (kept Installs <= 100,000,000): removed {before - inp1.shape[0]} rows")
    before = inp1.shape[0]
    inp1 = inp1[~inp1['Content Rating'].isin(["Adults only 18+", "Unrated"])]
    log.append(f"Dropped rare Content Rating categories: removed {before - inp1.shape[0]} rows")
    inp1.reset_index(inplace=True, drop=True)
    inp1['updated_month'] = pd.to_datetime(inp1['Last Updated'], errors="coerce").dt.month
    inp1['Size_Bucket'] = pd.qcut(inp1['Size'], [0, 0.2, 0.4, 0.6, 0.8, 1],
                                   labels=["VL", "L", "M", "H", "VH"])
    log.append(f"Final cleaned dataset: {inp1.shape[0]} rows, {inp1.shape[1]} columns")
    return inp0, inp1, log
st.sidebar.title("📱 Google Play Store")
st.sidebar.markdown("Data Visualisation Case Study")
CSV_NAME = "googleplaystore_v2.csv"
def find_csv(filename):
    candidate_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd(),
    ]
    for d in candidate_dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    for d in candidate_dirs:
        for root, _, files in os.walk(d):
            if filename in files:
                return os.path.join(root, filename)
    return None
uploaded = st.sidebar.file_uploader(f"Upload {CSV_NAME}", type=["csv"])
if uploaded is not None:
    data_file = uploaded
else:
    data_file = find_csv(CSV_NAME)
if data_file is None:
    st.error(
        f"Could not find `{CSV_NAME}` next to the app (looked in "
        f"`{os.path.dirname(os.path.abspath(__file__))}` and `{os.getcwd()}`). "
        "Upload the file using the sidebar to continue."
    )
    st.stop()
try:
    raw_df, df, cleaning_log = load_and_clean(data_file)
except FileNotFoundError:
    st.error(f"Could not read `{CSV_NAME}`. Upload the file using the sidebar to continue.")
    st.stop()
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
categories = sorted(df['Category'].dropna().unique().tolist())
sel_categories = st.sidebar.multiselect("Category", categories, default=[])
content_ratings = sorted(df['Content Rating'].dropna().unique().tolist())
sel_content = st.sidebar.multiselect("Content Rating", content_ratings, default=[])
app_type = st.sidebar.radio("Type", ["All", "Free", "Paid"], horizontal=True)
filtered = df.copy()
if sel_categories:
    filtered = filtered[filtered['Category'].isin(sel_categories)]
if sel_content:
    filtered = filtered[filtered['Content Rating'].isin(sel_content)]
if app_type != "All":
    filtered = filtered[filtered['Type'] == app_type]
st.sidebar.markdown(f"**{len(filtered):,}** apps match current filters")
st.title("📱 Google Play Store — Data Visualisation Case Study")
st.markdown(
    """
The Google Play Store team wants to boost visibility for the most promising apps.
This dashboard walks through the full analysis: **data cleaning**, **sanity checks**,
**outlier treatment**, and a wide range of **visualisations** (matplotlib, seaborn & plotly)
used to explore what separates a well-performing app from the rest.
"""
)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Raw rows", f"{raw_df.shape[0]:,}")
col2.metric("Cleaned rows", f"{df.shape[0]:,}")
col3.metric("Filtered rows", f"{len(filtered):,}")
col4.metric("Avg. Rating (filtered)", f"{filtered['Rating'].mean():.2f}" if len(filtered) else "—")
tab_overview, tab_cleaning, tab_dist, tab_cat, tab_scatter, tab_heat, tab_time, tab_data = st.tabs(
    [
        "🏠 Overview",
        "🧹 Data Cleaning",
        "📊 Distributions",
        "🧩 Categorical Charts",
        "🔵 Scatter / Pair Plots",
        "🌡️ Heat Map",
        "📈 Time & Stacked Charts",
        "🗂️ Raw Data",
    ]
)
with tab_overview:
    st.subheader("Problem Statement")
    st.markdown(
        """
- Does a higher **size** or **price** necessarily mean an app performs better than others?
- Does a higher number of **installs** give a clear picture of which app will have a better rating?
        """
    )
    st.subheader("Sample of the cleaned data")
    st.dataframe(filtered.head(10), use_container_width=True)
    st.subheader("Quick summary statistics")
    st.dataframe(
        filtered[['Rating', 'Reviews', 'Size', 'Installs', 'Price']].describe(),
        use_container_width=True,
    )
with tab_cleaning:
    st.subheader("Cleaning & Sanity-Check Pipeline")
    st.markdown(
        """
Steps performed on the raw dataset (mirrors the case-study notebook):

1. Drop rows with a null **Rating** (the target variable).
2. Drop the row with shifted/garbage values (`Category == "1.9"`).
3. Impute missing **Android Ver** / **Current Ver** with the mode.
4. Fix data types: **Price** → float, **Reviews** → int, **Installs** → int.
5. Sanity check: remove rows where `Reviews > Installs`.
6. Sanity check: remove Free apps with `Price > 0`.
7. Remove extreme **Price** outliers (`Price <= 30`).
8. Remove extreme **Reviews** outliers (`Reviews <= 1,000,000`).
9. Remove extreme **Installs** outliers (`Installs <= 100,000,000`).
10. Drop under-represented **Content Rating** categories (`Adults only 18+`, `Unrated`).
        """
    )
    for line in cleaning_log:
        st.markdown(f"- {line}")
    st.subheader("Outlier detection — Boxplots (matplotlib)")
    b1, b2, b3 = st.columns(3)
    with b1:
        fig, ax = plt.subplots()
        raw_price = raw_df['Price'].apply(clean_price).dropna()
        ax.boxplot(raw_price)
        ax.set_title("Price (raw)")
        st.pyplot(fig)
    with b2:
        fig, ax = plt.subplots()
        ax.boxplot(df['Reviews'])
        ax.set_title("Reviews (cleaned)")
        st.pyplot(fig)
    with b3:
        fig, ax = plt.subplots()
        ax.boxplot(df['Installs'])
        ax.set_title("Installs (cleaned)")
        st.pyplot(fig)
with tab_dist:
    st.subheader("Histograms")
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        fig, ax = plt.subplots()
        ax.hist(filtered['Reviews'], bins=30, color="#4C72B0")
        ax.set_title("Distribution of Reviews")
        ax.set_xlabel("Reviews")
        st.pyplot(fig)
    with hcol2:
        fig, ax = plt.subplots()
        ax.hist(filtered['Size'].dropna(), bins=30, color="#55A868")
        ax.set_title("Distribution of Size")
        ax.set_xlabel("Size (KB)")
        st.pyplot(fig)
    st.subheader("Distribution Plot (Seaborn) — Rating")
    d1, d2 = st.columns([1, 3])
    with d1:
        bins = st.slider("Number of bins", 5, 50, 20)
        color = st.color_picker("Plot colour", "#2E7D32")
        style = st.selectbox("Seaborn style", ["white", "dark", "whitegrid", "darkgrid", "ticks"])
    with d2:
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(filtered['Rating'], bins=bins, kde=True, color=color, ax=ax)
        ax.set_title("Distribution of app ratings", fontsize=12)
        st.pyplot(fig)
        sns.set_style("whitegrid")
with tab_cat:
    st.subheader("Content Rating — Pie & Bar Charts")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        filtered['Content Rating'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title("Share of apps by Content Rating")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        filtered['Content Rating'].value_counts().plot.bar(ax=ax, color="#4C72B0")
        ax.set_title("Count of apps by Content Rating")
        st.pyplot(fig)
    st.subheader("Average / Median Rating by Content Rating (Seaborn barplot)")
    estimator_choice = st.selectbox(
        "Aggregation", ["Mean", "Median", "Min", "5th percentile"]
    )
    est_map = {
        "Mean": np.mean,
        "Median": np.median,
        "Min": np.min,
        "5th percentile": lambda x: np.quantile(x, 0.05),
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=filtered, x="Content Rating", y="Rating",
                estimator=est_map[estimator_choice], ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig)
    st.subheader("Boxplot — Rating across Content Rating")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=filtered, x="Content Rating", y="Rating", ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig)
    st.subheader("Boxplot — Rating across top Genres")
    top_genres = df['Genres'].value_counts().head(10).index.tolist()
    sel_genres = st.multiselect("Pick genres to compare", top_genres, default=top_genres[:4])
    if sel_genres:
        subset = filtered[filtered['Genres'].isin(sel_genres)]
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=subset, x="Genres", y="Rating", ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig)
    else:
        st.info("Select at least one genre to display the boxplot.")
with tab_scatter:
    st.subheader("Scatter Plot — Size vs Rating")
    fig, ax = plt.subplots()
    ax.scatter(filtered['Size'], filtered['Rating'], alpha=0.4, s=15)
    ax.set_xlabel("Size")
    ax.set_ylabel("Rating")
    st.pyplot(fig)
    st.subheader("Joint Plot")
    jc1, jc2 = st.columns(2)
    with jc1:
        x_var = st.selectbox("X variable", ["Size", "Price", "Reviews", "Installs"], index=0)
    with jc2:
        kind = st.selectbox("Kind", ["scatter", "reg", "hex", "kde"])
    only_paid = st.checkbox("Only paid apps (Price > 0)", value=(x_var == "Price"))
    jp_data = filtered[filtered['Price'] > 0] if only_paid else filtered
    if len(jp_data) > 2:
        g = sns.jointplot(data=jp_data, x=x_var, y="Rating", kind=kind, height=6)
        st.pyplot(g.fig)
    else:
        st.info("Not enough data points for the current filter/selection.")
    st.subheader("Pair Plot — Reviews, Size, Price, Rating")
    if st.checkbox("Generate pair plot (can be slow on large filtered sets)"):
        sample = filtered[['Reviews', 'Size', 'Price', 'Rating']].dropna()
        if len(sample) > 3000:
            sample = sample.sample(3000, random_state=42)
            st.caption("Sampled to 3,000 rows for performance.")
        g = sns.pairplot(sample)
        st.pyplot(g.fig)
with tab_heat:
    st.subheader("Pivot Table Heat Map — Rating by Content Rating & Size Bucket")
    agg_choice = st.selectbox(
        "Aggregation function", ["Mean", "Median", "Min", "Max", "20th percentile"]
    )
    agg_map = {
        "Mean": np.mean,
        "Median": np.median,
        "Min": np.min,
        "Max": np.max,
        "20th percentile": lambda x: np.quantile(x, 0.2),
    }
    pivot = pd.pivot_table(
        data=filtered, index="Content Rating", columns="Size_Bucket",
        values="Rating", aggfunc=agg_map[agg_choice],
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, cmap="Greens", annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.caption("Size_Bucket: VL = Very Low ... VH = Very High (quintiles of app Size).")
with tab_time:
    st.subheader("Average Rating by Month Last Updated (Line Plot)")
    monthly_rating = filtered.groupby("updated_month")["Rating"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(monthly_rating["updated_month"], monthly_rating["Rating"], marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)
    st.subheader("Interactive version (Plotly)")
    fig_px = px.line(
        monthly_rating, x="updated_month", y="Rating",
        title="Monthly average rating", markers=True,
    )
    st.plotly_chart(fig_px, use_container_width=True)
    st.subheader("Stacked Bar Chart — Installs by Month & Content Rating")
    monthly = pd.pivot_table(
        data=filtered, values="Installs", index="updated_month",
        columns="Content Rating", aggfunc="sum",
    ).fillna(0)
    view = st.radio("View", ["Absolute", "Proportion"], horizontal=True)
    if view == "Proportion":
        monthly_show = monthly.apply(lambda x: x / x.sum() if x.sum() else x, axis=1)
    else:
        monthly_show = monthly
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_show.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Installs" if view == "Absolute" else "Proportion of Installs")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)
with tab_data:
    st.subheader("Filtered Data")
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_playstore_data.csv",
        mime="text/csv",
    )
st.markdown("---")
st.caption("Built with Streamlit · Recreates the Google Play Store data-visualisation case study.")

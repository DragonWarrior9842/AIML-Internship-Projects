import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_income_data():
    return pd.read_csv(os.path.join(BASE_DIR, "income.csv"))

st.set_page_config(page_title="K-Means Clustering Tutorial", layout="wide")

st.title("📊 K-Means Clustering — Interactive Tutorial")
st.caption(
    "Based on the 'Introduction to K Means Clustering' tutorial: "
    "customer segmentation by age & income, plus the iris petal-clustering exercise."
)

tab1, tab2 = st.tabs(["👥 Customer Income Clustering", "🌸 Iris Petal Clustering (Exercise)"])


# =================================================================
# Helper
# =================================================================
def elbow_sse(X, max_k):
    sse = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        sse.append(km.inertia_)
    return sse


def plot_clusters(ax, df, x_col, y_col, cluster_col, centers, title):
    colors = plt.cm.tab10(np.linspace(0, 1, df[cluster_col].nunique()))
    for i, cid in enumerate(sorted(df[cluster_col].unique())):
        subset = df[df[cluster_col] == cid]
        ax.scatter(subset[x_col], subset[y_col], color=colors[i], label=f"Cluster {cid}", s=50)
    ax.scatter(centers[:, 0], centers[:, 1], color="purple", marker="*", s=300, label="Centroid")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend()


# =================================================================
# TAB 1: Customer Income Clustering
# =================================================================
with tab1:
    st.header("Customer Segmentation by Age & Income")


    df_raw = load_income_data()

    with st.expander("🔍 Show raw data"):
        st.dataframe(df_raw)

    fig0, ax0 = plt.subplots(figsize=(5, 4))
    ax0.scatter(df_raw["Age"], df_raw["Income($)"])
    ax0.set_xlabel("Age")
    ax0.set_ylabel("Income($)")
    ax0.set_title("Raw Age vs Income")
    st.pyplot(fig0)

    st.sidebar.header("Customer Clustering Settings")
    k_income = st.sidebar.slider("Customers: number of clusters (k)", 2, 9, 3, key="k_income")
    max_k_income = st.sidebar.slider("Customers: max k for elbow", 3, 15, 10, key="maxk_income")

    st.subheader("Raw vs. Scaled Clustering — Side by Side")
    st.markdown(
        "Age and Income are on very different scales (years vs. dollars), so "
        "**MinMax scaling** typically produces more sensible, evenly-sized clusters."
    )

    col_raw, col_scaled = st.columns(2)

    # ---- Raw clustering ----
    X_raw = df_raw[["Age", "Income($)"]].values.astype(float)
    km_raw = KMeans(n_clusters=k_income, n_init=10, random_state=42)
    df_raw_c = df_raw.copy()
    df_raw_c["cluster"] = km_raw.fit_predict(X_raw)

    with col_raw:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        plot_clusters(ax1, df_raw_c, "Age", "Income($)", "cluster",
                      km_raw.cluster_centers_, "Raw (unscaled)")
        st.pyplot(fig1)

    # ---- Scaled clustering ----
    scaler_age = MinMaxScaler()
    scaler_income = MinMaxScaler()
    df_scaled = df_raw.copy()
    df_scaled["Age"] = scaler_age.fit_transform(df_scaled[["Age"]])
    df_scaled["Income($)"] = scaler_income.fit_transform(df_scaled[["Income($)"]])

    X_scaled = df_scaled[["Age", "Income($)"]].values
    km_scaled = KMeans(n_clusters=k_income, n_init=10, random_state=42)
    df_scaled["cluster"] = km_scaled.fit_predict(X_scaled)

    with col_scaled:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        plot_clusters(ax2, df_scaled, "Age", "Income($)", "cluster",
                      km_scaled.cluster_centers_, "MinMax Scaled")
        st.pyplot(fig2)

    st.subheader("📉 Elbow Plot (on scaled data)")
    sse_income = elbow_sse(X_scaled, max_k_income)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(range(1, max_k_income + 1), sse_income, marker="o")
    ax3.axvline(x=k_income, color="red", linestyle="--", label=f"Selected k = {k_income}")
    ax3.set_xlabel("K")
    ax3.set_ylabel("Sum of Squared Error (SSE)")
    ax3.set_title("Elbow Method — Customer Data")
    ax3.legend()
    st.pyplot(fig3)

    st.caption("The elbow typically appears around k = 3 for this dataset.")

    with st.expander("📋 Clustered data table (scaled clustering)"):
        st.dataframe(df_scaled)

# =================================================================
# TAB 2: Iris Petal Clustering (Exercise)
# =================================================================
with tab2:
    st.header("Exercise: Cluster Iris Flowers by Petal Length & Width")
    st.markdown(
        """
        1. Cluster iris flowers using **petal length** and **petal width** only (sepal features dropped).
        2. Check whether **scaling** helps.
        3. Use the **elbow plot** to pick the optimal k.
        """
    )


    @st.cache_data
    def load_iris_data():
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["flower"] = iris.target
        df["flower_name"] = df["flower"].map({i: n for i, n in enumerate(iris.target_names)})
        return df


    df_iris_full = load_iris_data()
    df_iris = df_iris_full[["petal length (cm)", "petal width (cm)"]].copy()

    with st.expander("🔍 Show raw data (petal features only)"):
        st.dataframe(df_iris.head(10))

    st.sidebar.header("Iris Clustering Settings")
    use_scaling = st.sidebar.checkbox("Iris: apply MinMax Scaling", value=True)
    k_iris = st.sidebar.slider("Iris: number of clusters (k)", 2, 9, 3, key="k_iris")
    max_k_iris = st.sidebar.slider("Iris: max k for elbow", 3, 15, 10, key="maxk_iris")

    X_iris = df_iris.values.astype(float)
    if use_scaling:
        iris_scaler = MinMaxScaler()
        X_iris_proc = iris_scaler.fit_transform(X_iris)
    else:
        X_iris_proc = X_iris

    st.subheader("📉 Elbow Plot")
    sse_iris = elbow_sse(X_iris_proc, max_k_iris)
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(range(1, max_k_iris + 1), sse_iris, marker="o")
    ax4.axvline(x=k_iris, color="red", linestyle="--", label=f"Selected k = {k_iris}")
    ax4.set_xlabel("K")
    ax4.set_ylabel("Sum of Squared Error (SSE)")
    ax4.set_title("Elbow Method — Iris Petal Data")
    ax4.legend()
    st.pyplot(fig4)
    st.caption("For this dataset, the elbow typically appears around k = 3.")

    st.subheader(f"🎯 Clustering Result (k = {k_iris}, scaled = {use_scaling})")
    km_iris = KMeans(n_clusters=k_iris, n_init=10, random_state=42)
    y_pred_iris = km_iris.fit_predict(X_iris_proc)

    result_iris = df_iris.copy()
    result_iris["cluster"] = y_pred_iris

    centers = km_iris.cluster_centers_
    if use_scaling:
        centers = iris_scaler.inverse_transform(centers)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        plot_clusters(ax5, result_iris, "petal length (cm)", "petal width (cm)",
                      "cluster", centers, f"K-Means Clusters (k={k_iris})")
        st.pyplot(fig5)
    with col2:
        st.markdown("**Cluster sizes**")
        st.dataframe(result_iris["cluster"].value_counts().rename("count"))
        st.markdown("**Sample data**")
        st.dataframe(result_iris.head(10))

    with st.expander("🌼 Compare clusters to true species (ground truth, for reference)"):
        compare_df = df_iris_full[["petal length (cm)", "petal width (cm)", "flower_name"]].copy()
        compare_df["cluster"] = y_pred_iris
        st.dataframe(compare_df.groupby(["flower_name", "cluster"]).size().unstack(fill_value=0))

st.markdown("---")
st.caption("Built with Streamlit • scikit-learn KMeans • Based on the PierianData K-Means tutorial.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
st.set_page_config(page_title="K-Means: Iris Petal Clustering", layout="wide")
st.title("🌸 K-Means Clustering — Iris Petal Length & Width")
st.markdown(
    """
    <style>
    .dev-banner {
        background: linear-gradient(90deg, #4C7DF5 0%, #8E5CF7 100%);
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        color: white;
    }
    .dev-banner .dev-name {
        font-weight: 700;
        font-size: 0.95rem;
    }
    .dev-banner .dev-role {
        font-size: 0.8rem;
        opacity: 0.9;
    }
    .dev-banner a {
        color: white;
        text-decoration: none;
        margin-left: 12px;
        font-size: 0.85rem;
        opacity: 0.95;
    }
    .dev-banner a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="dev-banner">
        <div>
            <span class="dev-name">Aditya Agarwal</span> &nbsp;·&nbsp;
            <span class="dev-role">Data Science / ML Enthusiast</span>
        </div>
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
st.markdown(
    """
    This app reproduces the K-Means tutorial exercise:
    - Cluster iris flowers using **petal length** and **petal width** only.
    - Explore whether **scaling** (MinMaxScaler) changes the clustering.
    - Use the **elbow plot** to pick the optimal number of clusters (k).
    """
)
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["flower"] = iris.target
    df["flower_name"] = df["flower"].map(
        {i: name for i, name in enumerate(iris.target_names)}
    )
    return df
df_full = load_data()
df = df_full[["petal length (cm)", "petal width (cm)"]].copy()
with st.expander("🔍 Show raw data (petal features only)"):
    st.dataframe(df.head(10))
st.sidebar.header("Settings")
use_scaling = st.sidebar.checkbox(
    "Apply MinMax Scaling",
    value=True,
    help="Task 2: check if preprocessing/scaling helps improve clustering.",
)
k = st.sidebar.slider(
    "Number of clusters (k)", min_value=2, max_value=9, value=3,
    help="Task 3: pick k, informed by the elbow plot below.",
)
max_k_for_elbow = st.sidebar.slider(
    "Max k to test in elbow plot", min_value=3, max_value=15, value=10
)
X = df.values.astype(float)
if use_scaling:
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X
X_df = pd.DataFrame(X_scaled, columns=df.columns)
st.subheader("📉 Elbow Plot")
@st.cache_data
def compute_sse(X_data, max_k):
    sse = []
    for kk in range(1, max_k + 1):
        km = KMeans(n_clusters=kk, n_init=10, random_state=42)
        km.fit(X_data)
        sse.append(km.inertia_)
    return sse
sse = compute_sse(X_scaled, max_k_for_elbow)
k_rng = list(range(1, max_k_for_elbow + 1))
fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
ax_elbow.plot(k_rng, sse, marker="o")
ax_elbow.axvline(x=k, color="red", linestyle="--", label=f"Selected k = {k}")
ax_elbow.set_xlabel("K")
ax_elbow.set_ylabel("Sum of Squared Error (SSE)")
ax_elbow.set_title("Elbow Method")
ax_elbow.legend()
st.pyplot(fig_elbow)
st.caption(
    "Look for the 'elbow' — the point where adding more clusters stops "
    "reducing SSE significantly. For this dataset it's typically around k = 3."
)
st.subheader(f"🎯 Clustering Result (k = {k})")
km = KMeans(n_clusters=k, n_init=10, random_state=42)
y_pred = km.fit_predict(X_scaled)
result_df = df.copy()
result_df["cluster"] = y_pred
colors = plt.cm.tab10(np.linspace(0, 1, k))
fig_cluster, ax_cluster = plt.subplots(figsize=(6, 5))
for cluster_id in range(k):
    subset = result_df[result_df["cluster"] == cluster_id]
    ax_cluster.scatter(
        subset["petal length (cm)"],
        subset["petal width (cm)"],
        color=colors[cluster_id],
        label=f"Cluster {cluster_id}",
        s=50,
    )
centroids = km.cluster_centers_
if use_scaling:
    centroids = scaler.inverse_transform(centroids)
ax_cluster.scatter(
    centroids[:, 0], centroids[:, 1],
    color="black", marker="X", s=200, label="Centroids"
)
ax_cluster.set_xlabel("Petal Length (cm)")
ax_cluster.set_ylabel("Petal Width (cm)")
ax_cluster.set_title(f"K-Means Clusters (k={k}, scaled={use_scaling})")
ax_cluster.legend()
col1, col2 = st.columns([2, 1])
with col1:
    st.pyplot(fig_cluster)
with col2:
    st.markdown("**Cluster sizes**")
    st.dataframe(result_df["cluster"].value_counts().rename("count"))
    st.markdown("**Clustered data (sample)**")
    st.dataframe(result_df.head(10))
with st.expander("🌼 Compare clusters to true species (ground truth, for reference)"):
    compare_df = df_full[["petal length (cm)", "petal width (cm)", "flower_name"]].copy()
    compare_df["cluster"] = y_pred
    st.dataframe(
        compare_df.groupby(["flower_name", "cluster"]).size().unstack(fill_value=0)
    )
st.markdown("---")
st.caption(
    "Built with Streamlit • scikit-learn KMeans • Iris dataset "
    "(sepal features dropped as per exercise instructions)."
)

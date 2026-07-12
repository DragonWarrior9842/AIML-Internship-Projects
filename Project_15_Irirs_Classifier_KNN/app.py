import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE

st.set_page_config(page_title="KNN — Iris Flower Classifier", page_icon="🌸", layout="wide")

sns.set_style("whitegrid")

SPECIES_NAMES = ["setosa", "versicolor", "virginica"]
SPECIES_COLORS = {"setosa": "red", "versicolor": "purple", "virginica": "orange"}


@st.cache_data
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame({
        "sepal_length": iris.data[:, 0],
        "sepal_width": iris.data[:, 1],
        "petal_length": iris.data[:, 2],
        "petal_width": iris.data[:, 3],
        "species": iris.target,
    })
    df["species_name"] = df["species"].map(dict(enumerate(SPECIES_NAMES)))
    return iris, df


@st.cache_data
def compute_tsne(df, random_state=0):
    features_with_target = df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]]
    tsne = TSNE(n_components=2, random_state=random_state)
    X_2d = tsne.fit_transform(features_with_target)
    return X_2d


@st.cache_resource
def train_knn(x_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    return knn


iris, df = load_data()

st.sidebar.title("🌸 Iris KNN Classifier")
st.sidebar.markdown("K-Nearest Neighbors on the classic Iris dataset")
st.sidebar.markdown("---")
st.sidebar.subheader("Model settings")
n_neighbors = st.sidebar.slider("Number of neighbors (k)", 1, 25, 7)
test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.30, 0.05)
random_state = st.sidebar.number_input("Random state", value=1, step=1)

x = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=random_state
)
knn = train_knn(x_train, y_train, n_neighbors)
pred = knn.predict(x_test)
accuracy_knn = accuracy_score(y_test, pred) * 100

st.sidebar.markdown("---")
st.sidebar.metric("Current accuracy", f"{accuracy_knn:.2f}%")

st.title("🌸 K-Nearest Neighbor Algorithm — Iris Flower Classification")
st.markdown(
    "This app recreates the KNN case-study notebook: dataset exploration, "
    "visualisation, model training, and evaluation — fully interactive."
)

tab_overview, tab_eda, tab_corr, tab_tsne, tab_model, tab_predict = st.tabs(
    [
        "🏠 Overview",
        "📊 EDA",
        "🌡️ Correlation",
        "🔵 t-SNE",
        "🤖 Model & Evaluation",
        "🔮 Try It Yourself",
    ]
)

# ==========================================================================
with tab_overview:
    st.subheader("Load Iris flower dataset")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Target names (species)**")
        st.write(list(iris.target_names))
        st.markdown("**Feature names (inputs)**")
        st.write(list(iris.feature_names))
    with c2:
        st.markdown("**Raw data sample (first 5 rows)**")
        st.write(iris.data[0:5])

    st.subheader("Understanding the dataset")
    st.markdown("**DataFrame preview**")
    st.dataframe(df.head(), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Summary statistics**")
        st.dataframe(df.describe(), use_container_width=True)
    with c4:
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown("**Null values per column**")
        st.dataframe(df.isnull().sum().rename("null_count"), use_container_width=True)

    st.markdown("**Column info**")
    info_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null_count": df.notnull().sum().values,
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown("**Species value counts** — `setosa: 0, versicolor: 1, virginica: 2`")
    st.dataframe(df["species"].value_counts().rename("count"), use_container_width=True)

# ==========================================================================
with tab_eda:
    st.subheader("Feature distributions")
    hc1, hc2 = st.columns(2)
    with hc1:
        fig, ax = plt.subplots()
        ax.hist(df["sepal_length"], bins=15, color="#4C72B0")
        ax.set_title("Sepal Length distribution")
        ax.set_xlabel("sepal_length")
        st.pyplot(fig)
    with hc2:
        fig, ax = plt.subplots()
        ax.hist(df["petal_length"], bins=15, color="#55A868")
        ax.set_title("Petal Length distribution")
        ax.set_xlabel("petal_length")
        st.pyplot(fig)

    st.subheader("Explore any feature")
    feature_choice = st.selectbox(
        "Feature", ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sp in SPECIES_NAMES:
        subset = df[df["species_name"] == sp]
        ax.hist(subset[feature_choice], bins=15, alpha=0.6, label=sp, color=SPECIES_COLORS[sp])
    ax.set_xlabel(feature_choice)
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of {feature_choice} by species")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Pairwise scatter — pick two features")
    sc1, sc2 = st.columns(2)
    with sc1:
        x_feat = st.selectbox("X axis", ["sepal_length", "sepal_width", "petal_length", "petal_width"], index=2)
    with sc2:
        y_feat = st.selectbox("Y axis", ["sepal_length", "sepal_width", "petal_length", "petal_width"], index=3)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sp in SPECIES_NAMES:
        subset = df[df["species_name"] == sp]
        ax.scatter(subset[x_feat], subset[y_feat], label=sp, color=SPECIES_COLORS[sp], alpha=0.7)
    ax.set_xlabel(x_feat)
    ax.set_ylabel(y_feat)
    ax.legend()
    st.pyplot(fig)

# ==========================================================================
with tab_corr:
    st.subheader("Correlation matrix")
    corr = df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]].corr()
    st.dataframe(corr, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm")
    st.pyplot(fig)

# ==========================================================================
with tab_tsne:
    st.subheader("t-SNE — 2D projection")
    st.caption(
        "Matches the original notebook, which runs t-SNE on all columns "
        "(including the species label) to visualise cluster separation."
    )
    tsne_seed = st.number_input("t-SNE random state", value=0, step=1, key="tsne_seed")
    X_2d = compute_tsne(df, random_state=tsne_seed)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=df["species"], cmap="viridis", alpha=0.8)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, SPECIES_NAMES, title="species")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("t-SNE projection of the Iris dataset")
    st.pyplot(fig)

# ==========================================================================
with tab_model:
    st.subheader("Train / test split")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**x_train shape:** {x_train.shape}")
        st.markdown(f"**x_test shape:** {x_test.shape}")
    with c2:
        st.markdown(f"**y_train shape:** {y_train.shape}")
        st.markdown(f"**y_test shape:** {y_test.shape}")

    st.subheader(f"K-Nearest Neighbors (k = {n_neighbors})")
    st.markdown(f"### Accuracy: **{accuracy_knn:.2f}%**")

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=SPECIES_NAMES, yticklabels=SPECIES_NAMES,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Accuracy vs. k")
    if st.checkbox("Show accuracy curve across different k values"):
        k_range = range(1, 26)
        accuracies = []
        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(x_train, y_train)
            accuracies.append(accuracy_score(y_test, model.predict(x_test)) * 100)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(list(k_range), accuracies, marker="o")
        ax.axvline(n_neighbors, color="red", linestyle="--", label=f"current k = {n_neighbors}")
        ax.set_xlabel("k (number of neighbors)")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        st.pyplot(fig)

# ==========================================================================
with tab_predict:
    st.subheader("Predict a flower's species")
    st.markdown("Enter measurements below and the trained KNN model will predict the species.")

    p1, p2 = st.columns(2)
    with p1:
        sepal_length = st.slider("Sepal length (cm)", float(x["sepal_length"].min()), float(x["sepal_length"].max()), float(x["sepal_length"].mean()))
        sepal_width = st.slider("Sepal width (cm)", float(x["sepal_width"].min()), float(x["sepal_width"].max()), float(x["sepal_width"].mean()))
    with p2:
        petal_length = st.slider("Petal length (cm)", float(x["petal_length"].min()), float(x["petal_length"].max()), float(x["petal_length"].mean()))
        petal_width = st.slider("Petal width (cm)", float(x["petal_width"].min()), float(x["petal_width"].max()), float(x["petal_width"].mean()))

    input_df = pd.DataFrame([{
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }])

    prediction = knn.predict(input_df)[0]
    proba = knn.predict_proba(input_df)[0]

    st.markdown(f"### Predicted species: **{SPECIES_NAMES[prediction]}** 🌸")

    proba_df = pd.DataFrame({"species": SPECIES_NAMES, "probability": proba})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(proba_df["species"], proba_df["probability"], color=[SPECIES_COLORS[s] for s in SPECIES_NAMES])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    st.caption(f"Prediction made using k = {n_neighbors} nearest neighbors.")

st.markdown("---")
st.caption("Built with Streamlit · Recreates the K-Nearest Neighbor Iris classification notebook.")
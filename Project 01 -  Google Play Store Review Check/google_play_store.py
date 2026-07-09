import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

# ---------------- Session 1: Data Handling and Cleaning ----------------

inp0 = pd.read_csv("googleplaystore_v2.csv")
print(inp0.head())
print(inp0.shape)
print(inp0.info())
print(inp0.isnull().sum())

# Drop rows with null Rating (target variable)
inp1 = inp0[~inp0.Rating.isnull()]
print(inp1.shape)
print(inp1.Rating.isnull().sum())
print(inp1.isnull().sum())

# Inspect / drop the shifted row in Android Ver
print(inp1[inp1['Android Ver'].isnull()])
inp1 = inp1[~(inp1['Android Ver'].isnull() & (inp1.Category == "1.9"))]
print(inp1[inp1['Android Ver'].isnull()])

# Impute Android Ver with mode
print(inp1['Android Ver'].value_counts())
inp1['Android Ver'] = inp1['Android Ver'].fillna(inp1['Android Ver'].mode()[0])
print(inp1['Android Ver'].isnull().sum())
print(inp1.isnull().sum())

# Impute Current Ver with mode
print(inp1['Current Ver'].value_counts())
inp1['Current Ver'] = inp1['Current Ver'].fillna(inp1['Current Ver'].mode()[0])
print(inp1['Current Ver'].value_counts())

# ---- Incorrect data types ----
print(inp1.dtypes)

print(inp1.Price.value_counts())
inp1.Price = inp1.Price.apply(lambda x: 0 if x == "0" else float(x[1:]))
print(inp1.Price.dtype)

print(inp1.Reviews.value_counts())
inp1.Reviews = inp1.Reviews.astype("int32")
print(inp1.Reviews.describe())

print(inp1.Installs.head())
def clean_installs(val):
    return int(val.replace(",", "").replace("+", ""))
inp1.Installs = inp1.Installs.apply(clean_installs)
print(inp1.Installs.describe())

# ---- Sanity checks ----
print(inp1[(inp1.Reviews > inp1.Installs)].shape)
inp1 = inp1[inp1.Reviews <= inp1.Installs]
print(inp1[(inp1.Type == "Free") & (inp1.Price > 0)])

# ---- Outlier analysis: Price ----
plt.boxplot(inp1.Price)
plt.title("Boxplot - Price"); plt.show()

print(inp1[inp1.Price > 200])
inp1 = inp1[inp1.Price < 200]
print(inp1.Price.describe())

inp1[inp1.Price > 0].Price.plot.box()
plt.title("Boxplot - Price (paid apps)"); plt.show()

print(inp1[inp1.Price > 30])
inp1 = inp1[inp1.Price <= 30]
print(inp1.shape)

# ---- Histograms ----
plt.hist(inp1.Reviews)
plt.title("Histogram - Reviews"); plt.show()

plt.boxplot(inp1.Reviews)
plt.title("Boxplot - Reviews"); plt.show()

print(inp1[inp1.Reviews >= 1000000].shape)
inp1 = inp1[inp1.Reviews <= 1000000]
print(inp1.shape)

plt.hist(inp1.Reviews)
plt.title("Histogram - Reviews (cleaned)"); plt.show()

plt.boxplot(inp1.Installs)
plt.title("Boxplot - Installs"); plt.show()
print(inp1.Installs.describe())
q1, q3 = inp1.Installs.quantile([0.25, 0.75])
print("IQR of Installs:", q3 - q1)

inp1 = inp1[inp1.Installs <= 100000000]
print(inp1.shape)

inp1.Size.plot.hist()
plt.title("Histogram - Size"); plt.show()
print(inp1.Size.describe())

plt.boxplot(inp1.Size.dropna())
plt.title("Boxplot - Size"); plt.show()
print("Median Size:", inp1.Size.median())

# ---------------- Session 2: Data Visualisation with Seaborn ----------------

# Distribution plots (sns.distplot deprecated -> sns.histplot with kde)
sns.histplot(inp1.Rating, kde=True)
plt.title("Distribution of Ratings"); plt.show()

sns.histplot(inp1.Rating, bins=20, kde=True)
plt.title("Distribution of Ratings (20 bins)"); plt.show()

sns.histplot(inp1.Rating, bins=20, kde=True, color="g")
plt.title("Distribution of app ratings", fontsize=12); plt.show()

# Styling options
sns.set_style("dark")
sns.histplot(inp1.Rating, bins=20, kde=True, color="g")
plt.title("Distribution of app ratings", fontsize=12); plt.show()

sns.set_style("white")
sns.histplot(inp1.Rating, bins=20, kde=True, color="g")
plt.title("Distribution of app ratings", fontsize=12); plt.show()

plt.style.use("ggplot")
sns.histplot(inp1.Rating, bins=20, kde=True)
plt.show()

plt.style.use("dark_background")
sns.histplot(inp1.Rating, bins=20, kde=True)
plt.show()

plt.style.use("default")

# ---- Pie chart / bar chart on Content Rating ----
print(inp1['Content Rating'].value_counts())
inp1 = inp1[~inp1['Content Rating'].isin(["Adults only 18+", "Unrated"])]
print(inp1.shape)
inp1.reset_index(inplace=True, drop=True)
print(inp1.info())
print(inp1['Content Rating'].value_counts())

inp1['Content Rating'].value_counts().plot.pie()
plt.title("Content Rating - Pie chart"); plt.ylabel("")
plt.show()

inp1['Content Rating'].value_counts().plot.bar()
plt.title("Content Rating - Bar chart"); plt.show()

# 4th highest Android version type
android_ver_counts = inp1['Android Ver'].value_counts()
print(android_ver_counts)
print("4th highest Android version:", android_ver_counts.index[3], "->", android_ver_counts.iloc[3])
android_ver_counts.plot.barh()
plt.title("Android Version distribution"); plt.show()

# ---- Scatter plots ----
plt.scatter(inp1.Size, inp1.Rating)
plt.xlabel("Size"); plt.ylabel("Rating"); plt.title("Size vs Rating"); plt.show()

sns.set_style("white")
sns.jointplot(x="Size", y="Rating", data=inp1)
plt.show()

sns.jointplot(x="Price", y="Rating", data=inp1)
plt.show()

sns.jointplot(x="Price", y="Rating", data=inp1, kind="reg")
plt.show()

sns.jointplot(x="Price", y="Rating", data=inp1[inp1.Price > 0], kind="reg")
plt.show()

# ---- Pair plots ----
sns.pairplot(inp1[['Reviews', 'Size', 'Price', 'Rating']])
plt.show()

# ---- Bar charts revisited ----
inp1.groupby(['Content Rating'])['Rating'].mean().plot.bar()
plt.title("Mean Rating by Content Rating"); plt.show()

inp1.groupby(['Content Rating'])['Rating'].median().plot.bar()
plt.title("Median Rating by Content Rating"); plt.show()

sns.barplot(data=inp1, x="Content Rating", y="Rating")
plt.show()

sns.barplot(data=inp1, x="Content Rating", y="Rating", estimator=np.median)
plt.show()

sns.barplot(data=inp1, x="Content Rating", y="Rating", estimator=lambda x: np.quantile(x, 0.05))
plt.show()

sns.barplot(data=inp1, x="Content Rating", y="Rating", estimator=np.min)
plt.title("Minimum Rating by Content Rating"); plt.show()

# ---- Box plots revisited ----
plt.figure(figsize=[9, 7])
sns.boxplot(data=inp1, x='Content Rating', y='Rating')
plt.show()

sns.boxplot(x=inp1.Rating)
plt.show()

print(inp1['Genres'].value_counts())
c = ['Tools', 'Entertainment', 'Medical', 'Education']
inp5 = inp1[inp1['Genres'].isin(c)]
sns.boxplot(data=inp5, x='Genres', y='Rating')
plt.title("Rating across top 4 Genres"); plt.show()

# ---- Heat maps ----
inp1['Size_Bucket'] = pd.qcut(inp1.Size, [0, 0.2, 0.4, 0.6, 0.8, 1], ["VL", "L", "M", "H", "VH"])

pivot_mean = pd.pivot_table(data=inp1, index="Content Rating", columns="Size_Bucket", values="Rating")
print(pivot_mean)

pivot_median = pd.pivot_table(data=inp1, index="Content Rating", columns="Size_Bucket", values="Rating", aggfunc=np.median)
print(pivot_median)

pivot_20p = pd.pivot_table(data=inp1, index="Content Rating", columns="Size_Bucket", values="Rating",
                            aggfunc=lambda x: np.quantile(x, 0.2))
print(pivot_20p)

res = pivot_20p
sns.heatmap(res)
plt.title("Heatmap - 20th percentile Rating"); plt.show()

sns.heatmap(res, cmap="Greens", annot=True)
plt.title("Heatmap - 20th percentile Rating (styled)"); plt.show()

# Replace Content Rating with Review buckets, aggregation at minimum
inp1['Review_Bucket'] = pd.qcut(inp1.Reviews, [0, 0.2, 0.4, 0.6, 0.8, 1], ["VL", "L", "M", "H", "VH"])
res2 = pd.pivot_table(data=inp1, index="Review_Bucket", columns="Size_Bucket", values="Rating", aggfunc=np.min)
print(res2)
sns.heatmap(res2, cmap="Greens", annot=True)
plt.title("Heatmap - Min Rating (Review Bucket vs Size Bucket)"); plt.show()

# ---------------- Session 3: Additional Visualisations ----------------

# Line plot
print(inp1['Last Updated'].head())
inp1['updated_month'] = pd.to_datetime(inp1['Last Updated']).dt.month
print(inp1.groupby(['updated_month'])['Rating'].mean())

plt.figure(figsize=[10, 5])
inp1.groupby(['updated_month'])['Rating'].mean().plot()
plt.title("Average Rating by Month"); plt.show()

# Stacked bar chart
monthly = pd.pivot_table(data=inp1, values="Installs", index="updated_month", columns="Content Rating", aggfunc="sum")
print(monthly)

monthly.plot(kind="bar", stacked=True, figsize=[10, 6])
plt.title("Installs by Month and Content Rating"); plt.show()

monthly_perc = monthly[["Everyone", "Everyone 10+", "Mature 17+", "Teen"]].apply(lambda x: x / x.sum(), axis=1)
monthly_perc.plot(kind="bar", stacked=True, figsize=[10, 6])
plt.title("Installs Proportion by Month and Content Rating"); plt.show()

# Plotly
try:
    import plotly.express as px
    res_plotly = inp1.groupby(["updated_month"])[['Rating']].mean().reset_index()
    fig = px.line(res_plotly, x="updated_month", y="Rating", title="Monthly average rating")
    fig.show()
except ImportError:
    print("plotly not installed - run 'pip install plotly' to view the interactive chart")

print("\nAll tasks completed successfully.")

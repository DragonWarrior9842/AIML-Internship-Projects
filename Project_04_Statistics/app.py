import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import scipy.stats as stats
from scipy.stats import binom, poisson, norm, chi2, f as f_dist
from statsmodels.stats import weightstats, proportion
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Statistics with Python", page_icon="📊", layout="wide")
st.title("📊 Statistics with Python — Interactive Course Companion")
st.caption(
    "Covers descriptive statistics, probability distributions, the Central Limit "
    "Theorem, one- and two-sample hypothesis tests, ANOVA, and chi-square tests — "
    "built from the course notebooks (Sections 2, 3, 6, 7, 8)."
)
DATASETS = {
    "Machine1.csv": "Machine1.csv",
    "Perfume_Volumes.csv": "Perfume_Volumes.csv",
    "Piece_Dim.csv": "Piece_Dim.csv",
    "Smokers.csv": "Smokers.csv",
    "Two_Machines.csv": "Two_Machines.csv",
    "bottle_caps.csv": "bottle_caps.csv",
    "titanic.csv": "titanic.csv",
}
def get_dataset(default_key, uploader_key, label="Or upload your own CSV"):
    options = list(DATASETS.keys()) + ["Upload my own"]
    default_index = options.index(default_key) if default_key in options else 0
    choice = st.selectbox("Dataset", options, index=default_index, key=f"sel_{uploader_key}")
    if choice == "Upload my own":
        upl = st.file_uploader(label, type=["csv"], key=f"upl_{uploader_key}")
        if upl is None:
            st.info("Upload a CSV to continue.")
            return None
        try:
            return pd.read_csv(upl)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            return None
    path = os.path.join(BASE_DIR, DATASETS[choice])
    if not os.path.exists(path):
        st.error(f"Bundled file `{DATASETS[choice]}` is missing from the app folder.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Couldn't read `{DATASETS[choice]}`: {e}")
        return None
section = st.sidebar.radio(
    "Choose a section",
    [
        "1. Descriptive Statistics",
        "2. Probability Distributions",
        "3. Central Limit Theorem",
        "4. One-Sample Tests",
        "5. Two-Sample Tests",
        "6. ANOVA",
        "7. Chi-Square Tests",
    ],
)
st.sidebar.markdown("---")
with st.sidebar.expander("👤 About the Developer"):
    st.markdown("**Aditya Agarwal**")
    st.caption("Data Science / ML Enthusiast")
    st.write(
        "B.Tech CSE student at Shri Ramswaroop Memorial College of "
        "Engineering and Management, Lucknow."
    )
    st.markdown(
        "[📧 Email](mailto:aasblko@gmail.com) &nbsp;·&nbsp; "
        "[💼 LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/)"
    )
    st.markdown(
        "[🐙 GitHub](https://github.com/DragonWarrior9842) &nbsp;·&nbsp; "
        "[🌐 Instagram](https://www.instagram.com/adityaagarwal67/)"
    )
st.sidebar.caption("Built with Streamlit • NumPy • SciPy • statsmodels")
def section_descriptive():
    st.header("Descriptive Statistics")
    st.write(
        "Mean, median, mode, range, variance, standard deviation, and IQR "
        "on any numeric column of your data."
    )
    df = get_dataset("titanic.csv", "desc")
    if df is None:
        return
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("This dataset has no numeric columns.")
        return
    col = st.selectbox("Numeric column", numeric_cols)
    series = df[col].dropna()
    mean_ = series.mean()
    median_ = series.median()
    mode_ = series.mode()
    var_ = series.var(ddof=0)
    std_ = series.std(ddof=0)
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    rng = series.max() - series.min()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{mean_:,.3f}")
    c2.metric("Median", f"{median_:,.3f}")
    c3.metric("Mode", ", ".join(f"{m:,.3f}" for m in mode_.tolist()[:3]))
    c4.metric("Range", f"{rng:,.3f}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Variance", f"{var_:,.3f}")
    c6.metric("Std Dev", f"{std_:,.3f}")
    c7.metric("Q1 / Q3", f"{q1:,.2f} / {q3:,.2f}")
    c8.metric("IQR", f"{iqr:,.3f}")
    st.caption(
        f"Outlier fences (1.5×IQR rule): below **{lower_fence:,.2f}** or above **{upper_fence:,.2f}**."
    )
    v1, v2 = st.columns(2)
    with v1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(series, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
    with v2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(y=series, ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)
def section_distributions():
    st.header("Probability Distributions")
    dist_tab = st.tabs(["Binomial", "Poisson", "Normal"])
    with dist_tab[0]:
        st.subheader("Binomial Distribution")
        c1, c2 = st.columns(2)
        n = c1.slider("n (number of trials)", 1, 100, 20)
        p = c2.slider("p (probability of success)", 0.0, 1.0, 0.12, step=0.01)
        x_ax = np.arange(0, n + 1)
        pmf = binom.pmf(x_ax, n, p)
        cdf = binom.cdf(x_ax, n, p)
        st.write(f"Mean = **{binom.mean(n, p):.3f}**, Variance = **{binom.var(n, p):.3f}**, "
                 f"Std Dev = **{binom.std(n, p):.3f}**")
        k = st.number_input("Compute P(X ≤ k) for k =", 0, n, min(2, n))
        st.write(f"P(X ≤ {k}) = **{binom.cdf(k, n, p):.4f}**  |  "
                 f"P(X = {k}) = **{binom.pmf(k, n, p):.4f}**  |  "
                 f"P(X > {k}) = **{binom.sf(k, n, p):.4f}**")
        v1, v2 = st.columns(2)
        with v1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=x_ax, y=pmf, ax=ax, color="steelblue")
            ax.set_title("PMF")
            ax.set_xticks(x_ax[::max(1, n // 15)])
            st.pyplot(fig)
        with v2:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=x_ax, y=cdf, ax=ax, color="darkorange")
            ax.set_title("CDF")
            ax.set_xticks(x_ax[::max(1, n // 15)])
            st.pyplot(fig)
    with dist_tab[1]:
        st.subheader("Poisson Distribution")
        mu = st.slider("λ (average rate)", 0.1, 30.0, 3.6, step=0.1)
        max_x = int(mu + 4 * np.sqrt(mu)) + 5
        x_ax = np.arange(0, max_x)
        pmf = poisson.pmf(x_ax, mu)
        cdf = poisson.cdf(x_ax, mu)
        st.write(f"Mean = **{poisson.mean(mu):.3f}**, Variance = **{poisson.var(mu):.3f}**, "
                 f"Std Dev = **{poisson.std(mu):.3f}**")
        k = st.number_input("Compute P(X = k) for k =", 0, max_x, min(7, max_x - 1))
        st.write(f"P(X = {k}) = **{poisson.pmf(k, mu):.4f}**  |  "
                 f"P(X ≤ {k}) = **{poisson.cdf(k, mu):.4f}**  |  "
                 f"P(X > {k}) = **{poisson.sf(k, mu):.4f}**")
        v1, v2 = st.columns(2)
        with v1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=x_ax, y=pmf, ax=ax, color="seagreen")
            ax.set_title("PMF")
            st.pyplot(fig)
        with v2:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=x_ax, y=cdf, ax=ax, color="firebrick")
            ax.set_title("CDF")
            st.pyplot(fig)
    with dist_tab[2]:
        st.subheader("Normal Distribution")
        c1, c2 = st.columns(2)
        mu_n = c1.number_input("Mean (μ)", value=150.0)
        sigma_n = c2.number_input("Std Dev (σ)", value=2.0, min_value=0.01)
        x_ax = np.linspace(mu_n - 4 * sigma_n, mu_n + 4 * sigma_n, 500)
        pdf = norm.pdf(x_ax, mu_n, sigma_n)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_ax, pdf)
        ax.fill_between(x_ax, pdf, alpha=0.1)
        ax.set_title("Normal PDF")
        st.pyplot(fig)
        st.markdown("**Probability calculator**")
        c3, c4 = st.columns(2)
        lo = c3.number_input("Lower bound (leave blank-like -inf by using a very low number)", value=float(mu_n - sigma_n))
        hi = c4.number_input("Upper bound", value=float(mu_n + sigma_n))
        prob = norm.cdf(hi, mu_n, sigma_n) - norm.cdf(lo, mu_n, sigma_n)
        st.write(f"P({lo:.2f} < X < {hi:.2f}) = **{prob:.4f}**")
        st.write(f"P(X > {hi:.2f}) = **{norm.sf(hi, mu_n, sigma_n):.4f}**  |  "
                 f"P(X < {lo:.2f}) = **{norm.cdf(lo, mu_n, sigma_n):.4f}**")
def section_clt():
    st.header("Central Limit Theorem")
    st.write(
        "Draw many samples from a (possibly very non-normal) population and watch "
        "the distribution of **sample means** become approximately normal as the "
        "sample size grows."
    )
    c1, c2, c3 = st.columns(3)
    population = c1.selectbox("Population shape", ["Uniform (die roll 1-6)", "Bimodal"])
    sample_size = c2.slider("Sample size (n)", 1, 200, 4)
    n_samples = c3.slider("Number of samples", 100, 200000, 10000, step=100)
    rng = np.random.default_rng(42)
    if population == "Uniform (die roll 1-6)":
        pop_data = rng.integers(1, 7, size=sample_size * n_samples)
    else:
        a = rng.normal(100, 5, size=(sample_size * n_samples) // 2)
        b = rng.normal(80, 5, size=(sample_size * n_samples) // 2)
        pop_data = np.append(a, b)
        rng.shuffle(pop_data)
    usable = (len(pop_data) // sample_size) * sample_size
    sample_means = pop_data[:usable].reshape(-1, sample_size).mean(axis=1)
    v1, v2 = st.columns(2)
    with v1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(pop_data, bins=40, ax=ax)
        ax.set_title("Individual values (population)")
        st.pyplot(fig)
    with v2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(sample_means, bins=40, kde=True, ax=ax, color="darkorange")
        ax.set_title(f"Sample means (n={sample_size})")
        st.pyplot(fig)
    c4, c5 = st.columns(2)
    c4.metric("Population mean / std", f"{pop_data.mean():.3f} / {pop_data.std():.3f}")
    c5.metric("Sample-means mean / std", f"{sample_means.mean():.3f} / {sample_means.std():.3f}")
    st.caption(
        "As n grows, the spread of the sample-means distribution shrinks "
        "(standard error ≈ population σ / √n), and its shape becomes normal "
        "regardless of the population's original shape."
    )
def section_one_sample():
    st.header("One-Sample Tests")
    test_type = st.selectbox(
        "Test", ["Mean (Z-test)", "Mean (t-test)", "Proportion", "Variance (Chi-square)"]
    )
    alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, step=0.01)
    if test_type in ("Mean (Z-test)", "Mean (t-test)"):
        df = get_dataset("Machine1.csv", "one_mean")
        if df is None:
            return
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns in this dataset.")
            return
        col = st.selectbox("Column to test", numeric_cols)
        data = df[col].dropna()
        c1, c2 = st.columns(2)
        mu0 = c1.number_input("Hypothesized mean (μ₀)", value=float(round(data.mean())))
        alternative = c2.selectbox("Alternative hypothesis", ["two-sided", "larger", "smaller"])
        st.write(f"Sample: n = {len(data)}, mean = {data.mean():.3f}, std = {data.std():.3f}")
        v1, v2 = st.columns(2)
        with v1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(data, kde=True, ax=ax)
            ax.axvline(mu0, color="red", linestyle="--", label="H₀ mean")
            ax.legend()
            st.pyplot(fig)
        with v2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(y=data, ax=ax)
            st.pyplot(fig)
        if test_type == "Mean (Z-test)":
            sigma = st.number_input(
                "Population std dev (σ) — a Z-test assumes this is known",
                value=float(data.std()), min_value=0.0001,
            )
            n = len(data)
            zstat = (data.mean() - mu0) / (sigma / np.sqrt(n))
            if alternative == "two-sided":
                pval = 2 * norm.sf(abs(zstat))
            elif alternative == "larger":
                pval = norm.sf(zstat)
            else:
                pval = norm.cdf(zstat)
            st.subheader("Result")
            st.write(f"Z statistic = (x̄ − μ₀) / (σ/√n) = **{zstat:.4f}**, p-value = **{pval:.4f}**")
        else:
            tstat, pval = stats.ttest_1samp(data, mu0, alternative=alternative)
            st.subheader("Result")
            st.write(f"t statistic = **{tstat:.4f}**, p-value = **{pval:.4f}**")
        if pval < alpha:
            st.success(f"p-value < α ({alpha}) → **Reject the null hypothesis**.")
        else:
            st.info(f"p-value ≥ α ({alpha}) → **Fail to reject the null hypothesis**.")
    elif test_type == "Proportion":
        st.write("Example: has the proportion of smokers changed from a historical value?")
        df = get_dataset("Smokers.csv", "one_prop")
        if df is None:
            return
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not cat_cols:
            st.warning("No categorical columns found — pick a dataset with a Yes/No style column.")
            return
        col = st.selectbox("Categorical column", cat_cols, index=cat_cols.index("Smokers") if "Smokers" in cat_cols else 0)
        categories = df[col].dropna().unique().tolist()
        success_val = st.selectbox("Value to count as 'success'", categories)
        n_total = df[col].dropna().shape[0]
        n_success = (df[col] == success_val).sum()
        st.write(f"Observed: **{n_success}** successes out of **{n_total}** (p̂ = {n_success / n_total:.4f})")
        c1, c2 = st.columns(2)
        p0 = c1.slider("Hypothesized proportion (p₀)", 0.0, 1.0, 0.21, step=0.01)
        alternative = c2.selectbox("Alternative hypothesis", ["two-sided", "less", "greater"], key="prop_alt")
        result = stats.binomtest(n_success, n_total, p0, alternative=alternative)
        st.subheader("Result (exact binomial test)")
        st.write(f"p-value = **{result.pvalue:.4f}**")
        if result.pvalue < alpha:
            st.success(f"p-value < α ({alpha}) → **Reject the null hypothesis**.")
        else:
            st.info(f"p-value ≥ α ({alpha}) → **Fail to reject the null hypothesis**.")
    else:
        st.write("Example: has the standard deviation of a filling process increased from its established value?")
        c1, c2, c3 = st.columns(3)
        n = c1.number_input("Sample size (n)", min_value=2, value=51)
        s = c2.number_input("Sample std dev (s)", min_value=0.0001, value=2.35)
        sigma0 = c3.number_input("Hypothesized population std dev (σ₀)", min_value=0.0001, value=2.0)
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "greater", "less"], key="var_alt")
        chi_cal = (n - 1) * (s ** 2) / (sigma0 ** 2)
        df_chi = n - 1
        st.write(f"Chi-square statistic = (n-1)·s² / σ₀² = **{chi_cal:.4f}**, df = **{df_chi}**")
        if alternative == "greater":
            pval = chi2.sf(chi_cal, df_chi)
            crit = chi2.isf(alpha, df_chi)
            st.write(f"Critical value (upper tail, α={alpha}) = **{crit:.4f}**")
        elif alternative == "less":
            pval = chi2.cdf(chi_cal, df_chi)
            crit = chi2.ppf(alpha, df_chi)
            st.write(f"Critical value (lower tail, α={alpha}) = **{crit:.4f}**")
        else:
            pval = 2 * min(chi2.cdf(chi_cal, df_chi), chi2.sf(chi_cal, df_chi))
            lo = chi2.ppf(alpha / 2, df_chi)
            hi = chi2.isf(alpha / 2, df_chi)
            st.write(f"Critical values (α={alpha}) = **{lo:.4f}**, **{hi:.4f}**")
        st.subheader("Result")
        st.write(f"p-value = **{pval:.4f}**")
        if pval < alpha:
            st.success(f"p-value < α ({alpha}) → **Reject the null hypothesis**.")
        else:
            st.info(f"p-value ≥ α ({alpha}) → **Fail to reject the null hypothesis**.")
        x_ax = np.linspace(0.01, max(chi_cal * 1.5, df_chi * 2), 300)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_ax, chi2.pdf(x_ax, df_chi))
        ax.axvline(chi_cal, color="red", linestyle="--", label="χ² calculated")
        ax.legend()
        ax.set_title(f"Chi-square distribution (df={df_chi})")
        st.pyplot(fig)
def get_two_groups(df, label_prefix=""):
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not cat_cols or not num_cols:
        st.warning("Need at least one categorical (grouping) column and one numeric column.")
        return None, None, None
    c1, c2 = st.columns(2)
    group_col = c1.selectbox("Grouping column", cat_cols, key=f"{label_prefix}_gcol")
    value_col = c2.selectbox("Value column", num_cols, key=f"{label_prefix}_vcol")
    levels = df[group_col].dropna().unique().tolist()
    if len(levels) < 2:
        st.warning(f"Column `{group_col}` needs at least 2 groups.")
        return None, None, None
    c3, c4 = st.columns(2)
    g1_name = c3.selectbox("Group 1", levels, index=0, key=f"{label_prefix}_g1")
    g2_name = c4.selectbox("Group 2", levels, index=1 if len(levels) > 1 else 0, key=f"{label_prefix}_g2")
    g1 = df[df[group_col] == g1_name][value_col].dropna()
    g2 = df[df[group_col] == g2_name][value_col].dropna()
    return g1, g2, (g1_name, g2_name, value_col)
def section_two_sample():
    st.header("Two-Sample Tests")
    test_type = st.selectbox(
        "Test",
        [
            "Two Means (Z-test)",
            "Two Means (t-test, equal variance)",
            "Two Means (t-test, unequal variance)",
            "Paired t-test",
            "Two Proportions",
            "Two Variances (Bartlett / Levene)",
        ],
    )
    alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, step=0.01, key="two_alpha")
    if test_type in (
        "Two Means (Z-test)",
        "Two Means (t-test, equal variance)",
        "Two Means (t-test, unequal variance)",
        "Two Variances (Bartlett / Levene)",
    ):
        df = get_dataset("Two_Machines.csv", "two_group")
        if df is None:
            return
        with st.expander("Preview data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        g1, g2, info = get_two_groups(df, "two")
        if g1 is None:
            return
        g1_name, g2_name, value_col = info
        v1, v2 = st.columns(2)
        with v1:
            st.write(f"**{g1_name}**: n={len(g1)}, mean={g1.mean():.3f}, std={g1.std():.3f}")
        with v2:
            st.write(f"**{g2_name}**: n={len(g2)}, mean={g2.mean():.3f}, std={g2.std():.3f}")
        plot_df = pd.concat([
            pd.DataFrame({value_col: g1, "group": str(g1_name)}),
            pd.DataFrame({value_col: g2, "group": str(g2_name)}),
        ])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=plot_df, x="group", y=value_col, ax=ax)
        st.pyplot(fig)
        if test_type == "Two Means (Z-test)":
            zstat, pval = weightstats.ztest(g1, g2)
            st.subheader("Result")
            st.write(f"Z statistic = **{zstat:.4f}**, p-value = **{pval:.4f}**")
        elif "equal variance" in test_type:
            tstat, pval = stats.ttest_ind(g1, g2, equal_var=True)
            st.subheader("Result")
            st.write(f"t statistic = **{tstat:.4f}**, p-value = **{pval:.4f}**")
        elif "unequal variance" in test_type:
            tstat, pval = stats.ttest_ind(g1, g2, equal_var=False)
            st.subheader("Result (Welch's t-test)")
            st.write(f"t statistic = **{tstat:.4f}**, p-value = **{pval:.4f}**")
        else:
            bstat, bp = stats.bartlett(g1, g2)
            lstat, lp = stats.levene(g1, g2)
            st.subheader("Result")
            st.write(f"Bartlett: statistic = **{bstat:.4f}**, p-value = **{bp:.4f}**")
            st.write(f"Levene: statistic = **{lstat:.4f}**, p-value = **{lp:.4f}**")
            st.caption(
                "Both test H₀: the two groups have equal variances. Levene is more "
                "robust when the data deviates from normality."
            )
            pval = bp
        if pval < alpha:
            st.success(f"p-value < α ({alpha}) → **Reject the null hypothesis**.")
        else:
            st.info(f"p-value ≥ α ({alpha}) → **Fail to reject the null hypothesis**.")
    elif test_type == "Paired t-test":
        st.write(
            "Example: blood pressure before vs. after a treatment, measured on the "
            "same subjects. Edit the table below (rows are paired)."
        )
        default_df = pd.DataFrame({
            "Before": [120, 122, 143, 100, 109],
            "After": [122, 120, 141, 109, 109],
        })
        edited = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)
        edited = edited.dropna()
        if len(edited) < 2:
            st.warning("Need at least 2 paired rows.")
            return
        before, after = edited.iloc[:, 0], edited.iloc[:, 1]
        tstat, pval = stats.ttest_rel(before, after)
        st.subheader("Result")
        st.write(f"t statistic = **{tstat:.4f}**, p-value = **{pval:.4f}**")
        st.write(f"Mean difference = **{(after - before).mean():.3f}**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=edited, ax=ax)
        st.pyplot(fig)
        if pval < alpha:
            st.success(f"p-value < α ({alpha}) → **Reject the null hypothesis**.")
        else:
            st.info(f"p-value ≥ α ({alpha}) → **Fail to reject the null hypothesis**.")
    else:
        st.write("Compare success proportions between two independent groups (manual entry).")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Group 1**")
            s1 = st.number_input("Successes (Group 1)", min_value=0, value=30)
            n1 = st.number_input("Trials (Group 1)", min_value=1, value=200)
        with c2:
            st.markdown("**Group 2**")
            s2 = st.number_input("Successes (Group 2)", min_value=0, value=10)
            n2 = st.number_input("Trials (Group 2)", min_value=1, value=100)
        method = st.selectbox("Method", ["wald", "score", "agresti-caffo"])
        zstat, pval = proportion.test_proportions_2indep(s1, n1, s2, n2, method=method)
        st.subheader("Result")
        st.write(f"p̂₁ = {s1/n1:.4f}, p̂₂ = {s2/n2:.4f}")
        st.write(f"Z statistic = **{zstat:.4f}**, p-value = **{pval:.4f}**")
        if pval < alpha:
            st.success(f"p-value < α ({alpha}) → **Reject the null hypothesis**.")
        else:
            st.info(f"p-value ≥ α ({alpha}) → **Fail to reject the null hypothesis**.")
def section_anova():
    st.header("One-Way ANOVA")
    st.write("Test whether 3 or more group means are all equal.")
    df = get_dataset("Piece_Dim.csv", "anova")
    if df is None:
        return
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not cat_cols or not num_cols:
        st.warning("Need at least one categorical column and one numeric column.")
        return
    c1, c2 = st.columns(2)
    group_col = c1.selectbox("Grouping column", cat_cols)
    value_col = c2.selectbox("Value column", num_cols)
    alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, step=0.01, key="anova_alpha")
    levels = df[group_col].dropna().unique().tolist()
    chosen = st.multiselect("Groups to include", levels, default=levels)
    if len(chosen) < 2:
        st.warning("Pick at least 2 groups.")
        return
    groups = [df[df[group_col] == lvl][value_col].dropna() for lvl in chosen]
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df = df[df[group_col].isin(chosen)]
    sns.boxplot(data=plot_df, x=group_col, y=value_col, ax=ax)
    st.pyplot(fig)
    fstat, pval = stats.f_oneway(*groups)
    st.subheader("Result")
    st.write(f"F statistic = **{fstat:.4f}**, p-value = **{pval:.4f}**")
    if pval < alpha:
        st.success(f"p-value < α ({alpha}) → **Reject H₀: not all group means are equal.**")
    else:
        st.info(f"p-value ≥ α ({alpha}) → **Fail to reject H₀: no evidence group means differ.**")
    with st.expander("Group summary"):
        st.dataframe(plot_df.groupby(group_col)[value_col].describe(), use_container_width=True)
def section_chi_square():
    st.header("Chi-Square Tests")
    test_type = st.selectbox("Test", ["Goodness of Fit", "Contingency Table (Independence)"])
    alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, step=0.01, key="chi_alpha")
    if test_type == "Goodness of Fit":
        st.write("Compare observed category counts to expected counts or proportions.")
        default_df = pd.DataFrame({
            "Category": ["A", "B", "C", "D"],
            "Observed": [25, 41, 91, 68],
            "Expected proportion": [0.1, 0.2, 0.4, 0.3],
        })
        edited = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)
        edited = edited.dropna()
        obs = edited["Observed"].astype(float)
        props = edited["Expected proportion"].astype(float)
        if not np.isclose(props.sum(), 1.0, atol=0.01):
            st.warning(f"Expected proportions sum to {props.sum():.3f}, not 1.0 — they'll be renormalized.")
            props = props / props.sum()
        expected = props * obs.sum()
        chi_stat, pval = stats.chisquare(obs, expected)
        st.subheader("Result")
        st.write(f"Chi-square statistic = **{chi_stat:.4f}**, df = **{len(obs) - 1}**, p-value = **{pval:.4f}**")
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(edited))
        width = 0.35
        ax.bar(x - width / 2, obs, width, label="Observed")
        ax.bar(x + width / 2, expected, width, label="Expected")
        ax.set_xticks(x)
        ax.set_xticklabels(edited["Category"])
        ax.legend()
        st.pyplot(fig)
        if pval < alpha:
            st.success(f"p-value < α ({alpha}) → **Reject H₀: observed distribution differs from expected.**")
        else:
            st.info(f"p-value ≥ α ({alpha}) → **Fail to reject H₀.**")
    else:
        st.write("Test whether two categorical variables are independent.")
        df = get_dataset("titanic.csv", "contingency")
        if df is None:
            return
        with st.expander("Preview data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(cat_cols) < 2:
            st.warning("Need at least 2 categorical columns.")
            return
        c1, c2 = st.columns(2)
        col1 = c1.selectbox("Variable 1", cat_cols, index=0)
        col2 = c2.selectbox("Variable 2", cat_cols, index=1 if len(cat_cols) > 1 else 0)
        table = pd.crosstab(df[col1], df[col2])
        st.write("Contingency table:")
        st.dataframe(table, use_container_width=True)
        chi_stat, pval, dof, expected = stats.chi2_contingency(table)
        st.subheader("Result")
        st.write(f"Chi-square statistic = **{chi_stat:.4f}**, df = **{dof}**, p-value = **{pval:.4f}**")
        with st.expander("Expected counts (under independence)"):
            st.dataframe(pd.DataFrame(expected, index=table.index, columns=table.columns))
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(table, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
        if pval < alpha:
            st.success(f"p-value < α ({alpha}) → **Reject H₀: the variables are not independent.**")
        else:
            st.info(f"p-value ≥ α ({alpha}) → **Fail to reject H₀: no evidence of association.**")
SECTIONS = {
    "1. Descriptive Statistics": section_descriptive,
    "2. Probability Distributions": section_distributions,
    "3. Central Limit Theorem": section_clt,
    "4. One-Sample Tests": section_one_sample,
    "5. Two-Sample Tests": section_two_sample,
    "6. ANOVA": section_anova,
    "7. Chi-Square Tests": section_chi_square,
}
SECTIONS[section]()

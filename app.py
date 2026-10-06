"""
EduPro — Learner Demographics & Course Enrollment Behavior Dashboard
Combined Data Analytics + Machine Learning dashboard.

Expects an Excel workbook (DATA_PATH) with sheets: Users, Courses, Transactions
(a Teachers sheet may also be present but is not required).

    Users:        UserID, UserName, Age, Gender, Email
    Courses:      CourseID, CourseName, CourseCategory, CourseType, CourseLevel,
                  CoursePrice, CourseDuration, CourseRating
    Transactions: TransactionID, UserID, CourseID, TransactionDate, Amount,
                  PaymentMethod, TeacherID
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, silhouette_score

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_PATH = "EduProOnlinePlatform.xlsx"  # <-- update to your workbook filename

LEVEL_MAP = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
AGE_BINS = [14, 17, 21, 25, 30, 35]
AGE_LABELS = ["<18 (15-17)", "18-21", "22-25", "26-30", "31-35"]

st.set_page_config(page_title="EduPro Learner Analytics", layout="wide")

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    users = pd.read_excel(path, sheet_name="Users")
    courses = pd.read_excel(path, sheet_name="Courses")
    transactions = pd.read_excel(path, sheet_name="Transactions")

    # basic type/column cleanup
    users.columns = [c.strip() for c in users.columns]
    courses.columns = [c.strip() for c in courses.columns]
    transactions.columns = [c.strip() for c in transactions.columns]

    users = users.drop_duplicates(subset="UserID")
    courses = courses.drop_duplicates(subset="CourseID")
    transactions = transactions.drop_duplicates(subset="TransactionID")

    if "TransactionDate" in transactions.columns:
        transactions["TransactionDate"] = pd.to_datetime(
            transactions["TransactionDate"], errors="coerce"
        )

    # keep only transactions that resolve to a valid user & course
    transactions = transactions[
        transactions["UserID"].isin(users["UserID"])
        & transactions["CourseID"].isin(courses["CourseID"])
    ]

    full = transactions.merge(users, on="UserID", how="left").merge(
        courses, on="CourseID", how="left"
    )

    full["CourseLevelNum"] = full["CourseLevel"].map(LEVEL_MAP)
    full["AgeBand"] = pd.cut(full["Age"], bins=AGE_BINS, labels=AGE_LABELS)

    return users, courses, transactions, full


@st.cache_data
def build_learner_table(full):
    agg = (
        full.groupby("UserID")
        .agg(
            Age=("Age", "first"),
            Gender=("Gender", "first"),
            Total_Enrollments=("TransactionID", "count"),
            Distinct_Categories=("CourseCategory", "nunique"),
            Avg_Level=("CourseLevelNum", "mean"),
            Paid_Share=("CourseType", lambda s: (s == "Paid").mean()),
            Avg_Course_Rating=("CourseRating", "mean"),
            Total_Spend=("Amount", "sum") if "Amount" in full.columns else ("TransactionID", "count"),
        )
        .reset_index()
    )
    agg["Gender_Male"] = (agg["Gender"] == "Male").astype(int)
    agg["AgeBand"] = pd.cut(agg["Age"], bins=AGE_BINS, labels=AGE_LABELS)

    median_enroll = agg["Total_Enrollments"].median()
    agg["Segment"] = np.where(
        agg["Total_Enrollments"] >= agg["Total_Enrollments"].quantile(0.85),
        "Heavy",
        "Light",
    )
    return agg


def popularity_index(full, dim):
    enroll_share = full[dim].value_counts(normalize=True)
    supply_share = full.drop_duplicates("CourseID")[dim].value_counts(normalize=True)
    idx = (enroll_share / supply_share * 100).reindex(supply_share.index)
    observed = full[dim].value_counts().reindex(supply_share.index).fillna(0)
    expected = supply_share * len(full)
    chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
    return idx.sort_values(ascending=False), chi2, p


def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------------
try:
    users, courses, transactions, full = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find '{DATA_PATH}'. Place your EduPro Excel workbook "
        "(sheets: Users, Courses, Transactions) in the project folder and "
        "update DATA_PATH at the top of app.py."
    )
    st.stop()

learners = build_learner_table(full)

# ----------------------------------------------------------------------------
# SIDEBAR — GLOBAL FILTERS
# ----------------------------------------------------------------------------
st.sidebar.title("EduPro Analytics")
page = st.sidebar.radio(
    "Go to",
    [
        "Learner Demographic Overview",
        "Age & Gender Enrollment Patterns",
        "Course Category & Level Popularity",
        "Engagement & Concentration Analysis",
        "ML Insights: Clustering & Prediction",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

age_band_sel = st.sidebar.multiselect(
    "Age band", options=AGE_LABELS, default=AGE_LABELS
)
gender_sel = st.sidebar.multiselect(
    "Gender", options=sorted(full["Gender"].dropna().unique()), default=sorted(full["Gender"].dropna().unique())
)
category_sel = st.sidebar.multiselect(
    "Course category",
    options=sorted(full["CourseCategory"].dropna().unique()),
    default=sorted(full["CourseCategory"].dropna().unique()),
)
level_sel = st.sidebar.multiselect(
    "Course level",
    options=["Beginner", "Intermediate", "Advanced"],
    default=["Beginner", "Intermediate", "Advanced"],
)
type_sel = st.sidebar.multiselect(
    "Course type", options=["Paid", "Free"], default=["Paid", "Free"]
)

filtered_full = full[
    full["AgeBand"].isin(age_band_sel)
    & full["Gender"].isin(gender_sel)
    & full["CourseCategory"].isin(category_sel)
    & full["CourseLevel"].isin(level_sel)
    & full["CourseType"].isin(type_sel)
]
filtered_learners = learners[
    learners["AgeBand"].isin(age_band_sel) & learners["Gender"].isin(gender_sel)
]

if filtered_full.empty:
    st.warning("No data matches the current filter selection.")
    st.stop()

# ----------------------------------------------------------------------------
# PAGE 1 — LEARNER DEMOGRAPHIC OVERVIEW
# ----------------------------------------------------------------------------
if page == "Learner Demographic Overview":
    st.title("Learner Demographic Overview")

    total_learners = filtered_full["UserID"].nunique()
    total_enrollments = len(filtered_full)
    avg_enroll_per_learner = total_enrollments / total_learners if total_learners else 0
    gender_ratio = filtered_full.drop_duplicates("UserID")["Gender"].value_counts(normalize=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Learners", f"{total_learners:,}")
    c2.metric("Total Enrollments", f"{total_enrollments:,}")
    c3.metric("Avg Enrollments / Learner", f"{avg_enroll_per_learner:.2f}")
    c4.metric(
        "Gender Split (F/M)",
        f"{gender_ratio.get('Female', 0)*100:.0f}% / {gender_ratio.get('Male', 0)*100:.0f}%",
    )

    col1, col2 = st.columns(2)
    with col1:
        age_dist = filtered_full.drop_duplicates("UserID")["Age"].value_counts().sort_index()
        fig = px.bar(
            x=age_dist.index, y=age_dist.values,
            labels={"x": "Age", "y": "Number of Learners"},
            title="Learner Age Distribution",
        )
        st.plotly_chart(fig, width="stretch")
    with col2:
        gender_counts = filtered_full.drop_duplicates("UserID")["Gender"].value_counts()
        fig = px.pie(
            names=gender_counts.index, values=gender_counts.values,
            title="Gender Distribution", hole=0.4,
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Enrollment Concentration Snapshot")
    sorted_counts = np.sort(filtered_learners["Total_Enrollments"].values)
    cum = np.cumsum(sorted_counts) / sorted_counts.sum()
    lorenz_x = np.linspace(0, 1, len(cum))
    gini = 1 - 2 * np.trapezoid(cum, lorenz_x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lorenz_x, y=cum, mode="lines", name="Lorenz curve"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality", line=dict(dash="dash")))
    fig.update_layout(title=f"Enrollment Concentration — Gini = {gini:.3f}",
                       xaxis_title="Cumulative share of learners", yaxis_title="Cumulative share of enrollments")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "A Gini coefficient closer to 1 indicates enrollment activity is concentrated "
        "among a small share of learners; closer to 0 indicates even activity across all learners."
    )

# ----------------------------------------------------------------------------
# PAGE 2 — AGE & GENDER ENROLLMENT PATTERNS
# ----------------------------------------------------------------------------
elif page == "Age & Gender Enrollment Patterns":
    st.title("Age & Gender Enrollment Patterns")

    st.subheader("Average Enrollments per Learner by Age Band")
    band_avg = filtered_learners.groupby("AgeBand", observed=True)["Total_Enrollments"].mean().reindex(AGE_LABELS)
    fig = px.bar(x=band_avg.index, y=band_avg.values,
                 labels={"x": "Age Band", "y": "Avg Enrollments per Learner"})
    st.plotly_chart(fig, width="stretch")

    groups = [g["Total_Enrollments"].values for _, g in filtered_learners.groupby("AgeBand", observed=True) if len(g) > 0]
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        st.info(
            f"**Kruskal-Wallis test (enrollments across age bands):** H = {h_stat:.2f}, "
            f"p = {p_val:.4f} — "
            + ("no significant difference across age bands." if p_val > 0.05 else "a significant difference was found.")
        )

    st.subheader("Average Enrollments per Learner by Gender")
    gender_avg = filtered_learners.groupby("Gender")["Total_Enrollments"].mean()
    fig = px.bar(x=gender_avg.index, y=gender_avg.values,
                 labels={"x": "Gender", "y": "Avg Enrollments per Learner"})
    st.plotly_chart(fig, width="stretch")

    male_vals = filtered_learners.loc[filtered_learners["Gender"] == "Male", "Total_Enrollments"]
    female_vals = filtered_learners.loc[filtered_learners["Gender"] == "Female", "Total_Enrollments"]
    if len(male_vals) > 0 and len(female_vals) > 0:
        u_stat, p_val = stats.mannwhitneyu(male_vals, female_vals)
        st.info(
            f"**Mann-Whitney U test (enrollments, Male vs Female):** p = {p_val:.4f} — "
            + ("no significant difference by gender." if p_val > 0.05 else "a significant difference was found.")
        )

    st.subheader("Enrollment Volume Heatmap — Age Band × Gender")
    pivot = filtered_full.drop_duplicates("UserID").pivot_table(
        index="AgeBand", columns="Gender", values="UserID", aggfunc="count", observed=True
    ).reindex(AGE_LABELS)
    fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                     labels=dict(color="Learner Count"))
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# PAGE 3 — COURSE CATEGORY & LEVEL POPULARITY
# ----------------------------------------------------------------------------
elif page == "Course Category & Level Popularity":
    st.title("Course Category & Level Popularity")

    st.subheader("Category Popularity Index (100 = proportional to supply)")
    idx, chi2, p = popularity_index(filtered_full, "CourseCategory")
    fig = px.bar(x=idx.index, y=idx.values, labels={"x": "Course Category", "y": "Popularity Index"})
    fig.add_hline(y=100, line_dash="dash", annotation_text="Proportional to supply")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Chi-square goodness-of-fit: χ² = {chi2:.2f}, p = {p:.4f} — "
               + ("no category is significantly over/under-subscribed." if p > 0.05 else "a significant deviation from proportional demand was found."))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Course Level Popularity Index")
        idx_lvl, chi2_l, p_l = popularity_index(filtered_full, "CourseLevel")
        fig = px.bar(x=idx_lvl.index, y=idx_lvl.values, labels={"x": "Course Level", "y": "Popularity Index"})
        fig.add_hline(y=100, line_dash="dash")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"χ² p = {p_l:.4f}")
    with col2:
        st.subheader("Course Type Popularity Index")
        idx_type, chi2_t, p_t = popularity_index(filtered_full, "CourseType")
        fig = px.bar(x=idx_type.index, y=idx_type.values, labels={"x": "Course Type", "y": "Popularity Index"})
        fig.add_hline(y=100, line_dash="dash")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"χ² p = {p_t:.4f}")

    st.subheader("Demographics × Course Preference Association (Cramér's V)")
    pairs = [
        ("AgeBand", "CourseCategory"), ("AgeBand", "CourseLevel"),
        ("Gender", "CourseCategory"), ("Gender", "CourseLevel"), ("Gender", "CourseType"),
    ]
    rows = []
    for a, b in pairs:
        ct = pd.crosstab(filtered_full[a], filtered_full[b])
        v = cramers_v(ct.values)
        rows.append({"Pairing": f"{a} × {b}", "Cramér's V": round(v, 3)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Values below ~0.1 indicate negligible association.")

    st.subheader("Age Band × Course Category Heatmap")
    heat = pd.crosstab(filtered_full["AgeBand"], filtered_full["CourseCategory"], normalize="index") * 100
    heat = heat.reindex(AGE_LABELS)
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                     labels=dict(color="% of age band's enrollments"))
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# PAGE 4 — ENGAGEMENT & CONCENTRATION ANALYSIS
# ----------------------------------------------------------------------------
elif page == "Engagement & Concentration Analysis":
    st.title("Engagement & Concentration Analysis")

    top_pct = st.slider("Top % of learners to inspect", 5, 50, 15, step=5)
    sorted_learners = filtered_learners.sort_values("Total_Enrollments", ascending=False)
    cutoff = max(1, int(len(sorted_learners) * top_pct / 100))
    top_share = sorted_learners.head(cutoff)["Total_Enrollments"].sum() / sorted_learners["Total_Enrollments"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Top {top_pct}% of learners", f"{cutoff:,} learners")
    c2.metric("Share of total enrollments", f"{top_share*100:.1f}%")
    light = (filtered_learners["Segment"] == "Light").sum()
    heavy = (filtered_learners["Segment"] == "Heavy").sum()
    c3.metric("Heavy-learner segment", f"{heavy:,} ({heavy/(heavy+light)*100:.1f}%)")

    st.subheader("Enrollment Count Distribution")
    fig = px.histogram(filtered_learners, x="Total_Enrollments", nbins=20,
                        labels={"Total_Enrollments": "Enrollments per Learner"})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Course Level Preference vs Rating (Reliable Learners, ≥3 Enrollments)")
    reliable = filtered_learners[filtered_learners["Total_Enrollments"] >= 3].copy()
    if len(reliable) > 0:
        # dominant level per learner
        dom = (
            filtered_full[filtered_full["UserID"].isin(reliable["UserID"])]
            .groupby("UserID")["CourseLevel"]
            .agg(lambda s: s.value_counts().idxmax())
        )
        reliable = reliable.set_index("UserID")
        reliable["Dominant_Level"] = dom
        summary = reliable.groupby("Dominant_Level", observed=True).agg(
            Avg_Rating=("Avg_Course_Rating", "mean"),
            Avg_Enrollments=("Total_Enrollments", "mean"),
            N=("Avg_Course_Rating", "count"),
        ).reindex(["Beginner", "Intermediate", "Advanced"])
        st.dataframe(summary.round(3), width="stretch")

        groups = [g["Avg_Course_Rating"].dropna().values for _, g in reliable.groupby("Dominant_Level", observed=True) if len(g) > 0]
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            st.info(f"**Kruskal-Wallis (rating by dominant level):** H = {h_stat:.2f}, p = {p_val:.5f}")
    else:
        st.warning("No learners with ≥3 enrollments in the current filter selection.")

    st.subheader("Top Learners")
    st.dataframe(
        sorted_learners.head(20)[["UserID", "Age", "Gender", "Total_Enrollments", "Distinct_Categories", "Total_Spend"]],
        width="stretch", hide_index=True,
    )

# ----------------------------------------------------------------------------
# PAGE 5 — ML INSIGHTS: CLUSTERING & PREDICTION 
# ----------------------------------------------------------------------------
elif page == "ML Insights: Clustering & Prediction":
    st.title("ML Insights: Clustering & Prediction")
    st.caption("Models recompute live on the currently filtered learner set.")

    st.subheader("K-Means Clustering")
    k = st.slider("Number of clusters (k)", 2, 8, 2)
    feature_cols = ["Total_Enrollments", "Distinct_Categories", "Avg_Level", "Paid_Share",
                     "Avg_Course_Rating", "Total_Spend", "Age", "Gender_Male"]
    cluster_df = filtered_learners.dropna(subset=feature_cols).copy()

    if len(cluster_df) >= k:
        X = StandardScaler().fit_transform(cluster_df[feature_cols])
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        cluster_df["Cluster"] = km.labels_
        sil = silhouette_score(X, km.labels_) if k > 1 and len(set(km.labels_)) > 1 else np.nan

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Silhouette score", f"{sil:.3f}" if not np.isnan(sil) else "n/a")
            fig = px.scatter(
                cluster_df, x="Total_Enrollments", y="Total_Spend", color=cluster_df["Cluster"].astype(str),
                hover_data=["Age", "Gender"], title="Clusters: Enrollments vs Spend",
            )
            st.plotly_chart(fig, width="stretch")
        with c2:
            gender_cluster = pd.crosstab(cluster_df["Cluster"], cluster_df["Gender"])
            st.write("Cluster × Gender crosstab")
            st.dataframe(gender_cluster, width="stretch")
            st.caption(
                "If clusters align closely with Gender, the split is demographic rather than "
                "behavioral — this was the key finding in the full project analysis at k=2."
            )
    else:
        st.warning("Not enough learners in the current filter selection to cluster.")

    st.markdown("---")
    st.subheader("Predictive Modeling: Can Demographics Predict Course Choice?")
    target = st.selectbox("Target variable", ["CourseCategory", "CourseLevel", "CourseType"])

    model_df = filtered_full.dropna(subset=["Age", "Gender_Male" if "Gender_Male" in filtered_full.columns else "Gender", target]).copy()
    if "Gender_Male" not in model_df.columns:
        model_df["Gender_Male"] = (model_df["Gender"] == "Male").astype(int)

    if model_df["UserID"].nunique() >= 10 and model_df[target].nunique() >= 2:
        X = model_df[["Age", "Gender_Male"]].values
        y = model_df[target].values
        groups = model_df["UserID"].values

        n_splits = min(5, model_df["UserID"].nunique())
        gkf = GroupKFold(n_splits=n_splits)

        baseline_acc, rf_acc, baseline_ll, rf_ll = [], [], [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=50, random_state=42).fit(X_train, y_train)

            baseline_acc.append(accuracy_score(y_test, dummy.predict(X_test)))
            rf_acc.append(accuracy_score(y_test, rf.predict(X_test)))

            classes = dummy.classes_
            baseline_ll.append(log_loss(y_test, dummy.predict_proba(X_test), labels=classes))
            rf_proba = rf.predict_proba(X_test)
            rf_ll.append(log_loss(y_test, rf_proba, labels=classes))

        results = pd.DataFrame({
            "Model": ["Baseline (Dummy)", "Random Forest"],
            "Accuracy": [np.mean(baseline_acc), np.mean(rf_acc)],
            "Log-Loss": [np.mean(baseline_ll), np.mean(rf_ll)],
        })
        st.dataframe(results.round(4), width="stretch", hide_index=True)

        if np.mean(rf_acc) <= np.mean(baseline_acc) * 1.02:
            st.success(
                f"Random Forest does not meaningfully outperform the baseline for predicting "
                f"**{target}** from Age/Gender — consistent with the project's overall finding "
                f"that demographics carry no predictive signal for course choice."
            )
        else:
            st.info(f"Random Forest shows some improvement over baseline for predicting {target}.")
    else:
        st.warning("Not enough data in the current filter selection to train a model.")

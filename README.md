# EduPro — Learner Demographics & Course Enrollment Behavior Analysis

Interactive Streamlit dashboard analyzing whether learner demographics (age, gender) shape course enrollment behavior on the EduPro online learning platform. Combines a Machine Learning phase (clustering, predictive modeling) with a Data Analytics phase (demographic breakdowns, popularity indices, association testing, behavioral concentration). Built as part of a Machine Learning internship project.

## Overview

EduPro offers courses across 12 subject categories, three difficulty levels, and both free and paid pricing. Stakeholders wanted to know whether a learner's age or gender meaningfully predicts what they enroll in — information that could inform targeted marketing, recommendations, or curriculum design.

**Approach:** deliberately sequenced Machine-Learning-first, then Data-Analytics-second. Unsupervised clustering and supervised prediction were used to test the demographics-drive-behavior hypothesis before any descriptive KPIs were built, so the DA phase could be interpreted in light of what the ML phase had already established.

## Key Findings

- **No evidence that age or gender predicts course choice.** Four independent methods — K-Means clustering, supervised classification (Random Forest vs. baseline), Cramér's V association testing, and a random-simulation baseline comparison — all converged on the same null result for CourseCategory, CourseLevel, and CourseType.
- **The one real demographic-looking split at k=2 clustering is actually a pure gender split** (1,520 Female vs. 1,480 Male, ~zero overlap) — but it carries no behavioral information: heavy learners split almost evenly across it (230 vs. 224).
- **Engagement, not demographics, is where the real structure lives.** The top 15% of learners generate 60.5% of all enrollments (Gini coefficient = 0.546). Heavy learners (9–16 enrollments, 15.1% of users) differ from light learners (1–4 enrollments, 84.9%) only in total spend — a mechanical consequence of volume, not a demographic trait.
- **No course category, type, or level is disproportionately popular** relative to its supply (Popularity Index 96–110 across the board; chi-square p-values all > 0.15).
- **One small but statistically real effect:** Beginner-dominant learners rate their courses modestly higher than Advanced-dominant learners (epsilon² = 0.0247, small effect; confirmed on a ≥3-enrollment "reliable" subset to rule out single-enrollment contamination).

Full methodology and results are documented in the accompanying technical report.

## Dashboard Modules

| Page | Contents |
|---|---|
| Learner Demographic Overview | Age distribution, gender split, KPI cards, enrollment concentration (Lorenz curve, Gini coefficient) |
| Age & Gender Enrollment Patterns | Avg enrollments by age band and gender with Kruskal-Wallis / Mann-Whitney significance tests, age × gender heatmap |
| Course Category & Level Popularity | Popularity indices (category, level, type) with chi-square tests, Cramér's V association table, age × category heatmap |
| Engagement & Concentration Analysis | Adjustable top-% concentration metric, enrollment distribution histogram, course-level vs. rating comparison, top-learners table |
| ML Insights: Clustering & Prediction | Live K-Means clustering (adjustable k) with silhouette score and cluster × gender crosstab; live Random Forest vs. baseline classifier comparison for predicting course choice from demographics |

All pages respond to global filters: age band, gender, course category, course level, and course type.

## KPIs Tracked

- **Total Enrollments & Avg Enrollments per Learner** — overall platform activity and its concentration (Gini coefficient)
- **Enrollments by Age Group** — tested for significance via Kruskal-Wallis
- **Gender Participation Ratio** — enrollment volume by gender, tested via Mann-Whitney U
- **Category Popularity Index** — enrollment share vs. supply share per course category (chi-square goodness-of-fit)
- **Level Preference Distribution** — enrollment share vs. supply share per course level
- **Cramér's V** — categorical association strength between demographics (age band, gender) and course preference (category, level, type)

## Tech Stack

- [Streamlit](https://streamlit.io/) — dashboard framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data processing
- [Plotly](https://plotly.com/python/) — interactive charts (bar, pie, heatmap, scatter, histogram)
- [scikit-learn](https://scikit-learn.org/) — K-Means clustering, Random Forest classification
- [SciPy](https://scipy.org/) — Kruskal-Wallis, Mann-Whitney U, chi-square, Cramér's V
- [openpyxl](https://openpyxl.readthedocs.io/) — Excel workbook loading

## Setup & Local Run

```bash
# clone the repo
git clone https://github.com/AlharathAjlan/Learner-Demographics-and-Course-Enrollment-Behavior-Analysis-on-EduPro.git
cd Learner-Demographics-and-Course-Enrollment-Behavior-Analysis-on-EduPro

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the app
python -m streamlit run app.py
```

The app expects an Excel workbook in the project folder with three sheets:

```
Users:        UserID, UserName, Age, Gender, Email
Courses:      CourseID, CourseName, CourseCategory, CourseType, CourseLevel,
              CoursePrice, CourseDuration, CourseRating
Transactions: TransactionID, UserID, CourseID, TransactionDate, Amount,
              PaymentMethod, TeacherID
```

Update the `DATA_PATH` variable near the top of `app.py` to match your filename.

## Data Notes

⚠️ **The raw dataset is not included in this repository** — see `.gitignore`.

**Known data characteristics** (documented and handled in `app.py`):
- The dataset is fully clean: zero duplicates, zero nulls, zero orphaned transactions, zero repeat (UserID, CourseID) pairs.
- Learner ages span 15–35 only; age bands in the dashboard (`<18`, `18-21`, `22-25`, `26-30`, `31-35`) are fitted to this actual range rather than the wider bands sometimes assumed in the original brief.
- Gender is recorded as a binary field (Male/Female) in this dataset.
- A Teachers sheet (TeacherID, TeacherName, Age, Gender, Expertise, YearsOfExperience, TeacherRating) exists in the source data but is outside this project's scope and is not loaded by the dashboard.

## Machine Learning Notes

The dashboard's ML module recomputes two things live on the currently filtered learner set, rather than showing static pre-computed results:
- **K-Means clustering** — adjustable k, with silhouette score and a cluster × gender crosstab so you can see directly whether any given cluster split is demographic or behavioral.
- **Predictive modeling** — trains a Random Forest (vs. a prior-probability baseline) on Age and Gender to predict CourseCategory, CourseLevel, or CourseType, using GroupKFold cross-validation (grouped by UserID to prevent leakage). Reproduces the project's central finding that demographics carry no real predictive signal for course choice.

## Project Structure

```
edupro_dashboard/
├── app.py              # Streamlit dashboard (5 modules)
├── requirements.txt
├── README.md
├── .gitignore
└── EduProOnlinePlatform.xlsx    
```

## Deliverables

- Combined ML + Data Analytics research paper (methodology, results, recommendations)
- This Streamlit dashboard (live analytics)
- Executive summary for platform stakeholders

## Status

All Core Modules and User Capabilities (age/gender/category/level/type filters, significance testing, live ML clustering and prediction) from the project requirements are implemented. The dashboard reproduces, on demand, every major finding from the research paper — including the well-corroborated null result on demographics and the real engagement-concentration and course-level-rating effects.

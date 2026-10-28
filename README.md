# Insurance Customer Risk Profiling | AXA Underwriting Context

**Academic group project · 2025 · AirData Six Consulting (five-member team)**

> **Project attribution and scope:** This repository documents a student **team** project conducted in an AXA insurance case-study context. It is not an AXA production system or a claim that the author independently built every component. The dataset was described in the group report as *fictional/synthetic customer records provided for this exercise*. No customer-level dataset is published here.

## Project overview

Developed and assessed machine-learning approaches for classifying insurance customers into **Low**, **Medium**, and **High** underwriting risk classes. The group explored demographic, policy, financial and behavioral features and reported model performance and limitations to a consulting-style audience.

**Business question:** Can historical-style customer records help distinguish underwriting risk groups, and what limitations should be communicated before such predictions influence decisions?

## Data and team setting

- **150,000** customer records documented in the final report: **142,500** labeled and **7,500** unlabeled.
- The target `Risk Profile` has three classes: Low, Medium, High; the initial labeled distribution is approximately **60% / 30% / 10%**.
- The report describes simulated/fictional customers; this repository does **not** include original data or sensitive project agreements.
- The project was completed collaboratively by five students; technical work was shared across the team.

## Workflow documented in the original project

1. Data understanding and exploratory analysis.
2. Missing-value imputation and feature engineering (policy duration and selected ratios).
3. Categorical encoding, normalization and class rebalancing.
4. Baseline comparisons: Random Forest, Logistic Regression, XGBoost, LightGBM and CatBoost.
5. 5-fold cross-validation, hyperparameter tuning and interpretation of feature importance.
6. Evaluation and discussion of model weaknesses, particularly the **Medium** class.

## Historical baseline performance (reported by the team)

| Model | Accuracy | F1 Low | F1 Medium | F1 High |
|---|---:|---:|---:|---:|
| Random Forest | **0.64** | 0.70 | **0.38** | **0.74** |
| Logistic Regression | 0.45 | 0.54 | 0.22 | 0.52 |
| XGBoost | 0.62 | 0.74 | 0.21 | 0.71 |
| LightGBM | 0.60 | 0.74 | 0.15 | 0.69 |
| CatBoost | 0.62 | 0.74 | 0.19 | 0.71 |

![Historical model comparison](images/baseline_model_comparison.png)

**Note:** The baseline F1 **0.74 for High** is **not** the tuned-model result. The final report separately lists a tuned validation accuracy of roughly **0.59**, macro F1 around **0.51**, and High-class F1 **0.69**. These should not be conflated.

## Technical limitations and methodological revision

The original notebook is retained only as a **historical reference**, with saved cell outputs removed. Its workflow oversampled with **SMOTE before splitting** train/validation and fit some preprocessing steps outside train-side CV. This can cause leakage and invalidate out-of-sample estimates. The original use of integer encoding for nominal categories and combined training/unlabeled feature processing also merits revision.

The separate [`risk_profiling_revised.ipynb`](notebooks/risk_profiling_revised.ipynb) documents a safer **unexecuted** example:

- splits **labeled rows before fitting preprocessing**;
- fits missing-value imputation/one-hot encoding **only on training folds**;
- performs training-only oversampling within an `imblearn` Pipeline;
- evaluates on untouched labeled validation rows;
- separates historical published figures from any future re-run results.

The revised example uses **RandomOverSampler** rather than the original SMOTE approach, deliberately avoiding synthetic interpolation across nominal one-hot dimensions. **No new results are claimed** without running the authorized data.

## Findings and caution

The team identified prior claims history as a prominent feature in the original feature-importance analysis. Predicting **Medium** risk remained notably weak. The report also discusses highly similar feature patterns across different labels and synthetic-data representativeness concerns. Results should be seen as an *academic analytical exercise*, **not a validated underwriting decision engine**. Additional validation, chronological feature availability checks, bias/fairness review and domain oversight would be essential for use in real underwriting.

## Repository contents

```text
notebooks/risk_profiling_original_reference.ipynb  Original historical workflow (outputs cleared)
notebooks/risk_profiling_revised.ipynb             New leakage-aware methodology example (not executed)
images/baseline_model_comparison.png             Chart redrawn from published team baseline table
reports/PROJECT_SUMMARY.md                       Public project brief and limitations
requirements.txt                                 Suggested runtime dependencies
```

## How to review/run

1. Read the historical findings above and the [project summary](reports/PROJECT_SUMMARY.md).
2. For a new reproducible experiment, obtain explicit permission to use the underlying dataset and place it locally at `data/Insurance Data.xlsx`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Start Jupyter from the repository root: `jupyter lab` (or equivalent). The notebooks use relative `data/` paths and do **not** download a dataset.
5. Run the **revised** notebook and independently review new results. The historical notebook is for inspection, not for production use.

## Technologies
Python · Pandas · NumPy · Scikit-learn · Imbalanced-learn · Random Forest · XGBoost · LightGBM · CatBoost · Optuna · Matplotlib · Seaborn

## Contribution statement

Technical development was **collaborative across the five-member team**, including data exploration, preprocessing, feature engineering, modeling, evaluation, documentation and presentation. The repository is a personal portfolio **record of a group project** and does not claim sole authorship. The team also received a **Best Presentation** recognition during the academic project activity (award certificate not included here).

## Publication and confidentiality

Original project reports, full presentation slides, internal documents and row-level data are intentionally omitted. Review and obtain any required permission before sharing original AXA-branded documentation or datasets. Model scores and methods are attributed here to the **2025 academic group report**, not independently reproduced by this repository.

# Public project summary

**Insurance Customer Risk Profiling (2025)** — academic group consulting project by AirData Six Consulting in an AXA underwriting context.

## Objective
Explore whether historical-style customer demographics, policy variables and claim information can support classification of Low, Medium and High risk labels.

## Inputs and process
150,000 fictional records documented in the team report, including 142,500 with known class labels and 7,500 without class labels. The original team workflow included EDA, missing-value treatment, feature creation, baseline modeling and 5-fold CV/tuning.

## Historical findings
Random Forest had baseline accuracy 0.64 and High-class F1 0.74. The tuned evaluation reported around 0.59 accuracy and 0.69 High F1, with weak performance on Medium. The historical workflow has potential data leakage (SMOTE before split). See README for limitations and separate unexecuted revised pipeline.

## Contribution
Work was performed collaboratively by a five-member academic team. No individual-ownership or production-system claims.

## Privacy
No underlying dataset, internal agreements or original company-branded full presentation are distributed here.

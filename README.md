
# Insurance Customer Risk Profiling | AXA Underwriting Case Study

**Data Science Consulting | Academic Group Project | 2025**  
**Team:** AirData Six Consulting (5 members)  
**Domain:** Insurance Analytics & Underwriting  
**Focus:** Data Analytics, Machine Learning, Risk Classification, and Model Interpretation

---

## 1. Project Overview

This project explored the application of data analytics and machine learning to classify insurance customers into **Low, Medium, and High risk categories** in the context of underwriting.

Conducted as an academic data science consulting project using a fictional customer dataset provided for an AXA case study, the project focused on analyzing customer characteristics, identifying potential risk indicators, comparing classification algorithms, and evaluating the reliability of model predictions.

The team developed an end-to-end analytical workflow covering exploratory data analysis, data preprocessing, feature engineering, model development, performance evaluation, and interpretation of findings.

### Project at a Glance

| Indicator | Details |
|---|---|
| Industry | Insurance |
| Business Area | Underwriting & Risk Analytics |
| Dataset | 150,000 fictional customer records |
| Labeled Data | 142,500 records |
| Unlabeled Data | 7,500 records |
| Target Variable | Risk Profile (Low, Medium, High) |
| Models Compared | Random Forest, Logistic Regression, XGBoost, LightGBM, CatBoost |
| Selected Historical Baseline | Random Forest |
| Baseline Accuracy | 0.64 |
| Baseline High-Risk F1-Score | 0.74 |
| Project Type | Academic Team Project |

**Project scope:** This repository documents the team's academic analysis rather than an AXA production system. Reported performance metrics originate from the original group project and have not been independently reproduced using the revised pipeline.

---

## 2. Business Problem

Insurance underwriting involves assessing the risks associated with prospective policyholders. Customer characteristics, policy information, financial indicators, and claims history can provide useful information for understanding different risk profiles.

However, identifying risk categories from customer data presents several analytical challenges.

First, customers with different risk classifications may share similar demographic, financial, or policy characteristics. Second, an imbalanced distribution of risk classes can influence model predictions. Finally, classification errors may have different implications depending on the actual and predicted risk categories.

The project therefore addressed the following business question:

> How effectively can customer and policy data distinguish insurance risk categories, and what analytical limitations should be considered before using these predictions to support underwriting decisions?

### Project Objectives

1. Analyze customer characteristics and identify patterns across insurance risk categories.
2. Develop and compare machine learning models for multiclass risk classification.
3. Identify variables highlighted by model-based feature importance.
4. Evaluate classification performance across individual risk categories.
5. Translate analytical findings into considerations relevant to underwriting risk assessment.

---

## 3. Dataset Understanding

The project used **150,000 fictional insurance customer records** described in the original academic report.

The dataset contained demographic, financial, behavioral, and insurance-related attributes, including:

- Customer demographics, such as age, occupation, and marital status.
- Financial information, including credit score and income level.
- Insurance policy characteristics, such as premium amount, coverage amount, and deductible.
- Customer interaction and behavioral indicators.
- Insurance claim history and other policy-related information.

The target variable, `Risk Profile`, consisted of three categories: **Low, Medium, and High**.


### Risk Class Distribution

The labeled dataset contained three risk categories with the following distribution:

| Risk Category | Proportion |
|---|---:|
| Low | 59.9% |
| Medium | 29.9% |
| High | 10.1% |

Low-risk customers represented the majority of the labeled dataset, while High-risk customers accounted for the smallest proportion.

**Analytical Insight**

The uneven class distribution created a challenge for classification, particularly when evaluating less-represented risk categories. A model could achieve reasonable overall accuracy while still performing poorly in identifying specific customer risk groups.

**Business Implication**

For underwriting-related analysis, performance should be assessed using class-specific precision, recall, and F1-scores. This helps identify whether the model is overlooking higher-risk customers or incorrectly assigning elevated risk to lower-risk customers.

**Analytical Insight**

The unequal class distribution required particular attention during model development and evaluation.

A model achieving reasonable overall accuracy could still perform poorly on less-represented categories. Therefore, the team examined class-specific metrics rather than relying exclusively on overall accuracy.

### Exploratory Data Analysis

EDA was conducted to examine customer distributions, missing values, relationships between variables, and differences across risk categories.

The analysis identified several challenges:

- Missing values in selected customer attributes.
- Unequal distributions across the target classes.
- Similar feature distributions among customers with different risk labels.
- Limited discriminatory patterns in several numerical and categorical variables.

These findings suggested that distinguishing risk categories would require more than selecting a complex classification algorithm.

In particular, the overlap between customer characteristics created challenges in separating Medium-risk customers from other risk groups.

---

## 4. Analytical Methodology

The team followed a structured analytical workflow from initial data exploration to model evaluation.

### 4.1 Data Understanding and Preparation

The initial stage focused on understanding the dataset structure, identifying variable types, examining missing values, and assessing data quality.

Categorical missing values were handled using frequency-based imputation in the historical workflow.

### 4.2 Feature Engineering

Additional features were created to capture information related to customer policies and financial characteristics.

Examples included:

| Engineered Feature | Description |
|---|---|
| Policy Duration | Duration associated with the insurance policy |
| Premium Coverage Ratio | Relationship between premium and coverage amounts |
| Premium Age Ratio | Premium amount relative to customer age |
| Age Credit Ratio | Relationship between customer age and credit score |

These features were developed to explore whether relationships between existing variables could provide additional information for risk classification.

### 4.3 Data Preprocessing

The historical workflow included:

- Missing-value imputation.
- Categorical encoding.
- Numerical feature scaling.
- Class balancing using SMOTE.

**Methodological note:** The original notebook applied SMOTE before separating training and validation data. This introduces potential data leakage and means historical performance estimates should be interpreted cautiously. A separate revised pipeline is included in this repository to demonstrate a more appropriate preprocessing and validation structure.

### 4.4 Model Development

Five classification algorithms were evaluated:

1. Random Forest
2. Logistic Regression
3. XGBoost
4. LightGBM
5. CatBoost

The models were compared using overall accuracy and F1-scores for each risk category.

### 4.5 Model Optimization and Evaluation

The historical project also included:

- Five-fold cross-validation.
- Hyperparameter optimization using Optuna.
- Classification performance evaluation.
- Confusion matrix analysis.
- Feature importance interpretation.

The team examined both the overall results and the differences in predictive performance across Low, Medium, and High risk categories.

---

## 5. Model Performance

### 5.1 Historical Baseline Comparison

The following results were documented in the original team report.

| Model | Accuracy | F1 Low | F1 Medium | F1 High |
|---|---:|---:|---:|---:|
| **Random Forest** | **0.64** | 0.70 | **0.38** | **0.74** |
| Logistic Regression | 0.45 | 0.54 | 0.22 | 0.52 |
| XGBoost | 0.62 | 0.74 | 0.21 | 0.71 |
| LightGBM | 0.60 | 0.74 | 0.15 | 0.69 |
| CatBoost | 0.62 | 0.74 | 0.19 | 0.71 |

![Historical Model Performance Comparison](images/baseline_model_comparison.png)

Random Forest was selected as the strongest historical baseline based on its overall classification results and comparatively better performance on the Medium-risk category.

Although XGBoost, LightGBM, and CatBoost achieved higher F1-scores for the Low-risk category, Random Forest provided the strongest F1-score for both Medium-risk and High-risk customers among the models evaluated.

### 5.2 Performance After Optimization

The final academic report separately documented the evaluation of an optimized Random Forest model.

| Evaluation Metric | Reported Result |
|---|---:|
| Validation Accuracy | Approximately 0.59 |
| Macro F1-Score | Approximately 0.51 |
| F1-Score — Low | 0.70 |
| F1-Score — Medium | 0.14 |
| F1-Score — High | 0.69 |

The optimized model continued to show difficulty identifying Medium-risk customers.

These results are distinct from the baseline comparison. The High-risk F1-score of **0.74** belongs to the historical baseline, whereas **0.69** was reported for the tuned validation evaluation.

The evaluation results should also be interpreted in light of the preprocessing and validation limitations described later in this repository.

---

## 6. Key Analytical Findings

### 6.1 Predictive Performance Differed Across Risk Categories

The model comparison demonstrated that classification performance varied substantially among the three risk categories.

In the historical Random Forest baseline:

- High-risk customers achieved an F1-score of 0.74.
- Low-risk customers achieved an F1-score of 0.70.
- Medium-risk customers achieved an F1-score of 0.38.

This indicates that the model was less effective at distinguishing the Medium-risk category.

The difference is important because overall accuracy alone does not reveal how well a model identifies each risk group.

**Business Insight**

For underwriting-related analysis, classification performance should be assessed separately for each risk category.

Misclassifying a High-risk customer as Low-risk may have different consequences from classifying a Low-risk customer as High-risk.

Therefore, model evaluation should consider class-level precision, recall, and the implications of different classification errors rather than relying solely on overall accuracy.

### 6.2 Claims History Was a Prominent Feature in Model Interpretation

The historical Random Forest feature importance analysis highlighted `Previous Claims History` as a prominent predictor.

Other features appearing among the leading variables included:

- Location
- Deductible
- Coverage Amount

These attributes represent different aspects of a customer's insurance history, geographic information, and policy characteristics.

**Business Insight**

Claims-related and policy-related variables may provide useful signals when evaluating customer risk profiles.

However, the historical feature importance results only describe patterns learned by the model from the academic dataset. They do not establish causal relationships or demonstrate that the same variables would have equivalent predictive value in real underwriting data.

The findings also suggest the importance of verifying whether relevant features are available and appropriate at the time an underwriting decision is made.

### 6.3 Similar Customer Characteristics Complicated Risk Classification

The team observed that some customers with different risk labels had similar or nearly identical combinations of predictor values.

This made it difficult for the classification models to establish clear distinctions between certain customer groups.

The issue was particularly visible in the Medium-risk category, which consistently showed weaker predictive performance.

**Business Insight**

Model performance depends not only on the algorithm but also on whether the available data contains enough information to distinguish the target categories.

When customer profiles overlap substantially, additional data quality assessment, feature validation, and review of class definitions may be necessary before developing a reliable classification system.

This finding highlights the importance of understanding the business meaning and quality of the data before attempting further model optimization.

---

## 7. Implications for Insurance Underwriting

The project provides several considerations for applying predictive analytics to underwriting-related problems.

### Risk Assessment

Claims history and policy characteristics emerged as potentially useful analytical indicators within the historical dataset.

These findings suggest areas worth investigating when designing a more comprehensive risk assessment process.

### Decision Support

The difference in predictive performance across risk categories indicates that automated classification should not be evaluated using a single aggregate metric.

An underwriting-related decision support workflow would need to consider class-specific prediction errors and their business consequences.

### Data Quality and Reliability

The difficulty in separating Medium-risk customers demonstrates the importance of having informative, consistent, and sufficiently representative features.

Further investigation of the data and risk-label definitions would be necessary before applying the approach beyond an academic setting.

### Model Governance

Insurance risk classification can affect consequential decisions for customers.

Any operational application would require additional validation, appropriate human oversight, bias and fairness assessment, and a review of how model outputs would be used in underwriting decisions.

**Overall Takeaway**

The central finding of this project is that model selection alone does not determine the reliability of risk classification. Understanding class-level performance, feature quality, and the business implications of prediction errors is equally important.

---

## 8. Methodological Limitations and Improvements

This repository preserves the historical project results while documenting methodological issues identified during subsequent review.

### Limitations of the Original Workflow

**Data Leakage**

The original notebook applied SMOTE before the train-validation split. This could introduce information from validation samples into the training process and affect the reliability of performance estimates.

Certain preprocessing transformations were also performed outside properly isolated training folds.

**Categorical Encoding**

The historical workflow used integer-based encoding for categorical variables. When nominal categories are encoded as integers, the resulting values may imply an artificial order.

**Class Overlap**

Customer records with similar predictor patterns sometimes had different risk labels, making reliable classification difficult.

**Synthetic Dataset**

The dataset was described as fictional in the original group report. Findings from this academic dataset cannot automatically be generalized to real insurance customers.

**Model Evaluation**

Historical baseline results and tuned-model evaluation results were produced at different stages and must be interpreted separately.

### Revised Methodological Pipeline

To document a more robust approach, this repository includes an additional notebook illustrating a revised modeling workflow.

The revised pipeline:

1. Separates labeled training and validation data before fitting preprocessing transformations.
2. Fits imputation and categorical encoding within the training workflow.
3. Uses one-hot encoding for nominal categorical features.
4. Performs oversampling within the training pipeline.
5. Evaluates predictions using validation data that remain separate from the training process.

The revised example uses `RandomOverSampler` instead of the original SMOTE approach.

**Important:** The revised notebook has not been executed using the original authorized dataset. No new model performance results are claimed.

---

## 9. Repository Structure

| File / Folder | Description |
|---|---|
| [`notebooks/risk_profiling_original_reference.ipynb`](notebooks/risk_profiling_original_reference.ipynb) | Historical notebook documenting the original team workflow, with saved outputs removed |
| [`notebooks/risk_profiling_revised.ipynb`](notebooks/risk_profiling_revised.ipynb) | Revised modeling pipeline illustrating improved preprocessing and validation practices |
| [`images/`](images/) | Visualizations derived from historical project results |
| [`reports/PROJECT_SUMMARY.md`](reports/PROJECT_SUMMARY.md) | Public project summary and methodological notes |
| [`requirements.txt`](requirements.txt) | Python dependencies |

### Running the Revised Notebook

The repository does not include the original AXA case-study dataset.

To execute the revised workflow:

1. Obtain permission to use the underlying project dataset.
2. Store the authorized dataset locally in the `data/` folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open `notebooks/risk_profiling_revised.ipynb`.
5. Run the notebook and independently review the resulting evaluation metrics.

Any future experimental results should be reported separately from the historical group findings.

---

## 10. Technologies and Tools

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Random Forest, Logistic Regression |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Model Optimization | Optuna |
| Class Balancing | Imbalanced-learn |
| Visualization | Matplotlib, Seaborn |
| Development Environment | Jupyter Notebook, Google Colab |
| Version Control | GitHub |

---

## 11. Team Collaboration and Recognition

### Project Team

**AirData Six Consulting — Five-Member Academic Team**

The project was completed collaboratively by five Data Science Technology students from Universitas Airlangga.

Technical development was shared across the team, including:

- Data exploration and preprocessing.
- Feature engineering.
- Machine learning model development.
- Model evaluation and interpretation.
- Documentation and presentation of analytical findings.

This repository serves as a personal portfolio record of a collaborative academic project and does not claim sole authorship of the work.

### Project Recognition

**Best Presentation — Academic Data Science Consulting Project**

The team received Best Presentation recognition during the academic project activity.

The recognition reflects the team's communication and presentation of its analytical findings. It is presented as a team achievement rather than an individual award.

---

## 12. Project Scope and Data Confidentiality

This repository presents the methods, historical results, and analytical findings of an academic case study.

The original dataset, internal project documents, complete group report, and AXA-branded presentation materials are not included.

The visualizations and numerical results are based on historical team documentation and have not been independently reproduced using the revised notebook.

The project does not claim:

- Deployment of a production underwriting system.
- Verified improvements in AXA's underwriting performance.
- Application of the model to real insurance customers.
- Independently validated performance using the revised pipeline.

The repository is intended to demonstrate collaborative analytical work, machine learning experimentation, model interpretation, and business-oriented evaluation of insurance risk classification.

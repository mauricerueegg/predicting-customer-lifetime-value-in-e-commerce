# 00 – Project Overview

## Assignment Context

**Course:** Fundamentals of Python Applications in Data Science (ZHAW, 8th Semester)
**Focus Area:** E — "Supervised Learning: Regression and XAI"
**Topic:** Predicting Customer Lifetime Value (CLV) in E-Commerce
**Student:** Maurice Rueegg
**Submission:** One Jupyter Notebook (.ipynb) + raw dataset CSV

---

## What is Customer Lifetime Value?

Customer Lifetime Value (CLV) is the total revenue a business can expect from a single customer account throughout their entire relationship. It is a core metric in e-commerce and CRM because:

- It identifies which customers are most valuable
- It guides decisions on marketing spend and customer acquisition cost
- It informs retention strategies (which customers to retain vs. let churn)
- It enables customer segmentation and personalisation

In this project, CLV is modelled as a **regression target** — a continuous numeric value in Swiss Francs (CHF).

---

## Project Objective

> Build regression models to predict `customer_lifetime_value_chf` from behavioural and demographic customer attributes. Use SHAP values to explain what drives predictions.

**Specific goals:**
1. Engineer meaningful features from raw behavioural data
2. Train at least two regression models (we train three: Linear Regression, Random Forest, XGBoost)
3. Evaluate and compare model performance using RMSE, MAE, and R²
4. Use SHAP values to explain both global patterns and individual predictions

---

## Dataset

The dataset is **synthetically generated** using a fixed random seed (808), ensuring full reproducibility. It simulates a realistic e-commerce customer base with 1100 records and 11 columns.

- **File:** `01_data/topic_E2_customer_lifetime_value_raw.csv`
- **Generation code:** Section 0 of `notebook.ipynb`
- **Full schema:** see `docs/01_data_description.md`

---

## Deliverables

| Deliverable | Description |
|-------------|-------------|
| `notebook.ipynb` | Fully executable Jupyter Notebook, top-to-bottom |
| `01_data/topic_E2_customer_lifetime_value_raw.csv` | Raw dataset (original, unmodified output from generation code) |
| `docs/` | Supporting documentation for all design decisions |

---

## Mandatory Notebook Sections

The notebook must follow this exact structure:

| # | Section | Content |
|---|---------|---------|
| 0 | Data Generation | Generate and save the raw dataset |
| 1 | Problem Framing | Define the task, goal, and target variable |
| 2 | Data Understanding | EDA: distributions, correlations, outliers |
| 3 | Method / Approach | Justify feature engineering, model choices, evaluation strategy |
| 4 | Implementation | Feature engineering, model training, evaluation, SHAP |
| 5 | Results, Validation & Robustness | Compare models, validate, assess biases |
| 6 | Interpretation & Critical Reflection | Interpret SHAP, discuss limitations |
| 7 | AI Usage Documentation | Document how AI tools were used |

---

## Tools & Libraries

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Linear Regression, Random Forest, preprocessing, metrics |
| `xgboost` | XGBoost Regressor |
| `shap` | SHAP explanations (global + local) |

---

## Project Timeline & Workflow

1. **Setup:** Update CLAUDE.md, create `/docs/` documentation
2. **Data Exploration:** EDA (Section 2)
3. **Feature Engineering:** Transform raw features (Section 4.1)
4. **Modelling:** Train 3 models (Section 4.2–4.3)
5. **Evaluation:** Compare metrics and residuals (Section 4.4 + Section 5)
6. **XAI:** SHAP analysis (Section 4.5)
7. **Reflection:** Interpret results and limitations (Section 6)
8. **Documentation:** AI usage section (Section 7)

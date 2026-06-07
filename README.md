# Predicting Customer Lifetime Value in E-Commerce

Regression models to predict customer lifetime value (CLV) from behavioural and demographic data, with SHAP-based explainability.

## Project Context

**Course:** Fundamentals of Python Applications in Data Science — ZHAW, Spring 2026  
**Focus Area:** E — Supervised Learning: Regression and XAI  
**Author:** Maurice Rueegg

## Objective

1. Engineer meaningful features from raw customer behaviour data
2. Train and compare three regression models (Linear Regression, Random Forest, XGBoost)
3. Evaluate performance using RMSE, MAE, and R²
4. Use SHAP values to explain global patterns and individual predictions

## Dataset

| Property | Value |
|----------|-------|
| File | `01_data/topic_E2_customer_lifetime_value_raw.csv` |
| Rows | 1100 |
| Features | 10 (numeric + categorical) |
| Target | `customer_lifetime_value_chf` (continuous, CHF) |
| Generation seed | `np.random.seed(808)` |

Features include customer tenure, purchase count, average order value, return rate, website visits, email click rate, customer segment, preferred device, support tickets, and discount usage rate.

## Project Structure

```
.
├── notebook.ipynb          # Main deliverable — full analysis pipeline
├── 01_data/
│   └── topic_E2_customer_lifetime_value_raw.csv
├── docs/
│   ├── 00_project_overview.md
│   ├── 01_data_description.md
│   ├── 02_feature_engineering.md
│   ├── 03_model_selection.md
│   ├── 04_evaluation_strategy.md
│   ├── 05_shap_methodology.md
│   └── 06_ai_usage_template.md
└── README.md
```

## Tech Stack

- **Python 3.12**
- pandas, numpy — data manipulation
- matplotlib, seaborn — visualisation
- scikit-learn — Linear Regression, Random Forest, preprocessing, metrics
- xgboost — XGBoost Regressor
- shap — SHAP explanations

## Getting Started

```bash
# Clone the repository
git clone <repository-url>
cd predicting-customer-lifetime-value-in-e-commerce

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap

# Run the notebook
jupyter lab notebook.ipynb
```

## Models and Results

Three regression models were trained on an 80/20 train-test split with `random_state=29`.

| Model | RMSE (CHF) | MAE (CHF) | R² |
|-------|-----------|----------|-----|
| **Linear Regression** | **124.15** | **102.34** | **0.88** |
| XGBoost | 129.91 | 104.26 | 0.87 |
| Random Forest | 142.63 | 114.44 | 0.84 |

Linear Regression outperformed both tree-based models because the CLV generating formula is predominantly linear. The theoretical noise floor is ~120 CHF (irreducible Gaussian noise in the data generation process), meaning Linear Regression nearly reached the best achievable performance.

## SHAP Explainability

SHAP (SHapley Additive exPlanations) analysis is applied to all three models using `LinearExplainer` for Linear Regression and `TreeExplainer` for Random Forest and XGBoost. The analysis includes global feature importance (beeswarm and bar plots), local explanations (waterfall plots for individual predictions), and dependence plots for top features.

## Documentation

Detailed design decisions and methodology are documented in the [`docs/`](docs/) folder:

- **[Project Overview](docs/00_project_overview.md)** — context and deliverables
- **[Data Description](docs/01_data_description.md)** — dataset schema and generation logic
- **[Feature Engineering](docs/02_feature_engineering.md)** — engineered features and encoding
- **[Model Selection](docs/03_model_selection.md)** — model choices and comparison
- **[Evaluation Strategy](docs/04_evaluation_strategy.md)** — metrics and validation approach
- **[SHAP Methodology](docs/05_shap_methodology.md)** — SHAP theory and interpretation
- **[AI Usage](docs/06_ai_usage_template.md)** — AI tool usage documentation

# 04 – Evaluation Strategy

## Overview

Evaluation answers: **How well does each model predict CLV?** We use a combination of hold-out test set evaluation and cross-validation to get reliable, bias-corrected performance estimates.

---

## Train / Test Split

We use an **80/20 stratified random split**:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

| Set | Size | Purpose |
|-----|------|---------|
| Training set | 880 samples (80%) | Fit all models |
| Test set | 220 samples (20%) | Final evaluation — untouched during training |

**Why 80/20?** With 1100 samples, 220 test samples is enough for stable metric estimates. A larger training set helps tree models learn more complex patterns.

**Important:** The StandardScaler is fitted **only on the training set** and applied to both. This prevents data leakage — the scaler must not "see" test data distributions.

---

## Evaluation Metrics

### 1. RMSE – Root Mean Squared Error

```
RMSE = sqrt( mean( (y_pred - y_true)^2 ) )
```

- **Unit:** CHF (same as target)
- **Interpretation:** On average, predictions deviate from actual CLV by X CHF
- **Sensitivity:** Penalises large errors more than MAE (due to squaring)
- **Primary metric** — most relevant for business impact (large CLV errors are costly)

### 2. MAE – Mean Absolute Error

```
MAE = mean( |y_pred - y_true| )
```

- **Unit:** CHF
- **Interpretation:** Average absolute prediction error in CHF
- **Sensitivity:** Less sensitive to outliers than RMSE
- **Complementary metric** — gives a more conservative, robust picture of typical error

### 3. R² – Coefficient of Determination

```
R² = 1 - SS_res / SS_tot
```

- **Range:** 0 (null model) to 1 (perfect fit)
- **Interpretation:** Proportion of CLV variance explained by the model
- **Useful for:** Comparing models regardless of scale; communicating model quality to non-technical stakeholders

### Theoretical Bounds

The CLV generating process includes Gaussian noise `N(0, σ=120)`. This means:

- **Minimum achievable RMSE** ≈ 120 CHF (irreducible noise floor)
- **Any model with RMSE > 300 CHF** is significantly under-fitting
- **Expected R² for a well-fitted model:** 0.85–0.95

---

## Cross-Validation

We use **5-fold cross-validation** on the training set to:

1. Get a more reliable performance estimate than a single train/test split
2. Detect overfitting (large gap between CV score and test score indicates overfitting)
3. Compare models fairly before committing to the test set evaluation

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)
rmse_cv = -cv_scores.mean()
rmse_cv_std = cv_scores.std()
```

**Why 5 folds?** A standard choice balancing bias/variance of the CV estimate. With 880 training samples, 5 folds gives 176 validation samples per fold — sufficient for stable estimates.

---

## Residual Analysis

Beyond scalar metrics, we inspect **residuals** = `y_pred - y_true`:

### Residual Plots
- **Residuals vs. predicted values:** Should be randomly scattered around zero. Patterns indicate model misspecification (e.g., non-linearity not captured).
- **Histogram of residuals:** Should be approximately normal for a well-specified model.
- **Q-Q plot (optional):** Compares residual quantiles to a theoretical normal distribution.

### What to look for:
- **Funnel shape** (residuals grow with predicted value): Heteroscedasticity — variance of errors is not constant. May indicate the need for log-transforming the target.
- **Systematic bias** for certain ranges: Model under-/over-predicts for specific CLV ranges (e.g., premium customers).

---

## Actual vs. Predicted Plots

A scatter plot of `y_test` vs. `y_pred` with the perfect prediction line (slope=1):
- Points close to the diagonal → good predictions
- Systematic deviation below/above → model bias
- Spread around the diagonal → random error (noise)

---

## Model Comparison

Final comparison table:

| Model | RMSE (test) | MAE (test) | R² (test) | RMSE (CV ± std) |
|-------|------------|-----------|----------|----------------|
| Linear Regression | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... |

**Decision criterion:** The model with the lowest test RMSE and stable CV score (small std) is considered the best performer. If CV and test scores diverge significantly, this is flagged as a potential overfitting or data split artefact.

---

## Data Leakage Prevention Checklist

- [ ] StandardScaler fitted only on training data
- [ ] Test set never used for model selection or hyperparameter tuning
- [ ] Cross-validation applied only within the training set
- [ ] Random seed fixed (`random_state=42`) for all stochastic steps

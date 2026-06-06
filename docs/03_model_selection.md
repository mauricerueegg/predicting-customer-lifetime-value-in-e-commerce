# 03 – Model Selection

## Requirement

The project requires **at least two regression models** with **different assumptions** (e.g., linear vs. non-linear). We train three models to provide a richer comparison:

1. Linear Regression (linear baseline)
2. Random Forest Regressor (non-linear ensemble)
3. XGBoost Regressor (non-linear gradient boosting)

---

## Model 1: Linear Regression

**Library:** `sklearn.linear_model.LinearRegression`

### Assumptions
- The relationship between features and CLV is linear
- Features are independent (no multicollinearity issues for prediction, though coefficients may be unreliable)
- Residuals are normally distributed with constant variance (homoscedasticity)

### Why include it?
- **Interpretability:** Coefficients directly indicate each feature's contribution
- **Baseline:** Establishes a lower bound for model performance
- **Reveals linear structure:** The CLV generating formula is largely linear — a good LR should explain a substantial proportion of variance
- **SHAP compatibility:** `shap.LinearExplainer` provides exact Shapley values for linear models

### Limitations
- Cannot capture interaction effects (e.g., premium order bonus) without explicit feature engineering
- Assumes constant feature effects across all segments (no conditional logic)
- The threshold penalty for `discount_usage_rate > 0.75` is invisible to LR

### Configuration
```python
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
```

---

## Model 2: Random Forest Regressor

**Library:** `sklearn.ensemble.RandomForestRegressor`

### Assumptions
- No assumptions about the functional form of the relationship
- Each tree learns a partition of the feature space
- Ensemble averaging reduces variance (overfitting risk)

### Why include it?
- **Handles non-linearities natively:** Finds optimal splits for threshold effects (like `discount_usage_rate > 0.75`)
- **Captures interaction effects:** The premium order bonus (`segment == "premium" & avg_order_value`) can be discovered through split combinations
- **Robust:** Ensemble averaging over many trees reduces overfitting
- **SHAP compatibility:** `shap.TreeExplainer` provides exact Shapley values efficiently

### Limitations
- **No extrapolation:** Cannot predict CLV values outside the training range
- **Black box:** Individual trees are hard to interpret (mitigated by SHAP)
- **Memory intensive:** 200 trees × 1100 samples is manageable but larger datasets would be slow

### Configuration
```python
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=200, random_state=29, n_jobs=-1)
model_rf.fit(X_train, y_train)
```

**Key hyperparameters:**
- `n_estimators=200`: Balance between performance and training time
- `random_state=29`: Ensures reproducibility
- `n_jobs=-1`: Use all CPU cores

---

## Model 3: XGBoost Regressor

**Library:** `xgboost.XGBRegressor`

### Assumptions
- Learns residuals iteratively — each new tree corrects errors of the previous ensemble
- Gradient descent in function space (boosting framework)

### Why include it?
- **Best-in-class for tabular data:** XGBoost consistently outperforms Random Forests on structured datasets in practice
- **Handles interactions and thresholds** like Random Forest, but with better generalisation
- **Regularisation built-in:** `reg_alpha` (L1) and `reg_lambda` (L2) prevent overfitting
- **SHAP native support:** XGBoost has native integration with the SHAP library (fastest exact Shapley computation)
- **Can detect the heavy discount threshold:** The step function at `discount_usage_rate > 0.75` is naturally captured by a tree split

### Limitations
- **More hyperparameters:** Requires careful tuning to avoid overfitting (we use conservative defaults)
- **Less interpretable than LR** (mitigated by SHAP)
- **Boosting is sequential** — slower to train than Random Forest for the same number of trees

### Configuration
```python
from xgboost import XGBRegressor
model_xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=29,
    verbosity=0
)
model_xgb.fit(X_train, y_train)
```

**Key hyperparameters:**
- `n_estimators=200`: Enough trees for convergence at learning_rate=0.05
- `learning_rate=0.05`: Conservative step size — better generalisation
- `max_depth=5`: Limits tree depth to prevent overfitting
- `subsample=0.8, colsample_bytree=0.8`: Stochastic sampling for regularisation

---

## Comparison Summary

| Aspect | Linear Regression | Random Forest | XGBoost |
|--------|-------------------|---------------|---------|
| Model type | Linear | Non-linear ensemble | Non-linear boosting |
| Handles interactions | Only via explicit FE | Yes (natively) | Yes (natively) |
| Handles thresholds | No | Yes | Yes |
| Overfitting risk | Low | Medium | Medium (regularised) |
| SHAP method | LinearExplainer | TreeExplainer | TreeExplainer |
| Expected performance | Moderate | High | Highest |
| Interpretability | High (coefficients) | Low (mitigated by SHAP) | Low (mitigated by SHAP) |

---

## Why Not Other Models?

| Model | Reason not chosen |
|-------|------------------|
| Ridge / Lasso Regression | Similar to LR but with regularisation — adds little value when LR already serves as baseline |
| SVR (Support Vector Regression) | No efficient exact SHAP support; slow on this data size |
| Neural Network | Excessive complexity for 1100 samples; SHAP requires costly KernelExplainer |
| Decision Tree | Included implicitly via RF/XGB; single trees overfit on small datasets |
| Gradient Boosting (sklearn) | XGBoost is a faster, better-regularised alternative |

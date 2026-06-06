# 05 – SHAP Methodology

## What is SHAP?

SHAP (SHapley Additive exPlanations) is a framework for **explaining individual predictions** of any machine learning model. It is grounded in cooperative game theory (Shapley values) and provides the only attribution method that satisfies the three key fairness axioms:

1. **Efficiency:** SHAP values for all features always sum to the prediction minus the base value
2. **Symmetry:** Features with identical contributions receive identical SHAP values
3. **Dummy:** Features with no contribution receive zero SHAP value
4. **Additivity:** Contributions from independent models can be combined

### SHAP Value Interpretation

For a single prediction:
```
prediction = base_value + SHAP_feature_1 + SHAP_feature_2 + ... + SHAP_feature_n
```

- `base_value` = mean CLV over all training samples
- `SHAP_feature_i > 0` = this feature pushes the prediction **above** the mean
- `SHAP_feature_i < 0` = this feature pushes the prediction **below** the mean

---

## Why SHAP for CLV Prediction?

1. **Model-agnostic comparability:** SHAP allows us to compare feature importance across Linear Regression, Random Forest, and XGBoost on a common scale
2. **Reveals interaction effects:** SHAP interaction values show how pairs of features jointly influence predictions (e.g., segment × order value)
3. **Satisfies XAI requirements:** The project explicitly requires explainability of predictions — SHAP is the gold standard
4. **Detects non-linear patterns:** SHAP dependence plots reveal threshold effects (e.g., `discount_usage_rate > 0.75`)

---

## SHAP Explainer Types

### `shap.LinearExplainer` — for Linear Regression

```python
explainer_lr = shap.LinearExplainer(model_lr, X_train_scaled)
shap_values_lr = explainer_lr.shap_values(X_test_scaled)
```

- **How it works:** Uses the model's coefficients × feature deviations from mean. Exact computation.
- **Masking:** Uses feature correlations to handle correlated features correctly (default: `masker=shap.maskers.Independent`)
- **Speed:** Very fast (O(n × features))

### `shap.TreeExplainer` — for Random Forest & XGBoost

```python
explainer_rf = shap.TreeExplainer(model_rf)
shap_values_rf = explainer_rf.shap_values(X_test)

explainer_xgb = shap.TreeExplainer(model_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_test)
```

- **How it works:** Traverses tree paths to compute exact Shapley values in polynomial time
- **Advantage over SHAP KernelExplainer:** Exact (not approximated), orders of magnitude faster
- **XGBoost native support:** XGBoost has SHAP built into its tree structure — TreeExplainer is optimal

---

## Global Analysis: What Drives CLV Overall?

### Summary Plot (Beeswarm)

```python
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

- Each row = one feature
- Each dot = one sample from the test set
- **x-axis:** SHAP value (positive = increases CLV, negative = decreases)
- **colour:** Feature value (red = high, blue = low)
- **Insight:** Which features matter most AND in what direction

### Bar Plot (Mean Absolute SHAP)

```python
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names)
```

- Shows the **average magnitude** of each feature's SHAP value across all samples
- Classic global feature importance — comparable to RF/XGB built-in importance but theoretically grounded

---

## Local Analysis: Why This Specific Prediction?

### Waterfall Plot

```python
shap.waterfall_plot(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[idx],
    feature_names=feature_names
))
```

- Shows how each feature pushes a single prediction up or down from the base value
- **Use case:** Explaining why a specific customer has high or low predicted CLV
- Pick representative examples: a high-value prediction, a low-value prediction, and a borderline case

### Force Plot

```python
shap.force_plot(
    explainer.expected_value,
    shap_values[idx],
    X_test.iloc[idx],
    feature_names=feature_names
)
```

- Horizontal alternative to waterfall — shows features as arrows pushing from base value to prediction
- Good for embedding in reports

---

## Interaction Analysis: How Do Features Interact?

### Dependence Plot

```python
shap.dependence_plot("purchase_count_last_12m", shap_values, X_test, interaction_index="customer_segment_premium")
```

- **x-axis:** Feature value
- **y-axis:** SHAP value for that feature
- **colour:** Interaction feature (reveals conditional effects)
- **Use case:** Visualise the premium order bonus — does a higher `avg_order_value_chf` have a stronger effect for premium customers?

---

## Expected SHAP Findings

Based on the known CLV generating formula, we expect:

| Feature | Expected Direction | Expected Global Importance |
|---------|-------------------|--------------------------|
| `purchase_count_last_12m` | Positive | Very high (coefficient 42) |
| `customer_segment_premium` | Positive | Very high (+420 + bonus) |
| `avg_order_value_chf` | Positive | High |
| `customer_tenure_months` | Positive | High |
| `email_click_rate` | Positive | Medium-high |
| `return_rate` | Negative | High (coefficient -520) |
| `discount_usage_rate` | Negative | Medium-high |
| `support_tickets_last_12m` | Negative | Medium |
| `total_spend_proxy` | Positive | High (captures interaction) |
| `engagement_score` | Positive | Medium |

**Key validation:** If SHAP results from tree models show these patterns, it validates that our models learned the true data-generating process. Linear Regression may miss the interaction effects (premium bonus, loyal tenure bonus).

---

## SHAP Analysis in the Notebook

The SHAP analysis (Section 4.5) includes:

1. Global summary plots for all 3 models — side-by-side comparison
2. Local waterfall plots for 3 representative test samples (high CLV, low CLV, borderline)
3. Dependence plot for `purchase_count_last_12m` and `avg_order_value_chf` (XGBoost)
4. Discussion: Do SHAP explanations match the ground truth formula?

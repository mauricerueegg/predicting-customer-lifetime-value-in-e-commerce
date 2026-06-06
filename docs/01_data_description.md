# 01 – Data Description

## Source

The dataset is **synthetically generated** using NumPy with `random_state=808`. It simulates a realistic e-commerce customer base. The generation code lives in Section 0 of `notebook.ipynb` and must be run first.

- **File:** `01_data/topic_E2_customer_lifetime_value_raw.csv`
- **Records:** 1100 rows
- **Features:** 10 input features + 1 target variable

---

## Column Schema

| Column | Type | Range / Values | Distribution |
|--------|------|----------------|-------------|
| `customer_tenure_months` | int | 1–60 | Uniform |
| `purchase_count_last_12m` | int | 0–45 | Poisson(λ=8), clipped |
| `avg_order_value_chf` | float | 10–400 | Normal(85, 35), clipped |
| `return_rate` | float | 0–1 | Beta(2, 8) — right-skewed, low values |
| `website_visits_last_3m` | int | ≥0 | Poisson(λ=24) |
| `email_click_rate` | float | 0–1 | Beta(2.5, 4.5) |
| `customer_segment` | category | new/occasional/loyal/premium | p=[0.20, 0.36, 0.30, 0.14] |
| `preferred_device` | category | mobile/desktop/tablet | p=[0.58, 0.34, 0.08] |
| `support_tickets_last_12m` | int | 0–12 | Poisson(λ=1.8), clipped |
| `discount_usage_rate` | float | 0–1 | Beta(3, 4) |
| **`customer_lifetime_value_chf`** | **float** | **≥20** | **Target variable** |

---

## Target Variable: `customer_lifetime_value_chf`

The CLV is computed from a **structured linear formula with non-linear interaction effects and noise**.

### Full Generating Formula

```
CLV = base_value
    + customer_tenure_months * 9
    + purchase_count_last_12m * 42
    + avg_order_value_chf * 5.2
    + website_visits_last_3m * 1.4
    + email_click_rate * 220
    - return_rate * 520
    - support_tickets_last_12m * 24
    - discount_usage_rate * 180
    + segment_effect
    + device_effect
    + premium_order_bonus      (only if segment == "premium")
    + loyal_tenure_bonus       (only if segment == "loyal")
    + high_discount_penalty    (only if discount_usage_rate > 0.75)
    + noise ~ N(0, 120)

Final: clip(CLV, 20, ∞), round to 2 decimal places
```

### Segment Effects

| Segment | Effect (CHF) |
|---------|-------------|
| premium | +420 |
| loyal | +180 |
| occasional | +40 |
| new | -60 |

### Device Effects

| Device | Effect (CHF) |
|--------|-------------|
| desktop | +20 |
| tablet | -10 |
| mobile | 0 |

### Non-Linear / Interaction Effects

| Condition | Effect |
|-----------|--------|
| `segment == "premium"` | +`avg_order_value_chf * 1.4` (premium order bonus) |
| `segment == "loyal"` | +`customer_tenure_months * 2.5` (loyal tenure bonus) |
| `discount_usage_rate > 0.75` | -120 CHF (heavy discount penalty) |

### Noise

- Additive Gaussian noise: `N(0, σ=120)`
- This introduces realistic variability — no model will achieve perfect predictions

---

## Key Characteristics for Modelling

### 1. Dominant drivers (by coefficient magnitude)
Ranked by expected impact on CLV:
1. `purchase_count_last_12m` (×42) — highest linear weight
2. `customer_segment` — up to 480 CHF difference (premium vs. new)
3. `avg_order_value_chf` (×5.2, plus ×1.4 bonus for premium)
4. `customer_tenure_months` (×9, plus ×2.5 for loyal)
5. `email_click_rate` (×220)
6. `return_rate` (−×520) — strong negative effect

### 2. Skewed distributions
- `return_rate`: Beta(2,8) — most customers have low return rates
- `discount_usage_rate`: Beta(3,4) — moderate usage, slight left skew
- `email_click_rate`: Beta(2.5,4.5) — skewed toward lower values
- `customer_lifetime_value_chf`: likely right-skewed due to premium segment bonuses

### 3. Non-linearities
- The premium order bonus and loyal tenure bonus create **interaction effects** between segment and numeric features
- The high discount penalty at `discount_usage_rate > 0.75` is a **threshold effect**
- Linear models will struggle to capture these — motivating tree-based models

### 4. Categorical features
- `customer_segment` and `preferred_device` must be **encoded** before modelling
- One-hot encoding is appropriate (no ordinal relationship)

### 5. Noise level
- Standard deviation of noise = 120 CHF
- The theoretical minimum prediction error (irreducible noise) is around 120 CHF RMSE
- Any model with RMSE significantly above this is under-fitting

---

## Missing Values

The dataset is synthetically generated and contains **no missing values**. No imputation is required.

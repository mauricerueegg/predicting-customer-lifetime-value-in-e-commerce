import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo

    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    np.random.seed(808)

    n = 1100

    customer_tenure_months = np.random.randint(1, 61, n)
    purchase_count_last_12m = np.random.poisson(lam=8, size=n)
    purchase_count_last_12m = np.clip(purchase_count_last_12m, 0, 45)

    avg_order_value_chf = np.round(np.random.normal(85, 35, n), 2)
    avg_order_value_chf = np.clip(avg_order_value_chf, 10, 400)

    return_rate = np.round(np.random.beta(2, 8, n), 3)
    website_visits_last_3m = np.random.poisson(lam=24, size=n)
    email_click_rate = np.round(np.random.beta(2.5, 4.5, n), 3)

    customer_segment = np.random.choice(
        ["new", "occasional", "loyal", "premium"],
        size=n,
        p=[0.2, 0.36, 0.3, 0.14]
    )

    preferred_device = np.random.choice(
        ["mobile", "desktop", "tablet"],
        size=n,
        p=[0.58, 0.34, 0.08]
    )

    support_tickets_last_12m = np.random.poisson(lam=1.8, size=n)
    support_tickets_last_12m = np.clip(support_tickets_last_12m, 0, 12)

    discount_usage_rate = np.round(np.random.beta(3, 4, n), 3)

    # CLV generating mechanism
    base_value = 40
    tenure_effect = customer_tenure_months * 9
    purchase_effect = purchase_count_last_12m * 42
    order_value_effect = avg_order_value_chf * 5.2
    visit_effect = website_visits_last_3m * 1.4
    click_effect = email_click_rate * 220
    return_penalty = -return_rate * 520
    support_penalty = -support_tickets_last_12m * 24
    discount_penalty = -discount_usage_rate * 180

    segment_effect = np.where(
        customer_segment == "premium", 420,
        np.where(customer_segment == "loyal", 180,
                 np.where(customer_segment == "occasional", 40, -60))
    )

    device_effect = np.where(
        preferred_device == "desktop", 20,
        np.where(preferred_device == "tablet", -10, 0)
    )

    # Non-linear / interaction effects
    premium_order_bonus = np.where(customer_segment == "premium", avg_order_value_chf * 1.4, 0)
    loyal_tenure_bonus = np.where(customer_segment == "loyal", customer_tenure_months * 2.5, 0)
    high_discount_penalty = np.where(discount_usage_rate > 0.75, -120, 0)

    noise = np.random.normal(0, 120, n)

    customer_lifetime_value_chf = (
        base_value
        + tenure_effect
        + purchase_effect
        + order_value_effect
        + visit_effect
        + click_effect
        + return_penalty
        + support_penalty
        + discount_penalty
        + segment_effect
        + device_effect
        + premium_order_bonus
        + loyal_tenure_bonus
        + high_discount_penalty
        + noise
    )

    customer_lifetime_value_chf = np.round(np.clip(customer_lifetime_value_chf, 20, None), 2)

    df = pd.DataFrame({
        "customer_tenure_months": customer_tenure_months,
        "purchase_count_last_12m": purchase_count_last_12m,
        "avg_order_value_chf": avg_order_value_chf,
        "return_rate": return_rate,
        "website_visits_last_3m": website_visits_last_3m,
        "email_click_rate": email_click_rate,
        "customer_segment": customer_segment,
        "preferred_device": preferred_device,
        "support_tickets_last_12m": support_tickets_last_12m,
        "discount_usage_rate": discount_usage_rate,
        "customer_lifetime_value_chf": customer_lifetime_value_chf
    })

    df.to_csv("topic_E2_customer_lifetime_value_raw.csv", index=False)
    print("Saved: topic_E2_customer_lifetime_value_raw.csv")
    print(df.head())
    print(df.shape)
    return


if __name__ == "__main__":
    app.run()

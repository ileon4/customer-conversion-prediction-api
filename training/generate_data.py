from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_marketing_data(n_rows=25000, random_state=12345):
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 70, n_rows)
    annual_income = rng.normal(72000, 22000, n_rows).clip(25000, 180000).round(0)

    country = rng.choice(
        ["USA", "Canada", "UK", "Australia"],
        size=n_rows,
        p=[0.65, 0.15, 0.12, 0.08],
    )

    device_type = rng.choice(
        ["Mobile", "Desktop", "Tablet"],
        size=n_rows,
        p=[0.58, 0.34, 0.08],
    )

    traffic_source = rng.choice(
        ["Google", "Instagram", "Facebook", "TikTok", "Email", "Referral"],
        size=n_rows,
        p=[0.30, 0.20, 0.16, 0.12, 0.14, 0.08],
    )

    campaign_type = rng.choice(
        ["Discount", "Product Launch", "Newsletter", "Retargeting", "Brand Awareness"],
        size=n_rows,
        p=[0.24, 0.18, 0.22, 0.20, 0.16],
    )

    # More realistic behavioral dependencies
    ad_spend = rng.gamma(2.2, 6, n_rows).clip(0.5, 80)

    base_pages = rng.poisson(4, n_rows)
    pages_visited = (
        base_pages
        + (ad_spend / 15)
        + np.where(traffic_source == "Referral", 2, 0)
        + np.where(traffic_source == "Email", 1, 0)
        + np.where(campaign_type == "Retargeting", 2, 0)
        + rng.normal(0, 1.5, n_rows)
    )
    pages_visited = np.round(pages_visited).clip(1, 35).astype(int)

    session_duration = (
        pages_visited * rng.normal(55, 12, n_rows)
        + np.where(device_type == "Desktop", 90, 0)
        + np.where(device_type == "Mobile", -35, 0)
        + rng.normal(0, 120, n_rows)
    )
    session_duration = np.round(session_duration).clip(10, 2200)

    email_opens = (
        rng.poisson(1.5, n_rows)
        + np.where(traffic_source == "Email", rng.poisson(2, n_rows), 0)
        + np.where(campaign_type == "Newsletter", rng.poisson(1.5, n_rows), 0)
        + np.where(campaign_type == "Retargeting", rng.poisson(1, n_rows), 0)
    )
    email_opens = email_opens.clip(0, 18)

    click_probability = np.clip(
        0.18
        + 0.035 * email_opens
        + np.where(campaign_type == "Discount", 0.08, 0)
        + np.where(campaign_type == "Retargeting", 0.06, 0),
        0,
        0.85,
    )
    email_clicks = rng.binomial(email_opens, click_probability)

    previous_purchases = (
        rng.poisson(0.5, n_rows)
        + np.where(age > 35, rng.binomial(1, 0.25, n_rows), 0)
        + np.where(traffic_source == "Referral", rng.binomial(1, 0.20, n_rows), 0)
    )
    previous_purchases = previous_purchases.clip(0, 10)

    days_since_last_visit = (
        rng.exponential(25, n_rows)
        - previous_purchases * 4
        - email_clicks * 2
        + np.where(campaign_type == "Retargeting", -6, 0)
    )
    days_since_last_visit = np.round(days_since_last_visit).clip(0, 180)

    discount_offered = rng.choice(
        [0, 5, 10, 15, 20, 25],
        size=n_rows,
        p=[0.30, 0.12, 0.20, 0.16, 0.14, 0.08],
    )

    # Conversion signal
    score = (
        -3.8
        + 0.07 * pages_visited
        + 0.0035 * session_duration
        + 0.18 * email_opens
        + 0.50 * email_clicks
        + 0.55 * previous_purchases
        - 0.018 * days_since_last_visit
        + 0.035 * discount_offered
        + 0.010 * ad_spend
        + np.where(traffic_source == "Referral", 0.65, 0)
        + np.where(traffic_source == "Email", 0.35, 0)
        + np.where(traffic_source == "TikTok", -0.10, 0)
        + np.where(campaign_type == "Retargeting", 0.55, 0)
        + np.where(campaign_type == "Discount", 0.35, 0)
        + np.where(device_type == "Desktop", 0.18, 0)
        + rng.normal(0, 0.9, n_rows)
    )

    conversion_probability = sigmoid(score)
    converted = rng.binomial(1, conversion_probability)

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "country": country,
            "device_type": device_type,
            "traffic_source": traffic_source,
            "campaign_type": campaign_type,
            "pages_visited": pages_visited,
            "session_duration": session_duration,
            "email_opens": email_opens,
            "email_clicks": email_clicks,
            "previous_purchases": previous_purchases,
            "days_since_last_visit": days_since_last_visit,
            "discount_offered": discount_offered,
            "ad_spend": ad_spend.round(2),
            "converted": converted,
        }
    )


if __name__ == "__main__":
    output_path = Path("data/marketing_conversion_data.csv")
    output_path.parent.mkdir(exist_ok=True)

    df = generate_marketing_data()
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Conversion rate: {df['converted'].mean():.2%}")
    print(df.head())
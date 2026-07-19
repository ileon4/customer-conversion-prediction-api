from pathlib import Path

import joblib
import pandas as pd

from app.schemas import Customer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "marketing_conversion_model.joblib"

model = joblib.load(MODEL_PATH)


def predict_conversion(customer: Customer) -> dict:
    customer_dict = customer.model_dump()
    input_df = pd.DataFrame([customer_dict]) # extra brackets to create a DataFrame with one row

    probabilities = model.predict_proba(input_df)
    conversion_probability = float(probabilities[0][1]) # probability of the positive class (conversion)

    raw_prediction = model.predict(input_df)[0]
    prediction = int(raw_prediction)

    if conversion_probability >= 0.80:
        prediction_label = "Highly Likely to Convert"
        recommendation = (
            "This customer appears to be a strong prospect. "
            "Recommend immediate sales outreach."
        )

    elif conversion_probability >= 0.60:
        prediction_label = "Moderately Likely to Convert"
        recommendation = (
            "Continue targeted engagement and follow up soon."
        )

    elif conversion_probability >= 0.40:
        prediction_label = "Potential Prospect"
        recommendation = (
            "Continue nurturing through email campaigns or personalized offers."
        )

    else:
        prediction_label = "Unlikely to Convert"
        recommendation = (
            "Focus marketing resources elsewhere or re-engage later."
        )

    return {
        "prediction": prediction_label,
        "conversion_probability": round(conversion_probability, 4),
        "recommendation": recommendation,
    }
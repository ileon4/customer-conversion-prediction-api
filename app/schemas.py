# This file gives us: automatic validation, automatic documentation, automatic Swagger UI forms, and automatic JSON parsing

from pydantic import BaseModel, ConfigDict

class Customer(BaseModel):
    age: int
    annual_income: float
    country: str
    device_type: str
    traffic_source: str
    campaign_type: str
    pages_visited: int
    session_duration: float
    email_opens: int
    email_clicks: int
    previous_purchases: int
    days_since_last_visit: int
    discount_offered: float
    ad_spend: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 35,
                "annual_income": 85000,
                "country": "USA",
                "device_type": "Desktop",
                "traffic_source": "Google",
                "campaign_type": "Retargeting",
                "pages_visited": 12,
                "session_duration": 950,
                "email_opens": 5,
                "email_clicks": 3,
                "previous_purchases": 4,
                "days_since_last_visit": 7,
                "discount_offered": 15,
                "ad_spend": 22
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: str
    conversion_probability: float
    recommendation: str
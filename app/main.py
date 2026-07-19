from fastapi import FastAPI
from app.predictor import predict_conversion
from app.schemas import Customer, PredictionResponse

app = FastAPI() # creates web application


@app.get("/") # defines a route for the root URL ("/") and specifies that it will respond to GET requests
def home():
    return {
        "message": "Marketing Conversion API is running!"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer):
    return predict_conversion(customer)
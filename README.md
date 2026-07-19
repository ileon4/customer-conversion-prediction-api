# Marketing Conversion Prediction API

---------------------------------------

🚀 Live Demo

🔗 API URL - https://marketing-conversion-api.onrender.com/

🔗 Swagger Documentation - https://marketing-conversion-api.onrender.com/docs

---------------------------------------

## Overview

A developed and deployed containerized machine learning REST API using FastAPI and Docker on Render, enabling real-time customer conversion predictions with a production-ready inference pipeline.

## Business Problem

A business is noticing more customers are canceling their subscriptions. They need a way to identify if a customer is likely to churn before they do so that targeted marketing strategies can be executed to retain these customers.

## Features

"age"
"annual_income"
"country"
"device_type"
"traffic_source"
"campaign_type"
"pages_visited"
"session_duration"
"email_opens"
"email_clicks"
"previous_purchases"
"days_since_last_visit"
"discount_offered"
"ad_spend"

## Tech Stack

Python
FastAPI
Docker
Render

## Architecture Diagram

Marketing Data
        │
        ▼
Training Pipeline
        │
        ▼
Random Forest Model
        │
        ▼
    joblib
        │
        ▼
    FastAPI
        │
        ▼
    Docker
        │
        ▼
    Render
        │
        ▼
    Public API

## Machine Learning Pipeline

Raw Customer Data
        │
        ▼
Train / Test Split
        │
        ▼
ColumnTransformer
   ├── "passthrough" (numeric features)
   └── OneHotEncoder (categorical features)
        │
        ▼
Random Forest Classifier
        │
        ▼
scikit-learn Pipeline
        │
        ▼
Serialized with joblib
        │
        ▼
Loaded by FastAPI for real-time inference

## Project Structure

marketing-conversion-api/
│
├── app/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── training/
│   ├── train.py
│   └── generate_data.py
│
├── data/
├── models/
├── notebooks/
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── uv.lock

## Installation

### Prerequisites

- Python 3.13+
- uv
- Docker Desktop (optional for containerized deployment)
- Git

### Clone the repository

```bash
git clone https://github.com/ileon4/marketing-conversion-api.git
cd marketing-conversion-api
```

### Install dependencies

```bash
uv sync
```

### Run the API locally

```bash
uv run uvicorn app.main:app --reload
```

Open:

http://127.0.0.1:8000/docs

## Docker Usage

Build the Docker image:

```bash
docker build -t marketing-conversion-api .
```

Run the container:

```bash
docker run \
--name marketing-conversion-container \
-p 8000:8000 \
marketing-conversion-api
```

Once the container is running, open:

http://127.0.0.1:8000/docs

To stop the container:

```bash
docker stop marketing-conversion-container
```

## API Documentation

### Live API

https://marketing-conversion-api.onrender.com

### Interactive Documentation

https://marketing-conversion-api.onrender.com/docs

### Health Check

**GET /**

Returns:

```json
{
    "message": "Marketing Conversion API is running!"
}
```

### Predict Customer Conversion

**POST /predict**

Request:

```json
{
  "age": 42,
  "annual_income": 95000,
  "country": "United States",
  ...
}
```

Response:

```json
{
  "prediction": "Highly Likely to Convert",
  "conversion_probability": 0.9557,
  "recommendation": "This customer appears to be a strong prospect. Recommend immediate sales outreach."
}
```

## Model Performance

found in reports folder

accuracy: 0.7104,
precision: 0.6180785612872692,
recall: 0.6707755521314843,
f1_score: 0.6433497536945813,
roc_auc: 0.7820059954331885,
confusion_matrix:
[2246, 807]
[641, 1306]

Note: Performance metrics are reported on a held-out test set generated from the synthetic marketing conversion dataset. The primary objective of this project is to demonstrate an end-to-end machine learning deployment workflow rather than optimize a production model.
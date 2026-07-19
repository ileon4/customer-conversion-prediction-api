from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "marketing_conversion_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "marketing_conversion_model.joblib"


df = pd.read_csv(DATA_PATH)

target = "converted"

numeric_features = [
    "age",
    "annual_income",
    "pages_visited",
    "session_duration",
    "email_opens",
    "email_clicks",
    "previous_purchases",
    "days_since_last_visit",
    "discount_offered",
    "ad_spend",
]

categorical_features = [
    "country",
    "device_type",
    "traffic_source",
    "campaign_type",
]

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=12345,
    stratify=y,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=12345,
                class_weight="balanced",
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
predicted_proba = pipeline.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, predictions))

print("ROC-AUC:")
print(roc_auc_score(y_test, predicted_proba))

joblib.dump(pipeline, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
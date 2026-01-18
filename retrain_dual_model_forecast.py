#!/usr/bin/env python3
"""
Retrain dual dengue forecasting models (2 months ahead prediction).
Trains regression model for case count and classifier for outbreak probability.
Saves models to /models folder.
"""

import pandas as pd
import numpy as np
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier

# ---- Load data ----
DATA_PATH = "final_monthly_df_forecast_ready.csv"
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded forecasting dataset: {df.shape}")

# ---- Define features and targets ----
target_reg = "Future_dengue"
features = [
    "Temperature", "Precipitation", "Wind speed", "humidity",
    "urban_density", "month", "year", "season", "state"
]

X = df[features]
y_cases = df[target_reg]

# ---- Create outbreak classification label ----
threshold = y_cases.median()
y_outbreak = (y_cases > threshold).astype(int)
print(f"📊 Outbreak threshold = {threshold:.2f}")

# ---- Train-test split ----
X_train, X_test, y_train_cases, y_test_cases, y_train_out, y_test_out = train_test_split(
    X, y_cases, y_outbreak, test_size=0.2, random_state=42
)

# ---- Preprocessing ----
num_cols = ["Temperature", "Precipitation", "Wind speed", "humidity", "urban_density", "month", "year"]
cat_cols = ["state", "season"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ---- Regression Pipeline ----
regressor = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    ))
])

print("🚀 Training regression model...")
regressor.fit(X_train, y_train_cases)
print("✅ Regression model trained.")

# ---- Classification Pipeline ----
classifier = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ))
])

print("🚀 Training outbreak classifier...")
classifier.fit(X_train, y_train_out)
print("✅ Classifier model trained.")

# ---- Save models ----
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(regressor, "models/xgb_cases_pipeline.pkl")
joblib.dump(classifier, "models/xgb_outbreak_pipeline.pkl")

print("\n💾 All models saved successfully in /models/")
print("✅ Regression model: models/xgb_cases_pipeline.pkl")
print("✅ Classifier model: models/xgb_outbreak_pipeline.pkl")

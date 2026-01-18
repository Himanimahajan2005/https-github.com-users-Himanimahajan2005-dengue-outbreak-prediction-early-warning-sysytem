#!/usr/bin/env python3
from flask import Flask, request, jsonify
import joblib, pandas as pd, numpy as np

app = Flask(__name__)

# Load pre-trained artifacts
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/xgb_outbreak_model.pkl")

# Expected features (same order used in training)
FEATURE_COLS = [
    'humidity', 'wind_speed', 'temperature', 'precipitation', 'urban_density',
    'lag_dengue_cases_1', 'lag_dengue_cases_2', 'lag_dengue_cases_3',
    'lag_dengue_cases_4', 'lag_dengue_cases_5', 'lag_dengue_cases_6',
    'lag_dengue_cases_12', 'lag_temperature_1', 'lag_temperature_2',
    'lag_temperature_3', 'lag_temperature_4', 'lag_humidity_1',
    'lag_humidity_2', 'lag_humidity_3', 'lag_humidity_4', 'lag_precipitation_1',
    'lag_precipitation_2', 'lag_precipitation_3', 'lag_precipitation_4',
    'lag_wind_speed_1', 'lag_wind_speed_2', 'lag_wind_speed_3',
    'lag_wind_speed_4', 'roll_mean_dengue_cases_3', 'lag_tempxhumid_3',
    'month_sin', 'month_cos', 'avg_neighbor_dengue', 'season', 'state'
]

def prepare_row(payload):
    """Prepare single-row dataframe from incoming JSON payload."""
    row = {c: payload.get(c, None) for c in FEATURE_COLS}
    df = pd.DataFrame([row])
    # Numeric conversions
    for col in ['month_sin', 'month_cos', 'avg_neighbor_dengue', 'lag_tempxhumid_3']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['season'] = df['season'].fillna('unknown')
    df['state'] = df['state'].fillna('unknown')
    return df

# ✅ Root route for browser testing
@app.route('/')
def home():
    return "✅ Dengue Prediction API is running! Use POST /predict to get predictions."

# 🔮 Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    payload = request.json or request.get_json(force=True)
    X_row = prepare_row(payload)
    X_trans = preprocessor.transform(X_row)
    prob = float(model.predict_proba(X_trans)[0, 1])
    return jsonify({"outbreak_probability": prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

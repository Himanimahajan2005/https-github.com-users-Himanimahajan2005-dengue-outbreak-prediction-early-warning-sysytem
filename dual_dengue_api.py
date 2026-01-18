#!/usr/bin/env python3
# dual_dengue_api_future.py
from flask import Flask, request, jsonify, render_template
import os, joblib, pandas as pd, numpy as np, math, json, logging
from feature_autofill import autofill_missing_features
from nasa_weather_fetcher import get_future_weather  # ✅ NASA POWER integration

app = Flask(__name__, template_folder="templates")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dual_dengue_future")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")

# ---- Load models ----
cases_pipeline = joblib.load(os.path.join(MODELS_DIR, "xgb_cases_pipeline.pkl"))
outbreak_pipeline = joblib.load("models/xgb_outbreak_pipeline_improved.pkl")
  

# ---- Helpers ----
def month_sin_cos(month):
    s = math.sin(2 * math.pi * month / 12)
    c = math.cos(2 * math.pi * month / 12)
    return s, c

def prepare_input(payload):
    df = pd.DataFrame([payload])
    if "month_sin" not in df:
        s, c = month_sin_cos(int(payload["month"]))
        df["month_sin"], df["month_cos"] = s, c
    for col in df.columns:
        if col not in ["state", "season"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def compute_predicted_month(year, month, horizon=2):
    m = month + horizon
    y = year + (m - 1) // 12
    mm = ((m - 1) % 12) + 1
    return f"{y}-{mm:02d}"

def season_risk_multiplier(season):
    """Adjust probability based on known dengue seasonality."""
    s = season.lower().strip()
    if s in ["monsoon"]:
        return 1.6
    elif s in ["post-monsoon", "postmonsoon"]:
        return 1.4
    elif s in ["pre-monsoon", "premonsoon"]:
        return 1.2
    elif s in ["winter"]:
        return 0.8
    return 1.0

# ---- Routes ----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Empty payload"}), 400

        # Extract key info
        state = payload.get("state", "Maharashtra")
        year = int(payload.get("year", 2025))
        month = int(payload.get("month", 6))
        season = payload.get("season", "Monsoon")

        # 🌦 STEP 1 — Ensure NASA weather is always fetched if missing
        required_weather = ["temperature", "humidity", "precipitation", "wind_speed"]
        missing_weather = [k for k in required_weather if k not in payload or payload[k] in [None, "", 0]]

        if missing_weather:
            try:
                log.info(f"🌤 Fetching projected weather for {state} ({year}-{month:02d}) from NASA POWER")
                weather = get_future_weather(state, year, month)
                for key in required_weather:
                    payload[key] = weather.get(key, 0)
                log.info(f"✅ Injected NASA POWER weather: {weather}")
            except Exception as e:
                log.warning(f"⚠️ NASA POWER fetch failed ({e}), using default average values")
                payload.update({
                    "temperature": 30,
                    "humidity": 70,
                    "precipitation": 120,
                    "wind_speed": 3.5
                })

        # 🌡 STEP 2 — Auto-fill other lag and derived features
        payload = autofill_missing_features(payload)

        # 🧮 STEP 3 — Prepare DataFrame for model input
        X = prepare_input(payload)
        missing_after = [col for col in ["temperature", "precipitation", "humidity", "wind_speed"] if col not in X.columns]
        if missing_after:
            log.error(f"❌ Still missing weather columns in X: {missing_after}")
            return jsonify({"error": f"Missing required columns even after autofill: {missing_after}"}), 500

        # 🔮 STEP 4 — Predict dengue cases and outbreak probability
        pred_cases = float(cases_pipeline.predict(X)[0])
        prob = float(outbreak_pipeline.predict_proba(X)[0, 1])

        # 🎯 Confidence and forecast month
        confidence = round(0.7 * prob + 0.3 * np.clip(1 - abs(pred_cases - np.mean([100, 1000])) / 1000, 0, 1), 3)
        forecast_month = compute_predicted_month(year, month)

        response = {
            "predicted_cases": round(pred_cases, 2),
            "outbreak_probability": round(prob, 3),
            "confidence_score": confidence,
            "predicted_month": forecast_month,
            "used_weather": {
                "temperature": payload["temperature"],
                "humidity": payload["humidity"],
                "precipitation": payload["precipitation"],
                "wind_speed": payload["wind_speed"]
            }
        }

        return jsonify(response)

    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

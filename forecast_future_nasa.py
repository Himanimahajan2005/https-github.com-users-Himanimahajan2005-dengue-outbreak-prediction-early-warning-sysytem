#!/usr/bin/env python3
# forecast_future_nasa.py
"""
Forecast future dengue cases (2024–2026) using trained models and NASA POWER weather data.
"""

import requests
import joblib
import pandas as pd
import numpy as np
from math import sin, cos, pi
from datetime import datetime

# ===============================
# 1️⃣  Load your trained models
# ===============================
cases_model = joblib.load("models/xgb_cases_pipeline.pkl")
clf_model = joblib.load("models/xgb_outbreak_pipeline.pkl")

# ===============================
# 2️⃣  NASA POWER API setup
# ===============================
NASA_BASE = "https://power.larc.nasa.gov/api/temporal/monthly/point"
PARAMS = ["T2M", "PRECTOTCORR", "RH2M", "WS2M"]  # temperature, precipitation, humidity, wind speed

# Coordinates for your 5 target states (approx)
STATE_COORDS = {
    "Maharashtra": (19.7515, 75.7139),
    "Delhi": (28.6139, 77.2090),
    "Kerala": (10.8505, 76.2711),
    "Tamil Nadu": (11.1271, 78.6569),
    "West Bengal": (22.9868, 87.8550),
}

# ===============================
# 3️⃣  Helper functions
# ===============================
def fetch_nasa_weather(lat, lon, start="2024", end="2026"):
    """Fetch monthly NASA POWER data for given location and years."""
    url = f"{NASA_BASE}?parameters={','.join(PARAMS)}&community=RE&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
    print(f"Fetching NASA weather for {lat},{lon} ({start}-{end}) ...")
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    records = data["properties"]["parameter"]
    df = pd.DataFrame({
        "year_month": list(records["T2M"].keys()),
        "temperature": list(records["T2M"].values()),
        "precipitation": list(records["PRECTOTCORR"].values()),
        "humidity": list(records["RH2M"].values()),
        "wind_speed": list(records["WS2M"].values()),
    })
    df["year"] = df["year_month"].str[:4].astype(int)
    df["month"] = df["year_month"].str[4:].astype(int)
    return df[["year", "month", "temperature", "precipitation", "humidity", "wind_speed"]]

def month_encoding(month):
    return sin(2 * pi * (month / 12.0)), cos(2 * pi * (month / 12.0))

# ===============================
# 4️⃣  Forecast loop
# ===============================
def forecast_state(state, lat, lon, start_year=2024, end_year=2026, last_cases=500):
    """Forecast dengue for a state given NASA data and last known cases."""
    df_weather = fetch_nasa_weather(lat, lon, str(start_year), str(end_year))
    results = []

    for _, row in df_weather.iterrows():
        year, month = int(row["year"]), int(row["month"])
        temp, prec, hum, wind = row["temperature"], row["precipitation"], row["humidity"], row["wind_speed"]
        m_sin, m_cos = month_encoding(month)

        X = pd.DataFrame([{
            "temperature": temp,
            "precipitation": prec,
            "wind_speed": wind,
            "humidity": hum,
            "urban_density": 8000,  # you can set actual density if available
            "state": state,
            "season": "Monsoon" if 6 <= month <= 9 else "Post-monsoon" if 10 <= month <= 11 else "Pre-monsoon" if 3 <= month <= 5 else "Winter",
            "month": month,
            "year": year,
            "month_sin": m_sin,
            "month_cos": m_cos,
            "lag_dengue_cases_1": last_cases,
            "lag_temperature_1": temp,
            "lag_precipitation_1": prec,
            "lag_humidity_1": hum,
            "lag_wind_speed_1": wind,
            "roll_mean_dengue_cases_3": last_cases,
            "lag_tempxhumid_3": temp * hum,
            "avg_neighbor_dengue": 600
        }])

        pred_cases = float(cases_model.predict(X)[0])
        outbreak_prob = float(clf_model.predict_proba(X)[0, 1])
        results.append({
            "state": state,
            "year": year,
            "month": month,
            "predicted_cases": round(pred_cases, 2),
            "outbreak_probability": round(outbreak_prob * 100, 2)
        })

        last_cases = pred_cases  # feedback for autoregression

    return pd.DataFrame(results)

# ===============================
# 5️⃣  Run for all states
# ===============================
all_forecasts = []
for state, (lat, lon) in STATE_COORDS.items():
    print(f"\n--- Forecasting for {state} ---")
    df_state = forecast_state(state, lat, lon, last_cases=400)
    all_forecasts.append(df_state)

final_df = pd.concat(all_forecasts)
final_df.to_csv("forecast_nasa_2024_2026.csv", index=False)

print("\n Forecast completed and saved as 'forecast_nasa_2024_2026.csv'")
print(final_df.head(15))

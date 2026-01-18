# feature_autofill.py
import numpy as np
import math

def month_sin_cos(month):
    s = math.sin(2 * math.pi * month / 12)
    c = math.cos(2 * math.pi * month / 12)
    return s, c

def autofill_missing_features(payload):
    """
    Ensures all required columns exist for the model, filling or computing defaults.
    """
    filled = payload.copy()

    # === Ensure core numeric features ===
    defaults = {
        "temperature": 30.0,
        "humidity": 70.0,
        "precipitation": 100.0,
        "wind_speed": 3.0,
        "urban_density": 8000,
        "lag_dengue_cases_1": 200,
    }
    for k, v in defaults.items():
        filled[k] = float(filled.get(k, v) or v)

    # === Derived lag-based / contextual features ===
    lag = filled["lag_dengue_cases_1"]
    # Rolling 3-month average (approximation)
    filled["rolling_3mo_avg"] = lag * 0.9
    # Month-over-month growth (dummy estimate if not historical)
    filled["case_growth_rate"] = np.clip((lag - 150) / 150, -1, 2)

    # === Month encoding ===
    month = int(filled.get("month", 6))
    s, c = month_sin_cos(month)
    filled["month_sin"] = s
    filled["month_cos"] = c

    # === Fallbacks for season/state ===
    filled["season"] = filled.get("season", "Monsoon")
    filled["state"] = filled.get("state", "Maharashtra")

    # === Fill other missing lag placeholders ===
    for lag_i in [2, 3, 4, 5, 6, 12]:
        filled[f"lag_dengue_cases_{lag_i}"] = lag * (0.95 ** lag_i)
    filled["lag_temperature_1"] = filled["temperature"]
    filled["lag_humidity_1"] = filled["humidity"]
    filled["lag_precipitation_1"] = filled["precipitation"]
    filled["lag_wind_speed_1"] = filled["wind_speed"]
    filled["lag_tempxhumid_3"] = filled["temperature"] * filled["humidity"] / 100
    filled["avg_neighbor_dengue"] = filled["lag_dengue_cases_1"] * 0.85
    filled["mean_state_dengue_date"] = filled["lag_dengue_cases_1"] * 0.92
    filled["roll_mean_dengue_cases_3"] = filled["lag_dengue_cases_1"] * 0.93

    return filled

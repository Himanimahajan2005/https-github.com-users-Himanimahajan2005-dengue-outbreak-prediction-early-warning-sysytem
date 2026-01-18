# test_predict_dual_keys.py
import requests, json

url = "http://127.0.0.1:5000/predict"
payload = {
  # include both common variants (lowercase_underscore and original training names)
  "temperature": 31, "Temperature": 31,
  "precipitation": 90, "Precipitation": 90,
  "wind_speed": 4, "Wind speed": 4,
  "humidity": 74, "urban_density": 10400,
  "state": "Maharashtra", "season": "Post-monsoon",
  "month": 10, "year": 2023,
  "Future_dengue": 0, "future_dengue": 0,
  "lag_dengue_cases_1": 400, "lag_dengue_cases_2": 320, "lag_dengue_cases_3": 200,
  "lag_temperature_1": 30, "lag_precipitation_1": 120, "lag_humidity_1": 70, "lag_wind_speed_1": 4
}

r = requests.post(url, json=payload, timeout=10)
print("Status:", r.status_code)
try:
    print(json.dumps(r.json(), indent=2))
except Exception:
    print(r.text)

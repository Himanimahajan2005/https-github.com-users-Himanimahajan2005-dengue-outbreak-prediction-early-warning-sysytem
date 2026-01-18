"""
nasa_weather_fetcher.py
Fetches projected or recent monthly weather data from NASA POWER API.
If future data (e.g., 2025–2026) is unavailable, it falls back to the previous year
or to local climatology averages.
"""

import requests
import logging
import pandas as pd
from datetime import datetime

log = logging.getLogger("nasa_weather_fetcher")

# --- State coordinates (approximate central locations)
STATE_COORDS = {
    "Maharashtra": (19.7515, 75.7139),
    "Delhi": (28.6139, 77.2090),
    "Kerala": (10.8505, 76.2711),
    "Tamil Nadu": (11.1271, 78.6569),
    "West Bengal": (22.9868, 87.8550),
}

# --- Climatology fallback (mean values per state/month from past data)
CLIMATOLOGY = {
    "Maharashtra": {1:(24.5, 5.0, 2.3, 55), 3:(30.2,12.0,3.5,48),7:(27.0,300.0,2.6,85),10:(28.5,80.0,3.1,70)},
    "Delhi":        {1:(15.0,20.0,2.0,60), 3:(26.0,15.0,3.5,42),7:(31.0,180.0,2.8,82),10:(29.0,20.0,2.2,65)},
    "Kerala":       {1:(27.0,25.0,1.8,78), 3:(29.0,40.0,2.2,72),7:(28.0,300.0,2.5,88),10:(27.5,150.0,2.3,80)},
    "Tamil Nadu":   {1:(26.0,20.0,2.3,70), 3:(30.0,25.0,3.0,60),7:(31.0,110.0,2.5,75),10:(28.0,220.0,2.7,80)},
    "West Bengal":  {1:(20.0,10.0,2.1,68), 3:(27.0,25.0,2.8,65),7:(30.0,320.0,3.0,85),10:(29.0,150.0,2.5,78)},
}

def get_climatology(state: str, month: int):
    """Return climatology (T, P, W, H) tuple for given state/month."""
    state = state.title()
    if state in CLIMATOLOGY and month in CLIMATOLOGY[state]:
        t, p, w, h = CLIMATOLOGY[state][month]
        return {"Temperature": t, "Precipitation": p, "Wind speed": w, "Humidity": h}
    # default fallback
    return {"Temperature": 30.0, "Precipitation": 100.0, "Wind speed": 2.5, "Humidity": 70.0}

def fetch_from_nasa(lat: float, lon: float, year: int, month: int):
    """Query NASA POWER API for given coordinates and month."""
    ym = f"{year}{month:02d}"
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point?"
        f"parameters=T2M,PRECTOTCORR,WS2M,RH2M&community=AG&longitude={lon}"
        f"&latitude={lat}&start={ym}&end={ym}&format=JSON"
    )
    resp = requests.get(url, timeout=30)
    if resp.status_code == 422:
        return None  # Data unavailable
    resp.raise_for_status()
    data = resp.json()
    monthly = data.get("properties", {}).get("parameter", {})
    if not monthly:
        return None
    return {
        "Temperature": list(monthly.get("T2M", {}).values())[0],
        "Precipitation": list(monthly.get("PRECTOTCORR", {}).values())[0],
        "Wind speed": list(monthly.get("WS2M", {}).values())[0],
        "Humidity": list(monthly.get("RH2M", {}).values())[0],
    }

def get_future_weather(state: str, year: int, month: int):
    """Get weather for given state and future month, using fallback if NASA data unavailable."""
    state = state.title()
    if state not in STATE_COORDS:
        raise ValueError(f"No coordinates for state: {state}")
    lat, lon = STATE_COORDS[state]

    log.info(f"Fetching projected weather for {state} ({lat},{lon}) {year}-{month:02d}")
    data = fetch_from_nasa(lat, lon, year, month)

    if data is None:
        log.warning(f"No NASA data for {state}, {year}-{month:02d}. Trying previous year...")
        # Try previous year (one-year fallback)
        data = fetch_from_nasa(lat, lon, year - 1, month)

    if data is None:
        log.warning(f"Falling back to climatology for {state}, month {month}")
        data = get_climatology(state, month)

    return data

if __name__ == "__main__":
    # Quick test
    print(get_future_weather("Maharashtra", 2025, 3))

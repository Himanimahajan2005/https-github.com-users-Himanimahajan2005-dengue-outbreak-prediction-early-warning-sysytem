#!/usr/bin/env python3
"""
Shift dengue cases 2 months backward so that features of month t
predict dengue cases of month t+2.
Generates: final_monthly_df_forecast_ready.csv
"""

import pandas as pd
import os

# ---- Load original dataset ----
DATA_PATH = "final_monthly_df_adjusted_for_training.csv"
df = pd.read_csv(DATA_PATH)

print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ---- Sort properly ----
df = df.sort_values(["state", "year", "month"]).reset_index(drop=True)

# ---- Create shifted dengue target ----
df["Future_dengue"] = df.groupby("state")["dengue_cases"].shift(-2)

# ---- Drop rows where Future_dengue is NaN ----
df = df.dropna(subset=["Future_dengue"]).reset_index(drop=True)

# ---- Save final forecasting dataset ----
OUTPUT_PATH = "final_monthly_df_forecast_ready.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"🎯 Successfully created 2-month shifted dataset → {OUTPUT_PATH}")
print(f"✅ Final shape: {df.shape}")

#!/usr/bin/env python3
"""
Generates expected dengue case ranges (mean ± std) per state and season
for explainable predictions in the dengue dashboard.
"""

import pandas as pd
import os

# ✅ Load your combined dataset (adjust filename if needed)
DATA_PATH = "final_monthly_df_adjusted_for_training.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

print("🔍 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ✅ Check that dengue_cases column exists
if "dengue_cases" not in df.columns:
    raise KeyError("❌ 'dengue_cases' column not found in dataset!")

# ✅ Compute seasonal mean and standard deviation for each state
print("📊 Computing mean and standard deviation by (state, season)...")
summary = df.groupby(["state", "season"])["dengue_cases"].agg(["mean", "std"]).reset_index()

# ✅ Add lower and upper range bounds
summary["lower_bound"] = (summary["mean"] - summary["std"]).clip(lower=0)
summary["upper_bound"] = summary["mean"] + summary["std"]

# ✅ Save results
os.makedirs("models", exist_ok=True)
OUT_PATH = os.path.join("models", "expected_ranges.csv")
summary.to_csv(OUT_PATH, index=False)

print(f"✅ Expected dengue ranges saved at: {OUT_PATH}")
print(summary.head())

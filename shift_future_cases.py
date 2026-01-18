#!/usr/bin/env python3
"""
Fix misaligned Future_dengue column and create a correct 2-month-ahead target
for each state. Saves final_training_engineered_shifted.csv.
"""

import pandas as pd
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "final_training_engineered.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "final_training_engineered_shifted.csv")

print("🔍 Loading dataset...")
df = pd.read_csv(INPUT_PATH)
print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

# ---- Drop wrong column if exists ----
if "Future_dengue" in df.columns:
    df = df.drop(columns=["Future_dengue"])
    print("⚙️ Dropped existing 'Future_dengue' column")

# ---- Create new 2-month-ahead Future_dengue ----
df["Future_dengue"] = df.groupby("state")["dengue_cases"].shift(-2)

# ---- Drop rows without valid shifted target ----
before = len(df)
df = df.dropna(subset=["Future_dengue"])
after = len(df)
print(f"🧹 Dropped {before - after} rows with missing future dengue values")

# ---- Save corrected file ----
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved corrected dataset as {OUTPUT_PATH}")
print("📊 Columns:", list(df.columns))
print(df.head(5))

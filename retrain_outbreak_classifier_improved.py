#!/usr/bin/env python3
# retrain_outbreak_classifier_improved.py (fixed for SMOTE + categorical encoding)

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ---------------- CONFIG ----------------
DATA_PATH = "final_monthly_df_shifted.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------- FEATURE ENGINEERING ----------------
df['case_growth_rate'] = (df['lag_dengue_cases_1'] - df['lag_dengue_cases_2']) / (df['lag_dengue_cases_2'] + 1)
df['rolling_3mo_avg'] = df[['lag_dengue_cases_1', 'lag_dengue_cases_2', 'lag_dengue_cases_3']].mean(axis=1)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

target = "outbreak_flag"
num_features = [
    'temperature','precipitation','wind_speed','humidity','urban_density',
    'lag_dengue_cases_1','lag_dengue_cases_2','lag_dengue_cases_3',
    'lag_dengue_cases_4','lag_dengue_cases_5','lag_dengue_cases_6',
    'lag_dengue_cases_12','lag_temperature_1','lag_precipitation_1',
    'lag_humidity_1','lag_wind_speed_1','roll_mean_dengue_cases_3',
    'lag_tempxhumid_3','month_sin','month_cos','mean_state_dengue_date',
    'avg_neighbor_dengue','case_growth_rate','rolling_3mo_avg'
]
cat_features = ['state','season']

# Drop missing targets
df = df[df[target].notna()]
X = df[num_features + cat_features]
y = df[target]

# Encode categorical columns for SMOTE
X_encoded = X.copy()
for c in cat_features:
    X_encoded[c] = LabelEncoder().fit_transform(X_encoded[c])

# ---------------- APPLY SMOTE ----------------
print("⚖️ Applying SMOTE balancing on encoded data...")
X_res, y_res = SMOTE(random_state=42).fit_resample(X_encoded, y)
print(f"Resampled dataset: {X_res.shape[0]} samples (balanced)")

# ---------------- PIPELINE ----------------
num_pipe = Pipeline([('scaler', StandardScaler())])
cat_pipe = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('cat', cat_pipe, cat_features)
])

xgb = XGBClassifier(
    n_estimators=350,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)
clf_calibrated = CalibratedClassifierCV(xgb, method="sigmoid", cv=5)

pipe_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', clf_calibrated)
])

# ---------------- TIME SERIES CV ----------------
print("🚀 Training improved outbreak classifier with time-series CV...")
tscv = TimeSeriesSplit(n_splits=5)
aucs = []

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_res), 1):
    X_train, X_test = X_res.iloc[train_idx], X_res.iloc[test_idx]
    y_train, y_test = y_res.iloc[train_idx], y_res.iloc[test_idx]
    pipe_clf.fit(X_train, y_train)
    y_prob = pipe_clf.predict_proba(X_test)[:, 1]
    
    # Skip if only one class in test set
    if len(np.unique(y_test)) < 2:
        print(f"⚠️  Skipping Fold {fold}: Only one class in y_test")
        continue
    
    auc = roc_auc_score(y_test, y_prob)
    aucs.append(auc)
    print(f"Fold {fold}: ROC-AUC = {auc:.3f}")


print(f"\n📊 Mean ROC-AUC across folds: {np.mean(aucs):.3f}")

# ---------------- FINAL TRAIN ----------------
pipe_clf.fit(X_res, y_res)
joblib.dump(pipe_clf, os.path.join(MODELS_DIR, "xgb_outbreak_pipeline_improved.pkl"))
print("💾 Saved improved classifier model → models/xgb_outbreak_pipeline_improved.pkl")

# ---------------- REPORT ----------------
report_path = os.path.join(REPORTS_DIR, "outbreak_classifier_report.pdf")
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(report_path, pagesize=A4)
story = []
story.append(Paragraph("Improved Outbreak Classifier Report", styles['Title']))
story.append(Spacer(1, 12))
story.append(Paragraph(f"Mean ROC-AUC: {np.mean(aucs):.3f}", styles['Normal']))
story.append(Paragraph("SMOTE applied after encoding categorical data.", styles['Normal']))
story.append(Paragraph("Added trend features: case_growth_rate, rolling_3mo_avg", styles['Normal']))
story.append(Paragraph("Calibration: Sigmoid", styles['Normal']))
doc.build(story)

print("📄 Report generated → reports/outbreak_classifier_report.pdf")
print("✅ Training completed successfully!")

#!/usr/bin/env python3
# retrain_dual_models_shifted_balanced.py
import pandas as pd
import numpy as np
import joblib, os, math, json, matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, classification_report
from xgboost import XGBRegressor, XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# =============== CONFIG ===============
DATA_PATH = "final_monthly_df_shifted.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("🔍 Loading shifted dengue dataset (balanced version)...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ---- Feature setup ----
target_reg = "Future_dengue"
target_clf = "outbreak_flag"

num_features = [
    'temperature','precipitation','wind_speed','humidity','urban_density',
    'lag_dengue_cases_1','lag_dengue_cases_2','lag_dengue_cases_3',
    'lag_dengue_cases_4','lag_dengue_cases_5','lag_dengue_cases_6',
    'lag_dengue_cases_12','lag_temperature_1','lag_precipitation_1',
    'lag_humidity_1','lag_wind_speed_1','roll_mean_dengue_cases_3',
    'lag_tempxhumid_3','month_sin','month_cos','mean_state_dengue_date',
    'avg_neighbor_dengue'
]
cat_features = ['state','season']

X = df[num_features + cat_features]
y_reg = df[target_reg]
y_clf = df[target_clf]

# Drop missing target rows
mask = y_reg.notna() & y_clf.notna()
X, y_reg, y_clf = X[mask], y_reg[mask], y_clf[mask]

# =============== PIPELINES ===============
num_pipe = Pipeline([('scaler', StandardScaler())])
cat_pipe = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('cat', cat_pipe, cat_features)
])

# ---- Regressor ----
regressor = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42,
    objective="reg:squarederror"
)

pipe_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', regressor)
])

# ---- Classifier ----
positive_weight = (y_clf == 0).sum() / max((y_clf == 1).sum(), 1)
xgb_clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=positive_weight,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
clf_calibrated = CalibratedClassifierCV(xgb_clf, method="isotonic", cv=3)
pipe_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', clf_calibrated)
])

# =============== CROSS-VALIDATION ===============
tscv = TimeSeriesSplit(n_splits=5)
r2_scores, mae_scores = [], []

print("\n📆 Performing Time-Series Cross-Validation...")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    pipe_reg.fit(X_train, y_train)
    y_pred = pipe_reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Fold {fold}: R² = {r2:.3f}, MAE = {mae:.2f}")
    r2_scores.append(r2)
    mae_scores.append(mae)

print(f"\n📊 Mean R² = {np.mean(r2_scores):.3f}, Mean MAE = {np.mean(mae_scores):.2f}")

# =============== FINAL TRAINING ===============
print("\n🚀 Training final models on full dataset...")
pipe_reg.fit(X, y_reg)
pipe_clf.fit(X, y_clf)
print("💾 Models saved successfully to 'models/' folder.")

joblib.dump(pipe_reg, os.path.join(MODELS_DIR, "xgb_cases_pipeline.pkl"))
joblib.dump(pipe_clf, os.path.join(MODELS_DIR, "xgb_outbreak_pipeline.pkl"))

# =============== FEATURE IMPORTANCE ===============
model = pipe_reg.named_steps['model']
importance = model.feature_importances_
try:
    feature_names = pipe_reg.named_steps['preprocessor'].get_feature_names_out()
except Exception:
    feature_names = np.arange(len(importance))

min_len = min(len(feature_names), len(importance))
plt.figure(figsize=(9, 6))
plt.barh(feature_names[:min_len], importance[:min_len])
plt.title("Feature Importance (XGBoost Regressor)")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=120)
plt.close()
print("✅ Feature importance plot saved successfully!")

# =============== REPORT GENERATION ===============
report_path = os.path.join(REPORTS_DIR, "model_training_report.pdf")
doc = SimpleDocTemplate(report_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Dengue Forecasting Model Report (Balanced Version)", styles['Title']))
story.append(Spacer(1, 12))
story.append(Paragraph(f"Mean R²: {np.mean(r2_scores):.3f}", styles['Normal']))
story.append(Paragraph(f"Mean MAE: {np.mean(mae_scores):.2f}", styles['Normal']))
story.append(Paragraph(f"Outbreak positive weight: {positive_weight:.2f}", styles['Normal']))
story.append(Paragraph("Calibration: Isotonic regression applied on classifier", styles['Normal']))
story.append(Spacer(1, 12))
story.append(Paragraph("Top features by importance:", styles['Heading2']))
for i, (fname, imp) in enumerate(zip(feature_names[:15], importance[:15])):
    story.append(Paragraph(f"{i+1}. {fname} — {imp:.4f}", styles['Normal']))
doc.build(story)

print("📄 Model training report generated successfully.")

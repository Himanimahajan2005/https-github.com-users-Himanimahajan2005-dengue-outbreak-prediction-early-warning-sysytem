#!/usr/bin/env python3
"""
generate_evaluation_report.py

Run this in the project root (where models/ and your CSV live).
Outputs: reports/metrics_table.csv, reports/metrics_table.md, reports/figures/*.png, reports/final_report.md
"""

import os, math, joblib, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             roc_auc_score, precision_recall_curve, roc_curve,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from datetime import datetime

# -------- CONFIG --------
DATA_PATH = "final_monthly_df_shifted.csv"   # adjust if your filename differs
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
THRESHOLD = 0.35   # chosen: option B
N_SPLITS = 5       # time-series CV

# -------- Load dataset --------
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Dataset rows,cols:", df.shape)

# Make sure target names match your dataset
TARGET_REG = "Future_dengue"      # numeric target for regression
TARGET_CLF = "outbreak_flag"      # 0/1 or boolean for classification

# -------- select features used in training pipeline --------
# If you know the exact num + cat features, list them here; otherwise the script will try to
# infer from the saved pipeline's feature_names_in_ or preprocessor.
# Fallback (common features used in this project):
default_num = [
    'temperature','precipitation','wind_speed','humidity','urban_density',
    'lag_dengue_cases_1','lag_dengue_cases_2','lag_dengue_cases_3',
    'lag_dengue_cases_4','lag_dengue_cases_5','lag_dengue_cases_6',
    'lag_dengue_cases_12','lag_temperature_1','lag_precipitation_1',
    'lag_humidity_1','lag_wind_speed_1','roll_mean_dengue_cases_3',
    'lag_tempxhumid_3','month_sin','month_cos','mean_state_dengue_date',
    'avg_neighbor_dengue'
]
default_cat = ['state','season']

# -------- Load pipelines (regression + classifier) --------
def try_load_model(path_candidates):
    for p in path_candidates:
        if os.path.exists(p):
            print("Loading:", p)
            return joblib.load(p)
    return None

cases_candidates = [os.path.join(MODELS_DIR, fn) for fn in [
    "xgb_cases_pipeline.pkl","xgb_cases_pipeline.joblib","xgb_cases_model.pkl","xgb_regression.pkl","xgb_cases.pkl"
]]
outbreak_candidates = [os.path.join(MODELS_DIR, fn) for fn in [
    "xgb_outbreak_pipeline.pkl","xgb_outbreak_pipeline.joblib","xgb_outbreak_model.pkl","xgb_classifier.pkl","xgb_outbreak.pkl"
]]

cases_pipe = try_load_model(cases_candidates)
clf_pipe = try_load_model(outbreak_candidates)

if cases_pipe is None:
    raise FileNotFoundError("Regression pipeline not found in models/. Expected names: "+",".join(cases_candidates))
if clf_pipe is None:
    print("Warning: classifier pipeline not found. Classification metrics will be skipped.")
else:
    print("Classifier loaded.")

# -------- determine expected feature list from pipeline if available --------
def get_feature_names_from_pipe(pipe, fallback):
    try:
        if hasattr(pipe, "feature_names_in_") and pipe.feature_names_in_ is not None:
            return list(pipe.feature_names_in_)
        if hasattr(pipe, "named_steps") and "preprocessor" in pipe.named_steps:
            pre = pipe.named_steps["preprocessor"]
            return list(pre.get_feature_names_out())
    except Exception:
        pass
    return fallback

cases_features = get_feature_names_from_pipe(cases_pipe, default_num + default_cat)
clf_features = get_feature_names_from_pipe(clf_pipe, cases_features if clf_pipe is not None else (default_num+default_cat))

print("Regression features (using):", cases_features[:10], " ... total:", len(cases_features))
print("Classifier features (using):", clf_features[:10], " ... total:", len(clf_features))

# -------- build X and y, remove rows with missing targets --------
all_features = sorted(list(dict.fromkeys(cases_features + (clf_features or []))))
# Ensure columns exist in df; if not, create zeros to avoid pipeline errors
for col in all_features:
    if col not in df.columns:
        df[col] = 0

# filter rows where targets are available
mask = df[TARGET_REG].notna()
if TARGET_CLF in df.columns:
    mask = mask & df[TARGET_CLF].notna()

df = df[mask].reset_index(drop=True)
print("After filtering for targets:", df.shape)

X = df[all_features].copy()
y_reg = df[TARGET_REG].astype(float).copy()
y_clf = df[TARGET_CLF].astype(int).copy() if TARGET_CLF in df.columns else None

# -------- TimeSeries CV evaluation for regression and classification --------
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
reg_metrics = []
clf_metrics = []

fold = 0
for train_idx, test_idx in tscv.split(X):
    fold += 1
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_reg, y_test_reg = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    # train reg on fold
    print(f"\n--- Fold {fold} (regression) ---")
    try:
        cases_pipe.fit(X_train, y_train_reg)
    except Exception as e:
        print("Warning: pipeline fit failed on fold:", e)
        # try fit on the entire training data using pipeline's inner model if available
        pass

    y_pred = cases_pipe.predict(X_test)
    r2 = r2_score(y_test_reg, y_pred)
    mae = mean_absolute_error(y_test_reg, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test_reg, y_pred))
    reg_metrics.append({"fold": fold, "r2": r2, "mae": mae, "rmse": rmse, "n_test": len(y_test_reg)})
    # save actual vs predicted scatter figure
    plt.figure(figsize=(6,4))
    plt.scatter(y_test_reg, y_pred, alpha=0.6)
    mn = min(y_test_reg.min(), y_pred.min()); mx = max(y_test_reg.max(), y_pred.max())
    plt.plot([mn,mx],[mn,mx], "--")
    plt.xlabel("Actual cases"); plt.ylabel("Predicted cases")
    plt.title(f"Actual vs Predicted (fold {fold})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"actual_vs_pred_fold{fold}.png"), dpi=150); plt.close()

    # classifier metrics on same fold (if available)
    if clf_pipe is not None and y_clf is not None:
        print(f"--- Fold {fold} (classification) ---")
        y_test_clf = y_clf.iloc[test_idx]
        # try to fit classifier on training fold
        try:
            clf_pipe.fit(X_train, y_clf.iloc[train_idx])
        except Exception as e:
            print("Warning: classifier fit failed on fold:", e)
        # predict proba
        try:
            y_prob = clf_pipe.predict_proba(X_test)[:,1]
        except Exception as e:
            print("Warning: classifier predict_proba failed on fold:", e)
            y_prob = np.zeros(len(X_test))
        y_pred_flag = (y_prob >= THRESHOLD).astype(int)

        # compute metrics
        try:
            auc = roc_auc_score(y_test_clf, y_prob)
        except Exception:
            auc = float("nan")
        acc = accuracy_score(y_test_clf, y_pred_flag)
        prec = precision_score(y_test_clf, y_pred_flag, zero_division=0)
        rec = recall_score(y_test_clf, y_pred_flag, zero_division=0)
        f1 = f1_score(y_test_clf, y_pred_flag, zero_division=0)
        cm = confusion_matrix(y_test_clf, y_pred_flag)

        clf_metrics.append({
            "fold": fold, "auc": auc, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "tn": int(cm[0,0]), "fp": int(cm[0,1]),
            "fn": int(cm[1,0]), "tp": int(cm[1,1]), "n_test": len(y_test_clf)
        })

        # ROC, PR curve plots
        fpr, tpr, _ = roc_curve(y_test_clf, y_prob, pos_label=1)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC fold {fold} (AUC={auc:.3f})"); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"roc_fold{fold}.png"), dpi=150); plt.close()

        precs, recs, _ = precision_recall_curve(y_test_clf, y_prob)
        plt.figure(figsize=(5,4))
        plt.plot(recs, precs); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR curve fold {fold}"); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"pr_fold{fold}.png"), dpi=150); plt.close()

        # confusion matrix plot
        plt.figure(figsize=(4,3))
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion Matrix (fold {fold})")
        plt.colorbar()
        plt.xticks([0,1]); plt.yticks([0,1])
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        for (i,j), val in np.ndenumerate(cm):
            plt.text(j, i, str(val), ha="center", va="center", color="white")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"cm_fold{fold}.png"), dpi=150); plt.close()

# -------- aggregate and save metrics --------
reg_df = pd.DataFrame(reg_metrics)
clf_df = pd.DataFrame(clf_metrics)

now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
reg_df.to_csv(os.path.join(REPORTS_DIR, f"regression_metrics_{now}.csv"), index=False)
if not clf_df.empty:
    clf_df.to_csv(os.path.join(REPORTS_DIR, f"classifier_metrics_{now}.csv"), index=False)

# Also write a human-readable Markdown summary
md_lines = []
md_lines.append("# Model Performance Report")
md_lines.append(f"Generated: {now} (UTC)")
md_lines.append(f"Threshold for outbreak classification: *{THRESHOLD}*")
md_lines.append("## Regression (Future case count)")
md_lines.append(reg_df.to_markdown(index=False))
md_lines.append("\n## Classification (Outbreak detection)")
if not clf_df.empty:
    md_lines.append(clf_df.to_markdown(index=False))
else:
    md_lines.append("Classifier data not available.")

with open(os.path.join(REPORTS_DIR, f"final_report_{now}.md"), "w", encoding="utf8") as f:
    f.write("\n\n".join(md_lines))

print("Saved reports in", REPORTS_DIR)
print("Figures in", FIG_DIR)
print("Open the Markdown report file to copy text into your thesis/presentation.")
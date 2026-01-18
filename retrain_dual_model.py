import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor, XGBClassifier
import joblib, os, numpy as np

print("🔍 Loading dataset...")
df = pd.read_csv("final_monthly_df_adjusted_for_training.csv")
print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# -----------------------------
# Ensure target column exists
# -----------------------------
if "dengue_cases" not in df.columns:
    raise ValueError("❌ 'dengue_cases' column not found in dataset!")

# -----------------------------
# Create outbreak flag
# -----------------------------
threshold = df["dengue_cases"].median()
df["outbreak_flag"] = (df["dengue_cases"] > threshold).astype(int)
print(f"📊 Outbreak threshold = {threshold:.2f}")

# -----------------------------
# Split features and targets
# -----------------------------
TARGET_CASES = "dengue_cases"
TARGET_FLAG = "outbreak_flag"

X = df.drop(columns=[TARGET_CASES, TARGET_FLAG])
y_cases = df[TARGET_CASES]
y_flag = df[TARGET_FLAG]

# Identify categorical and numeric columns
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if X[c].dtype in [np.float64, np.int64]]

print(f"🧩 Found {len(num_cols)} numeric and {len(cat_cols)} categorical columns")
print("📋 Categorical columns:", cat_cols)

# -----------------------------
# Split train/test
# -----------------------------
X_train, X_test, y_train_cases, y_test_cases, y_train_flag, y_test_flag = train_test_split(
    X, y_cases, y_flag, test_size=0.2, random_state=42
)

# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# -----------------------------
# Test preprocessing manually
# -----------------------------
print("🔧 Testing preprocessing...")
try:
    sample = X_train.head(5)
    X_trans = preprocessor.fit_transform(sample)
    print(f"✅ Transformation OK: input shape {sample.shape} → transformed {X_trans.shape}")
except Exception as e:
    raise RuntimeError(f"❌ Preprocessor failed: {e}")

# -----------------------------
# Models
# -----------------------------
regressor = XGBRegressor(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

classifier = XGBClassifier(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

# -----------------------------
# Train regression model
# -----------------------------
print("🚀 Training regression model...")
pipe_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("model", regressor)
])
pipe_reg.fit(X_train, y_train_cases)
print("✅ Regression model trained successfully!")

# -----------------------------
# Train outbreak classifier
# -----------------------------
print("🚀 Training outbreak classifier...")
pipe_clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", classifier)
])
pipe_clf.fit(X_train, y_train_flag)
print("✅ Classification model trained successfully!")

# -----------------------------
# Save models
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(pipe_reg, "models/xgb_cases_pipeline.pkl")
joblib.dump(pipe_clf, "models/xgb_outbreak_pipeline.pkl")

print("💾 All pipeline models saved successfully in /models/")

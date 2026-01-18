import pandas as pd
import numpy as np
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier

print("📥 Loading dataset...")
df = pd.read_csv("final_monthly_df_adjusted_for_training.csv")
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Ensure target exists
if "dengue_cases" not in df.columns:
    raise ValueError("❌ Missing column 'dengue_cases' in dataset")

# Create outbreak flag
threshold = df["dengue_cases"].median()
df["outbreak_flag"] = (df["dengue_cases"] > threshold).astype(int)
print(f"📊 Outbreak threshold: {threshold:.2f}")

# Split targets
y_cases = df["dengue_cases"]
y_flag = df["outbreak_flag"]
X = df.drop(columns=["dengue_cases", "outbreak_flag"])

# Force categorical columns explicitly
categorical_cols = ["state", "season"]
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype("category")
    else:
        print(f"⚠️ Warning: Missing expected categorical column '{col}'")

numeric_cols = [c for c in X.columns if c not in categorical_cols]

print(f"🧩 Numeric columns: {len(numeric_cols)} | Categorical columns: {len(categorical_cols)}")

# Preprocessing definitions
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Split dataset
X_train, X_test, y_train_cases, y_test_cases = train_test_split(
    X, y_cases, test_size=0.2, random_state=42
)
_, _, y_train_flag, y_test_flag = train_test_split(
    X, y_flag, test_size=0.2, random_state=42
)

# ✅ Test preprocessing manually
print("🔧 Testing transformation on small sample...")
try:
    preprocessor.fit(X_train)
    X_sample = X_train.head(5)
    X_transformed = preprocessor.transform(X_sample)
    print(f"✅ Sample transformation OK: {X_sample.shape} → {X_transformed.shape}")
except Exception as e:
    raise RuntimeError(f"❌ Preprocessing failed: {e}")

# Define models
regressor = XGBRegressor(
    n_estimators=300, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

classifier = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="logloss", random_state=42
)

# Regression pipeline
pipe_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("model", regressor)
])

# Classification pipeline
pipe_clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", classifier)
])

# Train regression model
print("🚀 Training regression model (case count)...")
pipe_reg.fit(X_train, y_train_cases)
print("✅ Regression model trained successfully!")

# Train classification model
print("🚀 Training outbreak probability model...")
pipe_clf.fit(X_train, y_train_flag)
print("✅ Outbreak classifier trained successfully!")

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(pipe_reg, "models/xgb_cases_pipeline.pkl")
joblib.dump(pipe_clf, "models/xgb_outbreak_pipeline.pkl")

print("💾 Models saved successfully to /models/")

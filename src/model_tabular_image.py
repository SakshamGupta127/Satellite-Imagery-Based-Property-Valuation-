# ==========================================
# Tabular + Image (PCA) XGBoost Model
# params.yaml driven
# ==========================================

import pandas as pd
import xgboost as xgb
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json



# ------------------------
# Load params.yaml
# ------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)


# ------------------------
# Paths
# ------------------------
DATA_DIR = Path("data/Model_data")
TRAIN_PATH = DATA_DIR / "train_tabular_image.csv"
TEST_PATH = DATA_DIR / "test_tabular_image.csv"
OUTPUT_PATH = Path("Submission/predictions_tabular_image.csv")


# ------------------------
# Columns
# ------------------------
ID_COL = params["model"]["id_col"]
TARGET_COL = params["model"]["target_col"]


# ------------------------
# Load data
# ------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X = train_df.drop(columns=[ID_COL, TARGET_COL])
y = train_df[TARGET_COL]

X_test = test_df.drop(columns=[ID_COL])
test_ids = test_df[ID_COL]


# ------------------------
# Train / Validation split
# ------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=params["model"]["test_size"],
    random_state=params["model"]["random_state"]
)


# ------------------------
# XGBoost model
# ------------------------
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=params["xgboost"]["n_estimators"],
    learning_rate=params["xgboost"]["learning_rate"],
    max_depth=params["xgboost"]["max_depth"],
    subsample=params["xgboost"]["subsample"],
    colsample_bytree=params["xgboost"]["colsample_bytree"],
    random_state=params["model"]["random_state"],
    n_jobs=-1
)

model.fit(X_train, y_train)


# ------------------------
# Validation evaluation
# ------------------------
val_preds = model.predict(X_val)

mae = mean_absolute_error(y_val, val_preds)
rmse = mean_squared_error(y_val, val_preds) ** 0.5
r2 = r2_score(y_val, val_preds)

print("\n📊 TABULAR + IMAGE XGBOOST VALIDATION METRICS")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# ------------------------
# Save metrics (Tabular + Image)
# ------------------------
metrics = {
    "model": "tabular_image",
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}

metrics_path = Path("Submission/metrics_tabular_image.json")
metrics_path.parent.mkdir(exist_ok=True)

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)



# ------------------------
# Train on full data
# ------------------------
model.fit(X, y)


# ------------------------
# Test prediction
# ------------------------
test_preds = model.predict(X_test)


# ------------------------
# Save predictions
# ------------------------
submission = pd.DataFrame({
    ID_COL: test_ids,
    "predicted_price": test_preds
})

submission.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Predictions saved to {OUTPUT_PATH}")

import os
import yaml
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import dump


# -----------------------------
# PATHS
# -----------------------------
PARAMS_PATH = "params.yaml"

TRAIN_IMG_FEATURES = "data/image_features/train_image_features.csv"
TEST_IMG_FEATURES  = "data/image_features/test_image_features.csv"

OUTPUT_DIR = "data/image_features_pca"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# LOAD PARAMS
# -----------------------------
with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

PCA_COMPONENTS = params["pca"]["n_components"]
RANDOM_STATE = params["pca"]["random_state"]
SCALE = params["pca"]["scale"]


print(f"🔧 PCA components: {PCA_COMPONENTS}")
print(f"🔧 Scaling enabled: {SCALE}")


# -----------------------------
# LOAD DATA
# -----------------------------
train_df = pd.read_csv(TRAIN_IMG_FEATURES)
test_df  = pd.read_csv(TEST_IMG_FEATURES)

train_ids = train_df["id"]
test_ids  = test_df["id"]

X_train = train_df.drop(columns=["id"]).values
X_test  = test_df.drop(columns=["id"]).values


# -----------------------------
# OPTIONAL SCALING
# -----------------------------
if SCALE:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    print("✅ Scaler saved")


# -----------------------------
# PCA (FIT ONLY ON TRAIN)
# -----------------------------
pca = PCA(
    n_components=PCA_COMPONENTS,
    random_state=RANDOM_STATE
)

X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

dump(pca, os.path.join(OUTPUT_DIR, "pca_model.joblib"))
print("✅ PCA model saved")


# -----------------------------
# SAVE OUTPUTS (ID PRESERVED)
# -----------------------------
pca_columns = [f"img_pca_{i}" for i in range(PCA_COMPONENTS)]

train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns)
train_pca_df.insert(0, "id", train_ids)

test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns)
test_pca_df.insert(0, "id", test_ids)

train_pca_df.to_csv(
    os.path.join(OUTPUT_DIR, "train_image_features_pca.csv"),
    index=False
)

test_pca_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_image_features_pca.csv"),
    index=False
)

print("🎯 PCA features saved successfully")
print("Train PCA shape:", train_pca_df.shape)
print("Test PCA shape:", test_pca_df.shape)

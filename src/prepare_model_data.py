import pandas as pd
import os

# -----------------------------
# PATHS
# -----------------------------
TRAIN_TAB_PATH = "data/final/train_processed.csv"
TEST_TAB_PATH = "data/final/test_processed.csv"

TRAIN_IMG_PATH = "data/image_features_pca/train_image_features_pca.csv"
TEST_IMG_PATH = "data/image_features_pca/test_image_features_pca.csv"

OUTPUT_DIR = "data/Model_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
train_tab = pd.read_csv(TRAIN_TAB_PATH)
test_tab = pd.read_csv(TEST_TAB_PATH)

train_img = pd.read_csv(TRAIN_IMG_PATH)
test_img = pd.read_csv(TEST_IMG_PATH)

# -----------------------------
# REMOVE DUPLICATE IDS (TABULAR)
# -----------------------------
train_tab = train_tab.drop_duplicates(subset="id")
test_tab = test_tab.drop_duplicates(subset="id")

# -----------------------------
# SAVE CLEAN TABULAR DATA
# -----------------------------
train_tab.to_csv(f"{OUTPUT_DIR}/train_tabular.csv", index=False)
test_tab.to_csv(f"{OUTPUT_DIR}/test_tabular.csv", index=False)

# -----------------------------
# COMBINE TABULAR + IMAGE (STRICT)
# -----------------------------
train_tab_img = train_tab.merge(
    train_img,
    on="id",
    how="inner",
    validate="one_to_one"
)

test_tab_img = test_tab.merge(
    test_img,
    on="id",
    how="inner",
    validate="one_to_one"
)

# -----------------------------
# SAFETY CHECKS
# -----------------------------
assert train_tab_img.isna().sum().sum() == 0
assert test_tab_img.isna().sum().sum() == 0
assert train_tab_img["id"].is_unique
assert test_tab_img["id"].is_unique

# -----------------------------
# SAVE FINAL MODEL DATA
# -----------------------------
train_tab_img.to_csv(
    f"{OUTPUT_DIR}/train_tabular_image.csv",
    index=False
)

test_tab_img.to_csv(
    f"{OUTPUT_DIR}/test_tabular_image.csv",
    index=False
)

# -----------------------------
# LOG SUMMARY
# -----------------------------
print("✅ Model data preparation complete\n")

print("Train tabular shape:", train_tab.shape)
print("Test tabular shape:", test_tab.shape)

print("Train tabular + image shape:", train_tab_img.shape)
print("Test tabular + image shape:", test_tab_img.shape)

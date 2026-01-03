import pandas as pd
import os

# -----------------------------
# PATHS
# -----------------------------
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
OUTPUT_DIR = "data/final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# -----------------------------
# COMMON FUNCTION
# -----------------------------
def feature_engineering(df, is_train=True):
    df = df.copy()

    # -------------------------
    # DATE → SALE YEAR
    # -------------------------
    df["date"] = pd.to_datetime(
        df["date"], format="%Y%m%dT%H%M%S"
    )
    df["sale_year"] = df["date"].dt.year

    # -------------------------
    # EFFECTIVE HOUSE AGE
    # -------------------------
    df["effective_house_age"] = (
        df["sale_year"]
        - df["yr_renovated"].where(df["yr_renovated"] > 0, df["yr_built"])
    )

    # Drop invalid negative ages
    df = df[df["effective_house_age"] >= 0]

    # -------------------------
    # DROP REDUNDANT COLUMNS
    # -------------------------
    
    df = df.drop(
        columns=[
            "date",
            "yr_built",
            "yr_renovated",
            "sale_year",
            "sqft_above",
            "sqft_basement",
            "zipcode",
            "lat",
            "long"
        ]
    )

    # -------------------------
    # BEDROOMS CLEANING
    # -------------------------
    df["bedrooms"] = df["bedrooms"].clip(lower=1, upper=8)

    # -------------------------
    # BATHROOMS CLEANING
    # -------------------------
    df.loc[df["bathrooms"] == 0, "bathrooms"] = 0.75
    df["bathrooms"] = df["bathrooms"].clip(lower=0.75, upper=6)

    # -------------------------
    # LOT SIZE (NEIGHBORHOOD) CAPPING
    # -------------------------
    lot15_cap = df["sqft_lot15"].quantile(0.99)
    df["sqft_lot15"] = df["sqft_lot15"].clip(upper=lot15_cap)

    # -------------------------
    # RATIO FEATURES
    # -------------------------
    df["living_ratio"] = df["sqft_living"] / df["sqft_living15"]
    df["lot_ratio"] = df["sqft_lot"] / df["sqft_lot15"]

    # -------------------------
    # LUXURY FLAG
    # -------------------------
    df["is_luxury"] = (df["grade"] >= 10).astype(int)

    return df.reset_index(drop=True)

# -----------------------------
# APPLY
# -----------------------------
train_processed = feature_engineering(train_df, is_train=True)
test_processed = feature_engineering(test_df, is_train=False)

# -----------------------------
# SAVE
# -----------------------------
train_processed.to_csv(
    f"{OUTPUT_DIR}/train_processed.csv", index=False
)
test_processed.to_csv(
    f"{OUTPUT_DIR}/test_processed.csv", index=False
)

print("✅ Feature engineering completed.")
print(f"Train shape: {train_processed.shape}")
print(f"Test shape: {test_processed.shape}")

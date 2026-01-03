import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_SIZE = (224, 224)

TRAIN_IMG_DIR = "data/images/Train_Photos"
TEST_IMG_DIR  = "data/images/Test_Photos"

OUTPUT_DIR = "data/image_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# LOAD RESNET50 (FEATURE EXTRACTOR)
# -----------------------------
model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"   # gives 2048-d vector
)


# -----------------------------
# IMAGE → EMBEDDING FUNCTION
# -----------------------------
def extract_embedding(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    embedding = model.predict(img, verbose=0)
    return embedding.flatten()


# -----------------------------
# PROCESS FOLDER
# -----------------------------
def process_folder(img_dir):
    records = []

    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_name)

        # Extract ID from filename (12345.jpg → 12345)
        img_id = os.path.splitext(img_name)[0]

        try:
            features = extract_embedding(img_path)
            record = [img_id] + features.tolist()
            records.append(record)
        except Exception as e:
            print(f"❌ Failed for {img_name}: {e}")

    return records


# -----------------------------
# TRAIN FEATURES
# -----------------------------
print("🚀 Extracting TRAIN image features...")
train_records = process_folder(TRAIN_IMG_DIR)

train_columns = ["id"] + [f"img_feat_{i}" for i in range(2048)]
train_df = pd.DataFrame(train_records, columns=train_columns)

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train_image_features.csv"),
    index=False
)

print("✅ Train image features saved.")


# -----------------------------
# TEST FEATURES
# -----------------------------
print("🚀 Extracting TEST image features...")
test_records = process_folder(TEST_IMG_DIR)

test_df = pd.DataFrame(test_records, columns=train_columns)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test_image_features.csv"),
    index=False
)

print("✅ Test image features saved.")
print("🎯 DONE.")

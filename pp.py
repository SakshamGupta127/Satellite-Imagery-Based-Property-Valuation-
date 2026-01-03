import pandas as pd
import os

df_test = pd.read_csv("data/raw/test.csv")

missing = []
for pid in df_test["id"]:
    if not os.path.exists(f"data/images/final_photos/{pid}.png"):
        missing.append(pid)

print("Missing test images:", len(missing))

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/Model_data")

files = {
    "train_tabular": "train_tabular.csv",
    "test_tabular": "test_tabular.csv",
    "train_tabular_image": "train_tabular_image.csv",
    "test_tabular_image": "test_tabular_image.csv",
}

print("\n📊 MODEL DATA SUMMARY\n" + "-"*50)

for name, file in files.items():
    path = DATA_DIR / file

    if not path.exists():
        print(f"\n❌ {file} NOT FOUND")
        continue

    df = pd.read_csv(path)

    print(f"\n✅ {file}")
    print(f"Shape  : {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    print(list(df.columns))

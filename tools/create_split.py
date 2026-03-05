import os
import pandas as pd
import numpy as np


# ==========================
# Paths
# ==========================

DATA_ROOT = "C:/Datasets/IU_Xray"
REPORTS_PATH = os.path.join(DATA_ROOT, "indiana_reports.csv")
OUTPUT_DIR = "splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# Load Reports
# ==========================

df = pd.read_csv(REPORTS_PATH)

uids = df["uid"].unique()

print("Total unique patients:", len(uids))


# ==========================
# Shuffle (fixed seed)
# ==========================

np.random.seed(42)
np.random.shuffle(uids)


# ==========================
# 70 / 10 / 20 split
# ==========================

n = len(uids)

train_uids = uids[:int(0.7 * n)]
val_uids   = uids[int(0.7 * n):int(0.8 * n)]
test_uids  = uids[int(0.8 * n):]


split_data = []

for u in train_uids:
    split_data.append((u, "train"))

for u in val_uids:
    split_data.append((u, "val"))

for u in test_uids:
    split_data.append((u, "test"))


split_df = pd.DataFrame(split_data, columns=["uid", "split"])


# ==========================
# Save
# ==========================

output_path = os.path.join(OUTPUT_DIR, "iu_split.csv")
split_df.to_csv(output_path, index=False)

print("Split file saved at:", output_path)
print("Train:", len(train_uids))
print("Val:", len(val_uids))
print("Test:", len(test_uids))
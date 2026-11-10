"""Bagi data.csv (dari archive.zip) menjadi train/test ala Kaggle.

Jalankan SEKALI, dari folder yang sama dengan notebook:
    1. ekstrak archive.zip ke folder `archive/`
    2. python siapkan_data.py

Hasil di data_latihan/text/multilabel/:
    hs_train.csv              id, Tweet, 12 label           (untuk melatih)
    hs_test.csv               id, Tweet                     (tanpa label)
    hs_test_labels.csv        id, 12 label                  (kunci jawaban, hanya demo)
    hs_sample_submission.csv  id, 12 label berisi 0         (format pengumpulan)
    abusive.csv, new_kamusalay.csv                          (disalin apa adanya)

Isi tweet dan label TIDAK diubah. Pembagian 80/20 distratifikasi menurut kombinasi 12 label
(kombinasi < 20 baris digabung) dan tweet yang sama persis selalu masuk satu sisi saja.
"""
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SOURCE_FOLDER = "archive"                       # UBAH: folder hasil ekstrak archive.zip
OUT = "data_latihan/text/multilabel"            # UBAH
SEED = 42

os.makedirs(OUT, exist_ok=True)
data = pd.read_csv(f"{SOURCE_FOLDER}/data.csv", encoding="latin-1")   # file asli bukan UTF-8 murni
LABELS = list(data.columns[1:])
combo = pd.Series(["".join(map(str, row)) for row in data[LABELS].to_numpy()])
key = np.where(combo.map(combo.value_counts()) >= 20, combo,
               data["HS"].astype(str) + data["Abusive"].astype(str) + "_jarang")
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
index_train, index_test = next(splitter.split(data, key, groups=data["Tweet"]))

train = data.iloc[index_train].reset_index(drop=True)
test = data.iloc[index_test].reset_index(drop=True)
train.insert(0, "id", range(len(train)))
test.insert(0, "id", range(len(test)))

train.to_csv(f"{OUT}/hs_train.csv", index=False)
test[["id", "Tweet"]].to_csv(f"{OUT}/hs_test.csv", index=False)
test[["id"] + LABELS].to_csv(f"{OUT}/hs_test_labels.csv", index=False)
sample_submission = test[["id"]].copy()
for column in LABELS:
    sample_submission[column] = 0
sample_submission.to_csv(f"{OUT}/hs_sample_submission.csv", index=False)
for name in ("abusive.csv", "new_kamusalay.csv"):
    shutil.copy(f"{SOURCE_FOLDER}/{name}", f"{OUT}/{name}")
print("train", train.shape, "| test", test.shape, "->", OUT)

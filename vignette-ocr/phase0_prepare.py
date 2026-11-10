"""
phase0_prepare.py -- Fase 0: protokol evaluasi + crop 'entete' dari anotasi COCO.

Menghasilkan di --out_dir:
  meta.csv             satu baris per gambar: split, blok waktu, kotak entete/vin, path crop
  crops/*.png          crop entete (dipaksa horizontal, orientasi 180 deg BELUM dikoreksi)
  gold_candidates.csv  sampel dari split VAL untuk dilabeli manual (kolom verified_text kosong)

Contoh:
  python phase0_prepare.py --coco train/_annotations.coco.json --out_dir phase0
"""
import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TS_RE = re.compile(r"WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})")


def parse_ts(name):
    """'WIN_20200702_14_56_08_Pro...' -> datetime; None jika pola tidak cocok."""
    m = TS_RE.search(name)
    return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S") if m else None


def largest(boxes):
    """Dari beberapa kotak [x, y, w, h], ambil yang terluas (None jika kosong)."""
    return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None


def load_coco(path):
    coco = json.load(open(path, encoding="utf-8"))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}  # petakan per NAMA, bukan indeks
    per_img = {}
    for a in coco["annotations"]:
        per_img.setdefault(a["image_id"], {}).setdefault(cat_name[a["category_id"]], []).append(a["bbox"])
    rows = []
    for im in coco["images"]:
        boxes = per_img.get(im["id"], {})
        rows.append(dict(
            image_id=im["id"], file_name=im["file_name"], width=im["width"], height=im["height"],
            ts=parse_ts(im["file_name"]),
            n_entete=len(boxes.get("entete", [])),
            entete=largest(boxes.get("entete", [])), vin=largest(boxes.get("vin", [])),
        ))
    return pd.DataFrame(rows)


def blocked_split(df, block_size=30, val_frac=0.2, seed=42):
    """
    Urutkan menurut waktu, potong jadi blok berurutan, lalu acak PER BLOK.
    Frame yang berdekatan waktunya selalu masuk split yang sama -> mengurangi kebocoran.
    """
    df = df.sort_values(["ts", "file_name"], na_position="first").reset_index(drop=True)
    df["block"] = np.arange(len(df)) // block_size
    blocks = sorted(df["block"].unique())
    val_blocks = set(random.Random(seed).sample(blocks, max(1, round(len(blocks) * val_frac))))
    df["split"] = np.where(df["block"].isin(val_blocks), "val", "train")
    return df


def crop_box(img, box, pad=0.08):
    """Potong kotak [x, y, w, h] dengan padding relatif; jika tegak, putar 90 deg agar horizontal."""
    H, W = img.shape[:2]
    x, y, w, h = box
    x1, y1 = max(0, int(x - pad * w)), max(0, int(y - pad * h))
    x2, y2 = min(W, int(x + w + pad * w)), min(H, int(y + h + pad * h))
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] > crop.shape[1]:
        crop = np.ascontiguousarray(np.rot90(crop))
    return crop


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images_dir", default=None, help="default: folder file COCO")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--block_size", type=int, default=30)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--n_gold", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    images_dir = Path(a.images_dir) if a.images_dir else Path(a.coco).parent
    out = Path(a.out_dir)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    df = blocked_split(load_coco(a.coco), a.block_size, a.val_frac, a.seed)

    crop_paths = []
    for r in df.itertuples():
        img = cv2.imread(str(images_dir / r.file_name))
        if img is None or r.entete is None:        # catat, jangan hilang diam-diam
            crop_paths.append("")
            continue
        p = out / "crops" / (Path(r.file_name).stem + ".png")
        cv2.imwrite(str(p), crop_box(img, r.entete))
        crop_paths.append(str(p))
    df["crop_path"] = crop_paths
    df.to_csv(out / "meta.csv", index=False)

    val = df[(df["split"] == "val") & (df["crop_path"] != "")]
    gold = val.sample(min(a.n_gold, len(val)), random_state=a.seed)[["file_name", "crop_path"]]
    gold = gold.rename(columns={"file_name": "id"}).assign(verified_text="")
    gold.to_csv(out / "gold_candidates.csv", index=False)

    print(df.groupby("split").size().to_string())
    print("gambar tanpa entete:", int((df["entete"].isna()).sum()),
          "| gambar dengan >1 entete:", int((df["n_entete"] > 1).sum()),
          "| crop gagal:", int((df["crop_path"] == "").sum()))
    print("gold candidates:", len(gold))


if __name__ == "__main__":
    main()

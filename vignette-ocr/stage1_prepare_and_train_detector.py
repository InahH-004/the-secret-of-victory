"""
TAHAP 1 - Detector area teks (kelas: entete, vin)

Alur:
  1. Baca anotasi COCO (Roboflow).
  2. Ubah ke format YOLO (satu .txt per gambar, koordinat ternormalisasi).
  3. Split train/val berbasis BLOK WAKTU (bukan acak) supaya tidak ada kebocoran
     dari frame webcam yang berurutan.
  4. Latih YOLO (Ultralytics) -> hasilnya dipakai untuk memotong (crop) area teks.

Jalankan dari folder yang berisi _annotations_coco.json dan folder train/ (gambar).
"""
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
COCO_PATH = ROOT / "_annotations_coco.json"
IMG_SRC = ROOT / "train"            # folder gambar training (sesuaikan)
OUT = ROOT / "yolo_data"            # folder keluaran dataset format YOLO
BLOCK = 25                          # ukuran blok frame berurutan
VAL_EVERY = 5                       # setiap blok ke-5 menjadi validasi (~20%)


def build_dataset():
    with open(COCO_PATH, encoding="utf-8") as f:
        coco = json.load(f)

    # Roboflow menambahkan kategori 'Images' (id 0) sebagai supercategory kosong.
    # Kelas nyata hanya id 1 (entete) dan id 2 (vin) -> petakan ke 0 dan 1 untuk YOLO.
    real_cats = sorted(
        (c for c in coco["categories"] if c["supercategory"] != "none"),
        key=lambda c: c["id"],
    )
    cat2yolo = {c["id"]: i for i, c in enumerate(real_cats)}
    names = [c["name"] for c in real_cats]

    # Kelompokkan anotasi per gambar agar pencarian O(1), bukan loop berulang.
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    # Urutkan berdasarkan nama asli (mengandung timestamp WIN_YYYYMMDD_HH_MM_SS).
    images = sorted(coco["images"], key=lambda im: im["extra"]["name"])

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_missing, counts = 0, {"train": 0, "val": 0}
    for idx, im in enumerate(images):
        split = "val" if (idx // BLOCK) % VAL_EVERY == VAL_EVERY - 1 else "train"
        W, H = im["width"], im["height"]
        stem = Path(im["file_name"]).stem

        lines = []
        for a in anns[im["id"]]:
            x, y, w, h = a["bbox"]                      # COCO: pojok kiri-atas + lebar/tinggi (piksel)
            cx, cy = (x + w / 2) / W, (y + h / 2) / H   # YOLO: titik tengah, ternormalisasi 0-1
            lines.append(f"{cat2yolo[a['category_id']]} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

        src = IMG_SRC / im["file_name"]
        if src.exists():
            shutil.copy(src, OUT / "images" / split / im["file_name"])
        else:
            n_missing += 1
        counts[split] += 1

    yaml_text = (
        f"path: {OUT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {dict(enumerate(names))}\n"
    )
    (OUT / "data.yaml").write_text(yaml_text)
    print(f"split: {counts} | gambar tidak ditemukan: {n_missing}")
    return OUT / "data.yaml"


def train(data_yaml):
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"   # 0 = GPU CUDA pertama
    print("device training:", device)

    model = YOLO("yolov8s.pt")      # model kecil: muat di GPU 6 GB dan cepat dilatih
    model.train(
        data=str(data_yaml),
        imgsz=416,                   # sama dengan resolusi data training
        epochs=60,
        batch=16,
        seed=42,
        device=device,               # paksa pakai GPU bila tersedia
        fliplr=0.0,                  # JANGAN flip horizontal: teks akan menjadi cermin
        degrees=5,                   # rotasi kecil saja
        project="runs_vignette",
        name="detector_v1",
    )


if __name__ == "__main__":
    yaml_path = build_dataset()
    # train(yaml_path)   # aktifkan setelah isi labels/ diperiksa

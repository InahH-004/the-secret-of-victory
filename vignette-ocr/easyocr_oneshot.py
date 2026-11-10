"""
easyocr_oneshot.py -- baseline SEKALI JALAN: EasyOCR pada FOTO UTUH (tanpa detektor sendiri).

Cara kerja:
  1. EasyOCR (CRAFT + recognizer) membaca SEMUA teks di foto; rotation_info mencoba putaran
     90/180/270 deg per kotak teks dan memilih yang paling yakin  -> menangani teks tegak/terbalik.
  2. Dari semua baris, ambil segmen yang diawali kata kunci LOT / FAB / EXP
     (satu baris bisa memuat dua kunci bila CRAFT menggabungkannya -> dipecah per kunci).
  3. Rakit dengan urutan tetap LOT -> FAB -> EXP, huruf besar, hanya A-Z 0-9 / -.

Ini HEURISTIK, bukan lokalisasi sungguhan: bila kata kunci salah terbaca, bagiannya hilang.
Hasilnya dipakai sebagai batas bawah (baseline), bukan jawaban akhir.

Uji kecil dulu (menulis submission_smoke.csv, JANGAN di-submit):
  python easyocr_oneshot.py --images_dir test --n 10
Jalan penuh:
  python easyocr_oneshot.py --images_dir test --sample_submission sample_submission.csv
Kaggle tanpa internet:  tambahkan  --model_dir <folder bobot> --no_download
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

KEYS = ("LOT", "FAB", "EXP")
KEY_RE = re.compile("|".join(KEYS))
EXTS = {".jpg", ".jpeg", ".png"}


def clean(text):
    return re.sub(r"[^A-Z0-9/\-]", "", str(text).upper())


def split_segments(norm):
    """'LOT014/18FAB08/2018' -> [('LOT','LOT014/18'), ('FAB','FAB08/2018')]; segmen dimulai di kata kunci."""
    pos = [(m.start(), m.group()) for m in KEY_RE.finditer(norm)]
    out = []
    for i, (p, k) in enumerate(pos):
        end = pos[i + 1][0] if i + 1 < len(pos) else len(norm)
        out.append((k, norm[p:end]))
    return out


def assemble(results):
    """results: daftar (bbox, teks, conf) dari readtext. Return (full, exp, bagian_per_kunci)."""
    best = {}
    for _, text, conf in results:
        for k, seg in split_segments(clean(text)):
            if k not in best or conf > best[k][1]:
                best[k] = (seg, float(conf))
    full = "".join(best[k][0] for k in KEYS if k in best)
    exp = best["EXP"][0][3:] if "EXP" in best else ""      # buang awalan 'EXP'
    return full, exp, {k: best[k][0] for k in best}


def read_one(reader, bgr, mag_ratio=None):
    if mag_ratio is None:
        mag_ratio = 2.0 if max(bgr.shape[:2]) < 800 else 1.0   # gambar kecil -> perbesar agar teks kecil terdeteksi
    return reader.readtext(bgr, rotation_info=[90, 180, 270], decoder="greedy",
                           batch_size=4, mag_ratio=mag_ratio, detail=1, paragraph=False)


def run(reader, images_dir, n=None, target="full", progress=True, mag_ratio=None):
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in EXTS)
    if n:
        paths = paths[:n]
    rows, fails = [], 0
    for i, p in enumerate(paths, 1):
        row = dict(id=p.name, n_boxes=0, raw="", lot="", fab="", exp_seg="", text_full="", text_exp="")
        try:
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise ValueError("gambar tidak terbaca")
            res = read_one(reader, bgr, mag_ratio)
            full, exp, parts = assemble(res)
            row.update(n_boxes=len(res), raw=" | ".join(r[1] for r in res), lot=parts.get("LOT", ""),
                       fab=parts.get("FAB", ""), exp_seg=parts.get("EXP", ""), text_full=full, text_exp=exp)
        except Exception as e:                 # satu gambar rusak tidak boleh menghentikan 674 lainnya
            fails += 1
            row["raw"] = f"ERROR: {e}"
        rows.append(row)
        if progress and (i % 25 == 0 or i == len(paths)):
            print(f"  {i}/{len(paths)} gambar", flush=True)
    dbg = pd.DataFrame(rows)
    dbg["text"] = dbg["text_full"] if target == "full" else dbg["text_exp"]
    print(f"Selesai: {len(dbg)} gambar | gagal: {fails} | kosong (tidak ada kata kunci): {(dbg['text'] == '').sum()}")
    return dbg


def write_submission(dbg, sample_path, out_path):
    sample = pd.read_csv(sample_path)
    idc, tc = sample.columns[0], sample.columns[1]
    sub = sample[[idc]].merge(dbg[["id", "text"]].rename(columns={"id": idc, "text": tc}), on=idc, how="left")
    sub[tc] = sub[tc].fillna("")
    sub.to_csv(out_path, index=False)
    return sub


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="test")
    ap.add_argument("--sample_submission", default="sample_submission.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--target", choices=["full", "exp"], default="full")
    ap.add_argument("--n", type=int, default=None, help="uji kecil: hanya n gambar pertama")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--mag_ratio", type=float, default=None, help="default: 2.0 untuk gambar <800px, selain itu 1.0")
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    a = ap.parse_args(argv)

    import easyocr  # diimpor di sini agar run() bisa diuji tanpa EasyOCR
    reader = easyocr.Reader(["en"], gpu=not a.cpu, model_storage_directory=a.model_dir,
                            download_enabled=not a.no_download)
    dbg = run(reader, a.images_dir, a.n, a.target, mag_ratio=a.mag_ratio)
    dbg.to_csv("oneshot_debug.csv", index=False)
    print("Debug per gambar -> oneshot_debug.csv (kolom raw = semua baris yang terbaca)")

    if a.n:
        out = Path(a.out).with_name(Path(a.out).stem + "_smoke.csv")
        dbg[["id", "text"]].to_csv(out, index=False)
        print(f"MODE UJI KECIL -> {out} (jangan di-submit)")
    else:
        sub = write_submission(dbg, a.sample_submission, a.out)
        print(f"Submission -> {a.out} | baris: {len(sub)} (harus sama dengan sample_submission)")


if __name__ == "__main__":
    main()

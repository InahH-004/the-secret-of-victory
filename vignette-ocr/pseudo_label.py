"""
pseudo_label.py -- membuat label DRAF untuk crop stiker obat memakai EasyOCR.

Alur per gambar:
  crop -> (jika vertikal, putar 90 deg) -> grayscale
       -> baca pada 0 deg dan 180 deg -> pilih yang confidence-nya lebih tinggi
       -> bersihkan karakter -> simpan CSV + halaman review HTML

Contoh:
  python pseudo_label.py --crops_dir crops/ --out_dir out/ --n 50
  # offline (Kaggle): bobot english_g2.pth sudah di folder model_dir
  python pseudo_label.py --crops_dir crops/ --out_dir out/ --model_dir weights/ --no_download
"""
import argparse
import base64
import html
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ALLOW = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-"  # sama dengan vocab notebook SVTR
EXTS = {".jpg", ".jpeg", ".png"}


def sample_images(folder, n, seed):
    """Ambil n gambar secara acak tapi REPRODUCIBLE (daftar diurutkan dulu)."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)
    return random.Random(seed).sample(paths, min(n, len(paths)))


def to_horizontal_gray(bgr):
    """Grayscale + pastikan teks memanjang horizontal (tinggi <= lebar)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] > gray.shape[1]:
        gray = np.ascontiguousarray(np.rot90(gray))  # arah ambigu; varian 180 deg menutupinya
    return gray


def clean_text(text):
    return re.sub(r"[^A-Z0-9/\-]", "", text.upper())


def read_variant(reader, gray, angle):
    """Baca satu orientasi. Return (teks_bersih, confidence)."""
    img = gray if angle == 0 else cv2.rotate(gray, cv2.ROTATE_180)
    # recognize() tanpa horizontal_list = seluruh gambar diperlakukan sebagai satu baris teks
    result = reader.recognize(img, detail=1, allowlist=ALLOW, decoder="greedy")
    if not result:
        return "", 0.0
    text = clean_text("".join(r[1] for r in result))
    conf = float(np.mean([r[2] for r in result]))
    return text, conf


def png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def write_review_html(rows, thumbs, path, blind):
    """Halaman review: baris yang perlu dicek ada di atas."""
    ordered = sorted(rows, key=lambda r: (not r["needs_review"], r["conf"]))
    head = "<tr><th>id</th><th>crop</th>" + ("" if blind else "<th>draf</th><th>conf</th><th>putar</th>") + "</tr>"
    body = []
    for r in ordered:
        img = f'<img src="data:image/png;base64,{thumbs[r["id"]]}" style="max-width:420px">' if r["id"] in thumbs else "-"
        extra = "" if blind else (
            f'<td style="font:16px monospace">{html.escape(r["draft_text"])}</td>'
            f'<td>{r["conf"]:.2f}{" ⚠" if r["needs_review"] else ""}</td><td>{r["angle"]}°</td>'
        )
        body.append(f'<tr><td>{html.escape(r["id"])}</td><td>{img}</td>{extra}</tr>')
    doc = ('<meta charset="utf-8"><style>td,th{border:1px solid #ccc;padding:4px}</style>'
           f'<table>{head}{"".join(body)}</table>')
    Path(path).write_text(doc, encoding="utf-8")


def run(reader, crops_dir, out_dir, n=50, seed=42, conf_thr=0.6, margin_thr=0.15, blind=False):
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    rows, thumbs = [], {}

    for p in sample_images(crops_dir, n, seed):
        bgr = cv2.imread(str(p))
        if bgr is None:  # file rusak: tetap dicatat agar tidak hilang diam-diam
            rows.append(dict(id=p.name, draft_text="", conf=0.0, angle=0, text_0="", conf_0=0.0,
                             text_180="", conf_180=0.0, needs_review=True, verified_text=""))
            continue

        gray = to_horizontal_gray(bgr)
        t0, c0 = read_variant(reader, gray, 0)
        t180, c180 = read_variant(reader, gray, 180)

        angle = 180 if c180 > c0 else 0          # seri -> anggap tidak terbalik
        text, conf = (t180, c180) if angle == 180 else (t0, c0)
        margin = abs(c0 - c180)                   # kecil = model "ragu" soal orientasi

        oriented = gray if angle == 0 else cv2.rotate(gray, cv2.ROTATE_180)
        cv2.imwrite(str(out_dir / "crops" / f"{p.stem}.png"), oriented)
        thumbs[p.name] = png_b64(oriented)

        rows.append(dict(
            id=p.name, draft_text=text, conf=round(conf, 4), angle=angle,
            text_0=t0, conf_0=round(c0, 4), text_180=t180, conf_180=round(c180, 4),
            needs_review=bool(conf < conf_thr or margin < margin_thr or text == ""),
            verified_text="",                      # DIISI MANUAL oleh kamu
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pseudo_labels.csv", index=False)
    write_review_html(rows, thumbs, out_dir / "review.html", blind)
    print(f"{len(df)} gambar | perlu review: {int(df['needs_review'].sum())} | "
          f"dibalik 180 deg: {int((df['angle'] == 180).sum())}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", default="phase0/crops")
    ap.add_argument("--out_dir", default="pseudo_out")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf_thr", type=float, default=0.6)
    ap.add_argument("--margin_thr", type=float, default=0.15)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--blind", action="store_true", help="sembunyikan draf di HTML (label buta)")
    if argv is None and "ipykernel" in sys.modules:  # dijalankan dari sel notebook: abaikan argumen kernel
        argv = []
    a = ap.parse_args(argv)

    import easyocr  # diimpor di sini agar run() bisa diuji tanpa EasyOCR
    reader = easyocr.Reader(["en"], gpu=not a.cpu, detector=False,  # CRAFT tidak diperlukan
                            model_storage_directory=a.model_dir,
                            download_enabled=not a.no_download)
    run(reader, a.crops_dir, a.out_dir, a.n, a.seed, a.conf_thr, a.margin_thr, a.blind)


if __name__ == "__main__":
    main()

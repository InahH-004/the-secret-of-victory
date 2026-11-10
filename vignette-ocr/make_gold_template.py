"""
make_gold_template.py -- buat template Excel untuk melabeli GOLD SET secara manual.

Mengambil N baris pertama dari gold_candidates.csv (hasil phase0_prepare.py; itu sudah sampel
ACAK dari split val, jadi N baris pertama tetap sampel acak), menyalin crop-nya ke gold/crops/,
lalu menulis gold/gold_template.xlsx dengan gambar tertanam di tiap baris.

Pakai:
  python make_gold_template.py --candidates phase0/gold_candidates.csv --out_dir gold --n 50
  (di notebook:  import make_gold_template as g;  g.main(["--n", "50"]) )
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image as PILImage

FONT = "Arial"
YELLOW = PatternFill("solid", start_color="FFFF99")   # sel yang HARUS diisi
HEAD = PatternFill("solid", start_color="D9E1F2")
THIN = Side(style="thin", color="999999")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
MAX_W, MAX_H = 360, 130                                # ukuran tampil gambar (piksel)

COLS = ["no", "id", "gambar", "orientasi", "teks_mentah", "exp_saja", "dapat_dibaca", "catatan"]
WIDTHS = [5, 30, 52, 14, 42, 16, 14, 28]


def f(bold=False, color="000000", size=10):
    return Font(name=FONT, bold=bold, color=color, size=size)


def build_panduan(wb, n):
    ws = wb.active
    ws.title = "Panduan"
    for col, w in zip("ABCDEFG", [30, 46, 16, 16, 16, 16, 16]):
        ws.column_dimensions[col].width = w
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0

    ws["A1"] = "Panduan pelabelan gold set"
    ws["A1"].font = f(True, size=14)
    ws["A2"] = "Progres (teks_mentah terisi):"
    ws["A2"].font = f(True)
    ws["B2"] = f'=COUNTA(Label!E2:E{n + 1})&" / {n}"'
    ws["B2"].font = f(True, "0000FF")

    rules = [
        ("Sel kuning", "di sheet 'Label', kolom orientasi, teks_mentah, exp_saja, dapat_dibaca, catatan -> ISI. Kolom lain jangan diubah."),
        ("orientasi", "pilih dari dropdown, MENGACU PADA GAMBAR SEPERTI TAMPIL: 'tegak' = teks terbaca normal kiri-ke-kanan; "
                      "'terbalik' = teks terbalik 180 derajat; 'lainnya' = miring/vertikal/tidak jelas."),
        ("teks_mentah", "ketik SEMUA teks pada jalur crop, urut baca, APA ADANYA sesuai yang tercetak (spasi boleh, huruf besar/kecil bebas). "
                        "Jangan memperbaiki salah cetak. Karakter yang benar-benar tak terbaca tulis '?' (satu '?' per karakter)."),
        ("exp_saja", "hanya tanggal setelah EXP, apa adanya. Kosongkan bila tidak ada/tak terbaca."),
        ("dapat_dibaca", "ya = semua terbaca; sebagian = ada karakter '?'; tidak = hampir tak terbaca."),
        ("Mengapa 2 kolom teks?", "Format teks target kompetisi belum pasti (seluruh jalur atau hanya EXP). Dengan dua kolom ini kamu "
                                  "tidak perlu melabeli ulang apa pun formatnya."),
        ("Label buta", "isi label SEBELUM melihat hasil EasyOCR, agar tidak terpengaruh drafnya."),
        ("Jangan", "buka lalu simpan sebagai CSV di Excel. Simpan tetap sebagai .xlsx."),
    ]
    r = 4
    for k, v in rules:
        lab = ws.cell(r, 1, k)
        lab.font, lab.alignment = f(True), Alignment(vertical="top", wrap_text=True)
        c = ws.cell(r, 2, v)
        c.font = f()
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=7)
        ws.row_dimensions[r].height = 44
        r += 1

    r += 1
    ws.cell(r, 1, "CONTOH format isian (ilustrasi dari 3 gambar di EDA; JANGAN diisi di sini)").font = f(True)
    r += 1
    for j, h in enumerate(["orientasi", "teks_mentah", "exp_saja", "dapat_dibaca"], start=1):
        c = ws.cell(r, j, h)
        c.font, c.fill, c.border = f(True), HEAD, BORDER
    examples = [
        ("tegak", "LOT 014/18 FAB 08/2018 EXP 08/2021", "08/2021", "ya"),
        ("terbalik", "N LOT 1014/0362 FAB 04/2019 EXP 03/2022", "03/2022", "ya"),
        ("tegak", "LOT N 251 FAB MAR 2020 EXP MAR 2022", "MAR 2022", "ya"),
    ]
    for ex in examples:
        r += 1
        for j, v in enumerate(ex, start=1):
            c = ws.cell(r, j, v)
            c.font, c.border, c.number_format = f(color="0000FF"), BORDER, "@"
    ws.cell(r + 2, 1, "Catatan: contoh di atas ditulis dari pembacaan visual gambar EDA dan hanya untuk menunjukkan format.").font = f(size=9)


def build_label(wb, rows, crops_dir):
    ws = wb.create_sheet("Label")
    n = len(rows)
    for j, (h, w) in enumerate(zip(COLS, WIDTHS), start=1):
        c = ws.cell(1, j, h)
        c.font, c.fill, c.border = f(True), HEAD, BORDER
        c.alignment = Alignment(horizontal="center")
        ws.column_dimensions[chr(64 + j)].width = w
    ws.freeze_panes = "A2"
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0

    dv1 = DataValidation(type="list", formula1='"tegak,terbalik,lainnya"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"ya,sebagian,tidak"', allow_blank=True)
    ws.add_data_validation(dv1)
    ws.add_data_validation(dv2)

    for i, name in enumerate(rows, start=1):
        r = i + 1
        ws.cell(r, 1, i)
        ws.cell(r, 2, name)
        with PILImage.open(crops_dir / name) as im:
            w, h = im.size
        s = min(MAX_W / w, MAX_H / h)
        img = XLImage(str(crops_dir / name))
        img.width, img.height = int(w * s), int(h * s)
        ws.add_image(img, f"C{r}")
        ws.row_dimensions[r].height = max(40, img.height * 0.75 + 8)   # piksel -> poin

        for j in range(1, len(COLS) + 1):
            c = ws.cell(r, j)
            c.font, c.border = f(), BORDER
            c.alignment = Alignment(vertical="center", wrap_text=True)
            if j in (4, 5, 6, 7, 8):
                c.fill = YELLOW
            if j in (5, 6):
                c.number_format = "@"          # simpan sebagai TEKS: '12/25' tidak jadi tanggal, '0123' tetap '0123'
        dv1.add(f"D{r}")
        dv2.add(f"G{r}")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="phase0/gold_candidates.csv")
    ap.add_argument("--out_dir", default="gold")
    ap.add_argument("--n", type=int, default=50)
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    a = ap.parse_args(argv)

    cand = pd.read_csv(a.candidates, dtype=str, keep_default_na=False).head(a.n)
    out = Path(a.out_dir)
    crops = out / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    names = []
    for p in cand["crop_path"]:
        src = Path(p)
        if not src.exists():
            print("LEWATI (tidak ada):", p)
            continue
        shutil.copy2(src, crops / src.name)
        names.append(src.name)

    wb = Workbook()
    build_panduan(wb, len(names))
    build_label(wb, names, crops)
    path = out / "gold_template.xlsx"
    wb.save(path)
    print(f"{len(names)} baris -> {path}\nCrop gold disalin ke {crops} (dipakai juga oleh pseudo_label.py)")


if __name__ == "__main__":
    main()

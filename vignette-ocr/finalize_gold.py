"""
finalize_gold.py -- baca gold_template.xlsx yang sudah kamu isi, validasi, normalisasi, dan
sambungkan ke hasil pseudo_label.py supaya eval_cer.py bisa dipakai.

  python finalize_gold.py --xlsx gold/gold_template.xlsx --pseudo pseudo_out/pseudo_labels.csv --target full
  python finalize_gold.py --labels gold/gold_labels.csv   --pseudo pseudo_out/pseudo_labels.csv   # dari GUI

--target full : verified_text = teks_mentah dinormalisasi   (kalau targetnya SELURUH jalur)
--target exp  : verified_text = exp_saja dinormalisasi      (kalau targetnya hanya tanggal EXP)
Normalisasi = HURUF BESAR + hanya karakter A-Z 0-9 / -  (sama dengan clean_text di pipeline referensi).
UBAH aturan ini bila format target kompetisi ternyata berbeda.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def normalize(s):
    return re.sub(r"[^A-Z0-9/\-]", "", str(s).upper())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="gold/gold_template.xlsx")
    ap.add_argument("--labels", default=None, help="gold_labels.csv dari gold_label_gui.py (menggantikan --xlsx)")
    ap.add_argument("--pseudo", default=None, help="pseudo_labels.csv dari pseudo_label.py (opsional)")
    ap.add_argument("--target", choices=["full", "exp"], default="full")
    ap.add_argument("--out", default=None)
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    a = ap.parse_args(argv)

    # dtype=str: nilai seperti 0123 tetap teks; keep_default_na=False: sel kosong = "" (bukan NaN)
    if a.labels:
        g = pd.read_csv(a.labels, dtype=str, keep_default_na=False)
    else:
        g = pd.read_excel(a.xlsx, sheet_name="Label", dtype=str, keep_default_na=False)
    src = "teks_mentah" if a.target == "full" else "exp_saja"
    g["verified_text"] = g[src].map(normalize)

    n = len(g)
    filled = (g[src].str.strip() != "").sum()
    print(f"Baris: {n} | {src} terisi: {filled} | kosong: {n - filled}")
    print("Orientasi:", g["orientasi"].replace("", "(kosong)").value_counts().to_dict())
    print("Dapat dibaca:", g["dapat_dibaca"].replace("", "(kosong)").value_counts().to_dict())
    q = g[g[src].str.contains(r"\?", regex=True)]
    if len(q):
        print(f"PERHATIAN: {len(q)} baris berisi '?' (karakter tak terbaca) -> dikeluarkan dari verified_text:", q["id"].tolist()[:5])
        g.loc[q.index, "verified_text"] = ""          # jangan menghitung CER terhadap label yang tidak pasti

    out = Path(a.out) if a.out else Path(a.labels or a.xlsx).with_name("gold.csv")
    g.to_csv(out, index=False)
    print("Ditulis:", out)

    if a.pseudo:
        p = pd.read_csv(a.pseudo, dtype=str, keep_default_na=False)
        m = p.drop(columns=["verified_text"], errors="ignore").merge(
            g[["id", "verified_text", "orientasi", "dapat_dibaca"]], on="id", how="inner")
        print(f"Cocok dengan pseudo-label: {len(m)} dari {len(g)} baris gold")
        # Orientasi: angle=0 berarti draf membaca crop apa adanya -> benar bila gold 'tegak'
        lab = m[m["orientasi"].isin(["tegak", "terbalik"])]
        if len(lab):
            pred_flip = lab["angle"].astype(int) == 180
            true_flip = lab["orientasi"] == "terbalik"
            print(f"Akurasi pemilihan orientasi EasyOCR (2 pass): {(pred_flip == true_flip).mean():.1%} dari {len(lab)} baris")
        out2 = Path(a.pseudo).with_name("pseudo_labels_verified.csv")
        m.to_csv(out2, index=False)
        print("Siap untuk eval_cer.py ->", out2)


if __name__ == "__main__":
    main()

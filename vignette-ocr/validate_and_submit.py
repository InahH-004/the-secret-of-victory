"""
validate_and_submit.py -- validasi EasyOCR terhadap label emas buatanmu, lalu buat submission.csv.

Dua pengukuran pada baris berlabel yang SAMA:
  [A] UJUNG-KE-UJUNG : EasyOCR pada FOTO UTUH sumber crop (pipeline persis seperti submission) -> wakil skor submission
  [B] HANYA PENGENALAN: EasyOCR pada CROP emas (anggap lokalisasi sempurna)                      -> batas atas
Selisih A - B = kerugian karena lokalisasi + orientasi heuristik.

CER mengikuti rumus kompetisi:  (S + D + I) / N,  N = total karakter referensi.
  - CER mikro  = total edit / total karakter referensi (agregat)
  - CER makro  = rata-rata CER per sampel
  - 95% CI     = bootstrap atas SAMPEL (bukan karakter), karena kesalahan satu sampel saling terkait

Contoh:
  python validate_and_submit.py --labels gold/gold_labels.csv --images_dir train --crops_dir gold/crops
  python validate_and_submit.py --skip_submission                 # hanya validasi
  python validate_and_submit.py --n_test 10                       # uji kecil submission (tidak untuk di-submit)
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

import easyocr_oneshot as eo
import pseudo_label as pl
from finalize_gold import normalize

PHOTO_EXTS = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------- metrik
def sdi(hyp, ref):
    """Hitung S/D/I menurut definisi kompetisi (D = karakter referensi yang hilang, I = karakter berlebih)."""
    c = Counter()
    for op in Levenshtein.editops(hyp, ref):       # editops mengubah hyp -> ref
        c[{"replace": "S", "insert": "D", "delete": "I"}[op.tag]] += 1
    return c["S"], c["D"], c["I"]


def cer_table(hyps, refs, boot=2000, seed=0):
    d = np.array([Levenshtein.distance(h, r) for h, r in zip(hyps, refs)], float)
    L = np.array([len(r) for r in refs], float)
    micro = d.sum() / L.sum()
    macro = float(np.mean(d / L))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (boot, len(d)))
    lo, hi = np.percentile(d[idx].sum(1) / L[idx].sum(1), [2.5, 97.5])
    S = D = I = 0
    for h, r in zip(hyps, refs):
        s, dd, i = sdi(h, r)
        S, D, I = S + s, D + dd, I + i
    return dict(n=len(d), micro=micro, lo=lo, hi=hi, macro=macro, exact=float((d == 0).mean()),
                S=S, D=D, I=I, N=int(L.sum())), d / L


def confusions(hyps, refs, k=8):
    c = Counter()
    for h, r in zip(hyps, refs):
        for op in Levenshtein.opcodes(h, r):
            if op.tag == "replace":
                for i, j in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
                    c[(r[j], h[i])] += 1
    return c.most_common(k)


def show(name, t):
    print(f"  {name:<34} CER mikro {t['micro']:.3f}  [95% CI {t['lo']:.3f}-{t['hi']:.3f}]   "
          f"makro {t['macro']:.3f}   exact {t['exact']:.0%}")
    print(f"  {'':<34} dari N={t['N']} karakter: S={t['S']}  D={t['D']}  I={t['I']}")


# ----------------------------------------------------------------------------- data
def load_gold(labels, xlsx, target):
    if labels and Path(labels).exists():
        g = pd.read_csv(labels, dtype=str, keep_default_na=False)
    else:
        g = pd.read_excel(xlsx, sheet_name="Label", dtype=str, keep_default_na=False)
    src = "teks_mentah" if target == "full" else "exp_saja"
    g["raw_label"] = g[src]
    g["ref"] = g[src].map(normalize)
    n0 = len(g)
    g = g[(g["raw_label"].str.strip() != "") & (~g["raw_label"].str.contains(r"\?", regex=True))]
    print(f"Label: {n0} baris | dipakai: {len(g)} (kosong atau berisi '?' dikeluarkan: {n0 - len(g)})")
    return g.reset_index(drop=True)


def find_photo(images_dir, crop_name):
    stem = Path(crop_name).stem               # stem tetap memuat '.rf.<hash>' -> hanya ekstensi terakhir dibuang
    for e in PHOTO_EXTS:
        p = Path(images_dir) / (stem + e)
        if p.exists():
            return p
    return None


# ----------------------------------------------------------------------------- validasi
def validate(reader, gold, images_dir, crops_dir, target, mag_ratio=None, report="validation_report.csv"):
    rows = []
    for r in gold.itertuples():
        row = dict(id=r.id, ref=r.ref, orientasi=getattr(r, "orientasi", ""))
        photo = find_photo(images_dir, r.id)
        row["photo_found"] = photo is not None
        if photo is not None:                                          # [A] foto utuh
            res = eo.read_one(reader, cv2.imread(str(photo)), mag_ratio)
            full, exp, _ = eo.assemble(res)
            row["hyp_e2e"] = full if target == "full" else exp
            row["raw_e2e"] = " | ".join(x[1] for x in res)
        crop = cv2.imread(str(Path(crops_dir) / r.id))
        if crop is not None:                                           # [B] crop emas, 2 orientasi
            gray = pl.to_horizontal_gray(crop)
            (t0, c0), (t1, c1) = pl.read_variant(reader, gray, 0), pl.read_variant(reader, gray, 180)
            row["hyp_crop"], row["angle"] = (t1, 180) if c1 > c0 else (t0, 0)
        rows.append(row)
    df = pd.DataFrame(rows)

    print(f"\n=== Validasi (target={target}) ===")
    out = {}
    for key, name in [("hyp_e2e", "[A] ujung-ke-ujung (foto utuh)"), ("hyp_crop", "[B] hanya pengenalan (crop emas)")]:
        if key not in df or df[key].isna().all():
            print(f"  {name}: tidak ada data (cek --images_dir / --crops_dir)")
            continue
        sub = df[df[key].notna()]
        t, per = cer_table(sub[key].tolist(), sub["ref"].tolist())
        df.loc[sub.index, "cer_" + key[4:]] = per
        out[key] = t
        show(name, t)
        cf = confusions(sub[key].tolist(), sub["ref"].tolist())
        if cf:
            print("  substitusi tersering (benar->terbaca): " + ", ".join(f"{a}->{b} x{n}" for (a, b), n in cf))
    if "hyp_e2e" in out and "hyp_crop" in out:
        print(f"\n  Kerugian karena lokalisasi+orientasi (A-B): {out['hyp_e2e']['micro'] - out['hyp_crop']['micro']:+.3f}")
    if "hyp_e2e" in df and (df["hyp_e2e"].fillna("") == "").any():
        print(f"  Foto tanpa kata kunci LOT/FAB/EXP terbaca: {(df['hyp_e2e'].fillna('') == '').sum()} dari {len(df)}")
    lab = df[df["orientasi"].isin(["tegak", "terbalik"]) & df["angle"].notna()] if "angle" in df else df.iloc[0:0]
    if len(lab):
        acc = ((lab["angle"] == 180) == (lab["orientasi"] == "terbalik")).mean()
        print(f"  Akurasi pemilihan orientasi (crop, 2 pass): {acc:.0%} dari {len(lab)} baris berlabel orientasi")
    df.to_csv(report, index=False)
    print(f"\nLaporan per sampel -> {report} (urutkan kolom cer_e2e untuk melihat kasus terburuk)")
    return df, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="gold/gold_labels.csv")
    ap.add_argument("--xlsx", default="gold/gold_template.xlsx", help="dipakai bila --labels tidak ada")
    ap.add_argument("--images_dir", default="train", help="folder FOTO UTUH sumber crop emas")
    ap.add_argument("--crops_dir", default="gold/crops")
    ap.add_argument("--test_dir", default="test")
    ap.add_argument("--sample_submission", default="sample_submission.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--target", choices=["full", "exp"], default="full")
    ap.add_argument("--mag_ratio", type=float, default=None)
    ap.add_argument("--skip_submission", action="store_true")
    ap.add_argument("--n_test", type=int, default=None, help="uji kecil submission: n gambar test pertama")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    a = ap.parse_args(argv)

    gold = load_gold(a.labels, a.xlsx, a.target)
    if gold.empty:
        sys.exit("Belum ada label yang bisa dipakai. Isi dulu lewat gold_label_gui.py.")

    import easyocr
    reader = easyocr.Reader(["en"], gpu=not a.cpu, model_storage_directory=a.model_dir,
                            download_enabled=not a.no_download)
    validate(reader, gold, a.images_dir, a.crops_dir, a.target, a.mag_ratio)

    if a.skip_submission:
        return
    print("\n=== Membuat submission ===")
    dbg = eo.run(reader, a.test_dir, a.n_test, a.target, mag_ratio=a.mag_ratio)
    dbg.to_csv("oneshot_debug.csv", index=False)
    if a.n_test:
        out = Path(a.out).with_name(Path(a.out).stem + "_smoke.csv")
        dbg[["id", "text"]].to_csv(out, index=False)
        print(f"MODE UJI KECIL -> {out} (jangan di-submit)")
    else:
        sub = eo.write_submission(dbg, a.sample_submission, a.out)
        print(f"Submission -> {a.out} | baris: {len(sub)} | teks kosong: {(sub.iloc[:, 1] == '').sum()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bentuk submission.csv dari hasil OCR yang sudah di-cache (tanpa YOLO / EasyOCR).

Contoh:
    python make_submission.py                       # memakai ocr_yolo.json
    python make_submission.py --ocr ocr_full.json   # hasil mode 'full'
    python make_submission.py --min-conf 0.25       # lebih ketat membuang bacaan ragu

Format keluaran (sesuai aturan lomba): kolom `id,text`; teks HURUF BESAR, hanya
A-Z, 0-9, '/' dan '-'; tanpa spasi atau baris baru.
"""
import argparse
import csv
import json
import re
import sys

_DASH = str.maketrans({'–': '-', '—': '-', '‒': '-', '−': '-', '_': '-', '\\': '/'})
_VALID = re.compile(r'[^A-Z0-9/-]')
_FORMAT = re.compile(r'[A-Z0-9/-]+')


def clean_text(s):
    """Huruf besar; hanya A-Z, 0-9, '/' dan '-'; semua spasi/baris baru hilang."""
    return _VALID.sub('', s.translate(_DASH).upper())


def to_items(dets):
    items = []
    for d in dets:
        xs = [p[0] for p in d['box']]
        ys = [p[1] for p in d['box']]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        w, h = x2 - x1, y2 - y1
        txt = clean_text(d['text'])
        items.append(dict(x1=x1, x2=x2, w=w, h=h, cy=(y1 + y2) / 2, text=txt, conf=d['conf'],
                          vertical=(h > 1.5 * w and len(txt) >= 2)))
    return [it for it in items if it['text']]


def merge_lines(items, y_tol=0.6, gap_tol=2.5):
    """Gabungkan potongan teks yang sebaris (EasyOCR sering memecah satu ID menjadi beberapa bagian)."""
    mean = lambda v: sum(v) / len(v)
    lines = []
    for it in sorted([i for i in items if not i['vertical']], key=lambda i: i['cy']):
        for ln in lines:
            cy = mean([m['cy'] for m in ln])
            hh = max(it['h'], mean([m['h'] for m in ln]))
            gap = min(max(0, it['x1'] - m['x2'], m['x1'] - it['x2']) for m in ln)
            if abs(it['cy'] - cy) < y_tol * hh and gap < gap_tol * hh:
                ln.append(it)
                break
        else:
            lines.append([it])
    out = []
    for ln in lines:
        ln = sorted(ln, key=lambda m: m['x1'])
        wts = [len(m['text']) for m in ln]
        out.append(dict(text=''.join(m['text'] for m in ln), cy=mean([m['cy'] for m in ln]),
                        conf=sum(m['conf'] * w for m, w in zip(ln, wts)) / sum(wts)))
    out += [dict(text=i['text'], cy=i['cy'], conf=i['conf']) for i in items if i['vertical']]
    return out


def crop_text(dets, min_conf):
    """Isi crop `entete`: semua baris dibaca atas -> bawah lalu digabung tanpa spasi."""
    lines = merge_lines([it for it in to_items(dets) if it['conf'] >= min_conf])
    lines.sort(key=lambda l: l['cy'])
    return ''.join(l['text'] for l in lines).strip('/-')


def score_line(ln):
    t = ln['text']
    if len(t.strip('/-')) < 3:
        return 0.0
    s = ln['conf'] * min(len(t), 20)
    if any(c.isdigit() for c in t) and any(c in '/-' for c in t):   # pola khas ID: angka + pemisah
        s *= 1.5
    return s


def best_line_text(dets):
    """Cadangan / mode full: pilih SATU baris terbaik dari seluruh gambar."""
    lines = merge_lines(to_items(dets))
    best = max(lines, key=score_line, default=None)
    if best is not None and score_line(best) > 0:
        return best['text'].strip('/-') or '0'
    return '0'


def final_text(entry, min_conf):
    if entry.get('src') == 'crop':
        return crop_text(entry['dets'], min_conf) or '0'
    return best_line_text(entry.get('dets', []))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', default='test.csv', help='CSV daftar id test (kolom pertama = id)')
    ap.add_argument('--ocr', default='ocr_yolo.json', help='cache hasil OCR')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--min-conf', type=float, default=0.15,
                    help='buang bacaan EasyOCR dengan confidence di bawah ini (hanya untuk crop)')
    a = ap.parse_args()

    with open(a.test, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        id_col = next(reader)[0]
        ids = [row[0] for row in reader if row]
    with open(a.ocr, encoding='utf-8') as f:
        cache = json.load(f)

    rows, n_missing, n_zero = [], 0, 0
    for img_id in ids:
        entry = cache.get(img_id)
        if entry is None:
            n_missing += 1
            text = '0'
        else:
            text = final_text(entry, a.min_conf)
        n_zero += text == '0'
        rows.append((img_id, text))

    bad = [r for r in rows if not _FORMAT.fullmatch(r[1])]
    if bad or len(set(ids)) != len(ids):
        sys.exit(f"Format tidak valid / id duplikat: {bad[:3]}")

    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow([id_col, 'text'])
        w.writerows(rows)

    print(f"{a.out} tersimpan: {len(rows)} baris")
    print(f"  belum di-OCR (diisi '0'): {n_missing}")
    print(f"  total berisi '0'        : {n_zero}")
    if n_missing:
        print("  PERINGATAN: OCR belum lengkap. Selesaikan sel OCR di notebook lalu jalankan ulang skrip ini.")
    for img_id, text in rows[:5]:
        print(f"  {img_id},{text}")


if __name__ == '__main__':
    main()

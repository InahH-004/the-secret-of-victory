"""
gold_label_gui.py -- GUI sederhana untuk melabeli gold set (pengganti mengisi Excel).

Jalankan dari TERMINAL (bukan sel notebook), dengan .venv aktif:
    python gold_label_gui.py
atau dengan path eksplisit:
    python gold_label_gui.py --crops_dir gold/crops --labels gold/gold_labels.csv --template gold/gold_template.xlsx

Hasil disimpan otomatis ke gold/gold_labels.csv setiap berpindah gambar. Tutup kapan saja, buka lagi
akan melanjutkan dari gambar pertama yang belum terisi. Jika gold_labels.csv belum ada tetapi
gold_template.xlsx sudah sebagian kamu isi, isiannya diimpor otomatis.

Lalu:  python finalize_gold.py --labels gold/gold_labels.csv --pseudo pseudo_out/pseudo_labels.csv

Tombol cepat:  F1/F2/F3 = orientasi tegak/terbalik/lainnya | Ctrl+R = putar TAMPILAN 180 deg
               Alt+Kiri/Kanan = sebelumnya/berikutnya | Ctrl+Enter = simpan & ke yang belum terisi
               Ctrl+E = isi exp_saja dari teks_mentah | Ctrl+S = simpan
"""
import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

FIELDS = ["id", "orientasi", "teks_mentah", "exp_saja", "dapat_dibaca", "catatan"]
EDIT = FIELDS[1:]
EXTS = {".png", ".jpg", ".jpeg"}


# ----------------------------------------------------------------------------- logika (tanpa GUI)
class LabelStore:
    """Menyimpan label per id, memuat/menyimpan CSV secara atomik, dan menghitung progres."""

    def __init__(self, crops_dir, csv_path, template=None):
        self.crops_dir, self.csv_path = Path(crops_dir), Path(csv_path)
        self.ids = self._list_ids(template)
        if not self.ids:
            raise SystemExit(f"Tidak ada gambar di {crops_dir}. Jalankan make_gold_template.py dulu.")
        self.rows = {i: {k: "" for k in EDIT} for i in self.ids}
        self._load(template)

    def _list_ids(self, template):
        on_disk = {p.name for p in self.crops_dir.iterdir() if p.suffix.lower() in EXTS} if self.crops_dir.exists() else set()
        if template and Path(template).exists():           # pakai urutan di template agar sama dengan Excel
            t = pd.read_excel(template, sheet_name="Label", dtype=str, keep_default_na=False)
            ordered = [i for i in t["id"] if i in on_disk]
            return ordered + sorted(on_disk - set(ordered))
        return sorted(on_disk)

    def _load(self, template):
        src = None
        if self.csv_path.exists():
            src = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
        elif template and Path(template).exists():           # impor isian Excel yang sudah ada
            src = pd.read_excel(template, sheet_name="Label", dtype=str, keep_default_na=False)
        if src is None:
            return
        for _, r in src.iterrows():
            if r["id"] in self.rows:
                for k in EDIT:
                    if k in src.columns:
                        self.rows[r["id"]][k] = str(r[k])

    def get(self, i):
        return self.rows[i]

    def set(self, i, **kw):
        for k, v in kw.items():
            if k in EDIT:
                self.rows[i][k] = v.strip() if isinstance(v, str) else v

    def is_done(self, i):
        r = self.rows[i]
        return bool(r["teks_mentah"].strip()) or r["dapat_dibaca"] == "tidak"

    def n_done(self):
        return sum(self.is_done(i) for i in self.ids)

    def next_undone(self, start):
        n = len(self.ids)
        for step in range(1, n + 1):
            j = (start + step) % n
            if not self.is_done(self.ids[j]):
                return j
        return None

    def first_undone(self):
        for j, i in enumerate(self.ids):
            if not self.is_done(i):
                return j
        return 0

    def save(self):
        """Tulis ke berkas sementara lalu ganti -> berkas tidak rusak bila program mati di tengah jalan."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{"id": i, **self.rows[i]} for i in self.ids], columns=FIELDS)
        fd, tmp = tempfile.mkstemp(dir=self.csv_path.parent, suffix=".tmp")
        os.close(fd)
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, self.csv_path)

    @staticmethod
    def extract_exp(text):
        """Ambil bagian setelah kata EXP TERAKHIR: 'LOT 1 FAB 08/2018 EXP: 08/2021' -> '08/2021'."""
        m = re.search(r".*EXP\W*(.*)$", text, flags=re.I | re.S)
        return m.group(1).strip() if m else ""

    @staticmethod
    def suggest_readable(teks):
        if not teks.strip():
            return ""
        return "sebagian" if "?" in teks else "ya"


# ----------------------------------------------------------------------------- GUI
class App:
    CW, CH = 960, 240          # ukuran kanvas gambar (piksel)

    def __init__(self, root, store):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root, self.store = root, store
        self.idx = store.first_undone()
        self.flipped = False
        self._photo = None

        root.title("Pelabelan gold set")
        root.geometry("1000x680")
        mono = ("Consolas", 14)

        self.head = ttk.Label(root, font=("Arial", 11, "bold"))
        self.head.pack(anchor="w", padx=12, pady=(10, 0))
        self.banner = ttk.Label(root, foreground="#b00020", font=("Arial", 10, "bold"))
        self.banner.pack(anchor="w", padx=12)
        self.canvas = tk.Canvas(root, width=self.CW, height=self.CH, bg="#e8e8e8", highlightthickness=1)
        self.canvas.pack(padx=12, pady=6)

        f = ttk.Frame(root)
        f.pack(fill="x", padx=12)
        self.v = {k: tk.StringVar() for k in EDIT}

        ttk.Label(f, text="orientasi (crop seperti tersimpan):").grid(row=0, column=0, sticky="w", pady=4)
        rf = ttk.Frame(f)
        rf.grid(row=0, column=1, sticky="w")
        for n, (val, key) in enumerate([("tegak", "F1"), ("terbalik", "F2"), ("lainnya", "F3")]):
            ttk.Radiobutton(rf, text=f"{val} ({key})", value=val, variable=self.v["orientasi"]).pack(side="left", padx=6)
        ttk.Button(rf, text="kosongkan", command=lambda: self.v["orientasi"].set("")).pack(side="left", padx=6)
        ttk.Button(rf, text="Putar tampilan 180° (Ctrl+R)", command=self.toggle_flip).pack(side="left", padx=18)

        ttk.Label(f, text="teks_mentah:").grid(row=1, column=0, sticky="w", pady=4)
        self.e_teks = ttk.Entry(f, textvariable=self.v["teks_mentah"], font=mono, width=62)
        self.e_teks.grid(row=1, column=1, sticky="w")

        ttk.Label(f, text="exp_saja:").grid(row=2, column=0, sticky="w", pady=4)
        ef = ttk.Frame(f)
        ef.grid(row=2, column=1, sticky="w")
        self.e_exp = ttk.Entry(ef, textvariable=self.v["exp_saja"], font=mono, width=24)
        self.e_exp.pack(side="left")
        ttk.Button(ef, text="isi dari teks_mentah (Ctrl+E)", command=self.auto_exp).pack(side="left", padx=8)

        ttk.Label(f, text="dapat_dibaca:").grid(row=3, column=0, sticky="w", pady=4)
        df = ttk.Frame(f)
        df.grid(row=3, column=1, sticky="w")
        for val in ("ya", "sebagian", "tidak"):
            ttk.Radiobutton(df, text=val, value=val, variable=self.v["dapat_dibaca"]).pack(side="left", padx=6)
        ttk.Label(df, text="(kosong = diisi otomatis: '?' -> sebagian, selain itu ya)").pack(side="left", padx=10)

        ttk.Label(f, text="catatan:").grid(row=4, column=0, sticky="w", pady=4)
        self.e_cat = ttk.Entry(f, textvariable=self.v["catatan"], font=("Arial", 11), width=70)
        self.e_cat.grid(row=4, column=1, sticky="w")

        nav = ttk.Frame(root)
        nav.pack(fill="x", padx=12, pady=10)
        ttk.Button(nav, text="◀ Sebelumnya (Alt+←)", command=lambda: self.go(self.idx - 1)).pack(side="left")
        ttk.Button(nav, text="Berikutnya ▶ (Alt+→)", command=lambda: self.go(self.idx + 1)).pack(side="left", padx=6)
        ttk.Button(nav, text="Simpan & ke yang belum terisi (Ctrl+Enter)", command=self.save_next).pack(side="left", padx=16)
        ttk.Button(nav, text="Simpan (Ctrl+S)", command=self.save_only).pack(side="left")
        self.jump = ttk.Entry(nav, width=5)
        self.jump.pack(side="right")
        ttk.Button(nav, text="Lompat ke #", command=self.do_jump).pack(side="right", padx=4)

        self.status = ttk.Label(root, text="", foreground="#1b5e20")
        self.status.pack(anchor="w", padx=12)

        root.bind("<F1>", lambda e: self.v["orientasi"].set("tegak"))
        root.bind("<F2>", lambda e: self.v["orientasi"].set("terbalik"))
        root.bind("<F3>", lambda e: self.v["orientasi"].set("lainnya"))
        root.bind("<Control-r>", lambda e: self.toggle_flip())
        root.bind("<Control-e>", lambda e: self.auto_exp())
        root.bind("<Control-s>", lambda e: self.save_only())
        root.bind("<Control-Return>", lambda e: self.save_next())
        root.bind("<Alt-Left>", lambda e: self.go(self.idx - 1))
        root.bind("<Alt-Right>", lambda e: self.go(self.idx + 1))
        self.e_teks.bind("<Return>", lambda e: self.e_exp.focus_set())
        self.e_exp.bind("<Return>", lambda e: self.save_next())
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.show()

    # --- tampilan
    def current_id(self):
        return self.store.ids[self.idx]

    def show(self):
        from PIL import Image
        i = self.current_id()
        r = self.store.get(i)
        for k in EDIT:
            self.v[k].set(r[k])
        self.flipped = False
        im = Image.open(self.store.crops_dir / i).convert("RGB")
        self._draw(im)
        self._im = im
        n = len(self.store.ids)
        self.head.config(text=f"Crop {self.idx + 1} / {n}    |    terisi {self.store.n_done()} / {n}    |    {i}")
        self.banner.config(text="")
        self.status.config(text="")
        self.e_teks.focus_set()

    def _draw(self, im):
        from PIL import Image, ImageTk
        if self.flipped:
            im = im.rotate(180)
        s = min(self.CW / im.width, self.CH / im.height, 6.0)
        im2 = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(im2)
        self.canvas.delete("all")
        self.canvas.create_image(self.CW // 2, self.CH // 2, image=self._photo)

    def toggle_flip(self):
        self.flipped = not self.flipped
        self._draw(self._im)
        self.banner.config(text="TAMPILAN DIPUTAR 180° - 'orientasi' tetap mengacu ke crop ASLI (sebelum diputar)" if self.flipped else "")

    # --- data
    def commit(self):
        vals = {k: self.v[k].get() for k in EDIT}
        if not vals["dapat_dibaca"]:
            vals["dapat_dibaca"] = LabelStore.suggest_readable(vals["teks_mentah"])
        self.store.set(self.current_id(), **vals)
        self.store.save()

    def auto_exp(self):
        exp = LabelStore.extract_exp(self.v["teks_mentah"].get())
        if exp:
            self.v["exp_saja"].set(exp)
            self.msg("exp_saja diisi dari teks_mentah - periksa kembali.")
        else:
            self.msg("Kata EXP tidak ditemukan di teks_mentah; ketik exp_saja manual.", warn=True)

    def msg(self, text, warn=False):
        self.status.config(text=text, foreground="#b00020" if warn else "#1b5e20")

    def save_only(self):
        self.commit()
        self.head_refresh()
        self.msg("Tersimpan.")

    def head_refresh(self):
        n = len(self.store.ids)
        self.head.config(text=f"Crop {self.idx + 1} / {n}    |    terisi {self.store.n_done()} / {n}    |    {self.current_id()}")

    def go(self, j):
        self.commit()
        self.idx = j % len(self.store.ids)
        self.show()

    def save_next(self):
        self.commit()
        nxt = self.store.next_undone(self.idx)
        if nxt is None:
            self.head_refresh()
            self.msg(f"SEMUA {len(self.store.ids)} gambar sudah terisi. Jalankan finalize_gold.py.")
            return
        self.idx = nxt
        self.show()

    def do_jump(self):
        try:
            j = int(self.jump.get()) - 1
            assert 0 <= j < len(self.store.ids)
        except Exception:
            self.msg("Nomor tidak valid.", warn=True)
            return
        self.go(j)

    def on_close(self):
        self.commit()
        self.root.destroy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", default="gold/crops")
    ap.add_argument("--labels", default="gold/gold_labels.csv")
    ap.add_argument("--template", default="gold/gold_template.xlsx")
    if argv is None and "ipykernel" in sys.modules:
        sys.exit("Jalankan dari terminal:  python gold_label_gui.py  (jendela GUI tidak cocok di sel notebook)")
    a = ap.parse_args(argv)

    import tkinter as tk
    store = LabelStore(a.crops_dir, a.labels, a.template)
    root = tk.Tk()
    App(root, store)
    root.mainloop()


if __name__ == "__main__":
    main()

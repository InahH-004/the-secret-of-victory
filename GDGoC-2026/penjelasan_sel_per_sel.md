# GDGoC UGM Academic Hacker 2026/2027 — Prediksi Diagnosis Pasien (4 Kelas)

**Tugas:** memprediksi `target_class` ∈ {`viral_sinusitis`, `viral_pharyngitis`, `acute_bronchitis`, `strep_pharyngitis`}
dari data rekam medis relasional (multi-CSV). **Metrik:** Macro-F1 (rata-rata F1 keempat kelas dengan bobot sama).

## Cara membaca notebook ini
Setiap **sel kode** diikuti sel **"Alasan"** yang menjelaskan *apa* yang dilakukan, *mengapa* dipilih, dan *temuan* yang relevan untuk write-up.
Angka di sel Alasan yang bertanda **[run referensi]** berasal dari satu kali eksekusi verifikasi dengan model cadangan `HistGradientBoosting`
(LightGBM tidak terpasang di lingkungan verifikasi). Angka model di komputermu bisa sedikit berbeda, terutama bila memakai LightGBM — **pakai angka dari run-mu sendiri di write-up.**
Angka tentang *data* (jumlah baris, format, dsb.) tidak bergantung pada model.

## Peta notebook
1. Setup & memuat data → 2. EDA tahap 1 (audit kualitas data & kebocoran) → 3. Preprocessing / feature engineering →
4. EDA tahap 2 (hubungan fitur–kelas pada data bersih) → 5. Modeling → 6. Evaluasi → 7. Submission → 8. Catatan write-up

# 1. Setup & Memuat Data

### Sel 1 · Impor, konfigurasi, dan pencarian folder data

```python
import re, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:                      # tanpa LightGBM -> HistGradientBoosting (lihat bagian 5)
    HAS_LGB = False

ZIP_NAMA = "gdgoc-ugm-academic-hacker-2026.zip"

def cari_folder():
    """Folder data = folder yang berisi train.csv DAN index_encounters.csv. Jika hanya ada zip, ekstrak ke ./data."""
    kaggle = sorted(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []
    kandidat = [Path("../data"), Path("data"), Path("."), Path("../GDGoC-2026"),
                Path("../gdgoc-ugm-academic-hacker-2026")] + kaggle             # UBAH: tambah kandidat bila perlu
    for c in kandidat:
        if (c / "train.csv").exists() and (c / "index_encounters.csv").exists():
            return c
    for z in [Path(ZIP_NAMA), Path("..") / ZIP_NAMA, Path("../data") / ZIP_NAMA, Path("data") / ZIP_NAMA]:
        if z.exists():
            zipfile.ZipFile(z).extractall("data"); return Path("data")
    raise FileNotFoundError("train.csv + index_encounters.csv / zip tidak ditemukan; atur DATA manual")

def sub(nama):
    """Subfolder triase/riwayat. Zip: 'Triase/Triase'; juga menerima versi diratakan."""
    for c in [DATA / nama / nama, DATA / nama, DATA / nama.lower().replace(" ", "-")]:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"subfolder '{nama}' tidak ditemukan di {DATA}")

DATA = cari_folder()
TRIASE, RIWAYAT_DIR = sub("Triase"), sub("Riwayat Medis")
OUT = Path("../outputs"); OUT.mkdir(parents=True, exist_ok=True)

KEY, PASIEN = "encounter_id", "patient_id"          # KEY: kunci sampel -> tabel per-kunjungan | PASIEN: kunci entitas -> tabel statis/riwayat
TARGET, ID = "target_class", "sample_id"
SEED, N_FOLD, N_REPEAT = 42, 5, 3
RUN_ABLATION = False                                # True -> jalankan ablasi di bagian 6 (±6 menit)
RIWAYAT = {                                         # tabel riwayat -> nama kolom tanggalnya
    "conditions": "recorded_at", "medications": "recorded_at", "procedures": "recorded_at",
    "immunizations": "recorded_at", "allergies": "recorded_at", "observations": "observed_at",
    "previous_encounter": "history_start",
}
print("data:", DATA.resolve(), "| LightGBM:", HAS_LGB)
```

**Alasan — Sel 1**

**Apa:** mengimpor pustaka, mendefinisikan konstanta global, dan mencari folder data secara otomatis.

**Mengapa:**
- Semua konstanta (`KEY`, `PASIEN`, `SEED`, …) dikumpulkan di satu tempat agar tiap sel berikutnya tidak memuat "angka ajaib" dan mudah diubah untuk soal lain (baris bertanda `UBAH`).
- `cari_folder()` mensyaratkan **`train.csv` *dan* `index_encounters.csv`** ada di folder yang sama. Ini mencegah salah mengambil `train.csv` milik kompetisi lain (di folder kerja yang sama pernah ada `train.csv` Gemastik — file berita 229 MB — yang tidak ada hubungannya dengan tugas ini).
- `KEY` (encounter) dan `PASIEN` (patient) dibedakan sengaja: tabel triase terhubung lewat *kunjungan*, sedangkan tabel pasien & riwayat lewat *pasien*. Salah memilih kunci di sini adalah sumber bug join yang paling sering.
- `N_FOLD=5` dan `N_REPEAT=3`: kelas `strep_pharyngitis` hanya 195 sampel (≈39 per fold). Dengan hanya satu kali CV, estimasi skor kelas ini sangat berisik, jadi CV diulang 3× dengan pengacakan berbeda.
- LightGBM dibuat opsional (`HAS_LGB`) supaya notebook tetap jalan di lingkungan tanpa LightGBM; modelnya otomatis diganti `HistGradientBoostingClassifier` (lihat bagian 5).

**Untuk write-up:** sebutkan bahwa seluruh pipeline dapat dijalankan ulang dari nol (path dicari otomatis, seed tetap).

### Sel 2 · Memuat tabel utama

```python
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
waktu = pd.read_csv(DATA / "index_encounters.csv")
waktu["prediction_time"] = pd.to_datetime(waktu["prediction_time"], utc=True)
patients = pd.read_csv(DATA / "patients.csv")

assert TARGET in train.columns and TARGET not in test.columns, "train.csv/test.csv ini bukan milik kompetisi"
print(f"train {train.shape} | test {test.shape} | index_encounters {waktu.shape} | patients {patients.shape}")
print("kolom train:", list(train.columns))
print("kolom test :", list(test.columns))
```

**Alasan — Sel 2**

**Apa:** memuat empat tabel inti dan memastikan strukturnya sesuai harapan.

**Mengapa:**
- `train`/`test` adalah **tabel utama**: satu baris = satu sampel yang diprediksi. Semua fitur nantinya ditempel ke tabel ini.
- `index_encounters` menyimpan `prediction_time` (kapan prediksi dibuat). Kolom ini kunci anti-kebocoran: hanya informasi **sebelum** waktu itu yang boleh dipakai sebagai fitur.
- `assert` pada kolom target memastikan file yang terbaca memang milik kompetisi ini (test tidak boleh punya target).

**Temuan:** train 3.600 baris, test 1.400 baris, tiap sampel = 1 pasien = 1 kunjungan.

### Sel 3 · Fungsi `norm()` dan `angka()` (pembersih teks & angka)

```python
def norm(s):
    """teks → huruf kecil, simbol jadi spasi (menyatukan 'Cough / productive' dan 'cough productive')."""
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()

def angka(s):
    """'USD 1,234' / '$1,234' / '115.6k' / '28,02' → float. Koma = ribuan kalau diikuti 3 digit, selain itu desimal."""
    s = s.astype(str).str.lower().str.replace(r"usd|\$|\s", "", regex=True)
    kali = np.where(s.str.endswith("k"), 1000, 1)
    s = s.str.rstrip("k").str.replace(r",(?=\d{3}(\D|$))", "", regex=True).str.replace(",", ".")
    return pd.to_numeric(s, errors="coerce") * kali

# uji kecil: fungsi pembersih harus benar SEBELUM dipakai di seluruh data
assert angka(pd.Series(["USD 1,234", "$34,487", "115.6k", "28,02", "0,9", "12679.0", None])).round(2).tolist()[:6] == \
    [1234, 34487, 115600, 28.02, 0.9, 12679]
```

**Alasan — Sel 3**

**Apa:** dua fungsi pembersih yang dipakai berulang: `norm()` untuk teks, `angka()` untuk angka yang formatnya kacau.

**Mengapa:** data ini sengaja "dikotori" (noise injection). Tanpa pembersihan:
- `norm()`: nama gejala yang sama ditulis berbeda (`Cough / productive` vs `cough productive`) dihitung sebagai gejala berbeda.
- `angka()`: nilai seperti `USD 1,234`, `$34,487`, `115.6k`, dan `28,02` (koma desimal) tidak bisa dikonversi ke angka. Aturan komanya penting: koma **diikuti tepat 3 digit** dianggap pemisah ribuan (`34,487`), selain itu dianggap koma desimal (`28,02`). Notasi `k` dikali 1000.
- `assert` di akhir berfungsi sebagai *unit test*: kalau seseorang mengubah regex dan merusak parsing, notebook langsung berhenti, bukan diam-diam menghasilkan fitur salah.

**Untuk write-up:** fungsi ini bentuk nyata dari "penanganan data kotor"; kasus uji (`USD 1,234`, `115.6k`, `28,02`) bisa dikutip sebagai contoh.

# 2. EDA Tahap 1 — Audit Kualitas Data dan Kebocoran

## 2.1 Gambaran umum

### Sel 4 · Distribusi kelas dan baseline

```python
vc = train[TARGET].value_counts()
print(pd.DataFrame({"jumlah": vc, "persen": (vc / len(train) * 100).round(1)}))
print("rasio kelas terbanyak : tersedikit =", round(vc.max() / vc.min(), 1))

tebak_mayoritas = np.full(len(train), vc.idxmax())
print("macro-F1 jika selalu menebak kelas terbanyak:",
      round(f1_score(train[TARGET], tebak_mayoritas, average="macro"), 3))

ax = vc.plot.bar(color="#4C78A8", figsize=(6, 3.2), rot=15)
ax.set_title("Distribusi kelas (train)"); ax.set_ylabel("jumlah"); ax.set_ylim(0, vc.max() * 1.12)
for i, v in enumerate(vc.values):
    ax.text(i, v + 15, f"{v}", ha="center")
plt.tight_layout(); plt.show()
```

**Alasan — Sel 4**

**Apa:** menghitung proporsi tiap kelas dan skor *baseline* "selalu menebak kelas terbanyak".

**Mengapa:** ini langkah pertama EDA karena menentukan strategi seluruh pipeline.

**Temuan:**
- Kelas **tidak seimbang**: `viral_sinusitis` 41,1%, `viral_pharyngitis` 27,4%, `acute_bronchitis` 26,1%, dan `strep_pharyngitis` hanya **5,4%** (195 sampel); rasio terbesar:terkecil 7,6.
- Menebak kelas terbanyak terus-menerus memberi **macro-F1 hanya 0,146** (akurasinya 41%, tetapi tiga kelas lain F1-nya nol). Inilah alasan metrik kompetisi adalah macro-F1: kelas kecil bobotnya sama dengan kelas besar.

**Implikasi desain:** (1) validasi harus **stratified** agar proporsi kelas kecil terjaga di tiap fold; (2) keputusan akhir perlu **penyesuaian bobot kelas** (bagian 6.2), sebab argmax probabilitas cenderung mengabaikan kelas 5%.

### Sel 5 · Inventaris seluruh file

```python
inventaris = []
for folder in (DATA, TRIASE, RIWAYAT_DIR):
    for f in sorted(folder.glob("*.csv")):
        d = pd.read_csv(f)
        inventaris.append({
            "file": str(f.relative_to(DATA)),
            "baris": len(d),
            "n_encounter": d[KEY].nunique() if KEY in d.columns else None,
            "n_pasien": d[PASIEN].nunique() if PASIEN in d.columns else None,
            "kosong_maks_%": round(d.isna().mean().max() * 100, 1),
            "kolom": ", ".join(d.columns),
        })
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 220)
inventaris = pd.DataFrame(inventaris)
inventaris
```

**Alasan — Sel 5**

**Apa:** satu tabel ringkas berisi jumlah baris, jumlah kunjungan/pasien unik, persentase kosong tertinggi, dan daftar kolom untuk **setiap** CSV.

**Mengapa:** pada data multi-tabel, hal pertama yang harus dipahami adalah *granularitas* tiap tabel (satu baris mewakili apa?). Dari sini ditentukan cara menggabungkannya ke tabel utama:
- **1 baris per pasien** (`patients`): cukup `merge`.
- **Banyak baris per kunjungan** (`triage_vitals` 30.000 baris, `triage_symptoms` 19.065 baris): *tabel long* → harus dipivot/diringkas menjadi satu baris per kunjungan.
- **Banyak baris per pasien dengan tanggal** (tujuh tabel riwayat): harus **difilter waktu** lebih dulu, baru diringkas.

**Temuan:** `patients` punya kolom dengan kosong hingga 42,3% (kota); `triage_symptom_severity` hanya mencakup 4.961 dari 5.000 kunjungan; `allergies` hanya 2 baris (nyaris tidak berguna).

## 2.2 Audit tabel pasien

### Sel 6 · Audit umur

```python
AUDIT = []                                           # tabel temuan kualitas data → bahan write-up

patients.info()
umur = patients["age_at_visit"]
n_999 = int((umur == 999).sum())
n_luar = int((umur.notna() & (umur != 999) & ~umur.between(0, 120)).sum())   # nilai tak masuk akal selain 999
print(f"umur: kode 999 = {n_999} | di luar 0-120 selain 999 = {n_luar} | kosong = {umur.isna().sum()}")
print(umur[umur.between(0, 120)].describe().round(1).to_string())
AUDIT.append(("umur kode 999 / di luar 0-120", n_999 + n_luar, "diganti NaN"))
```

**Alasan — Sel 6**

**Apa:** memeriksa tipe data, nilai kosong, dan nilai mustahil pada umur.

**Mengapa:** umur adalah fitur klinis penting, jadi nilai yang salah harus ditemukan sebelum modeling.

**Temuan:** ada **19 baris berkode `999`** (kode "kosong" yang menyamar sebagai angka) dan 60 baris kosong sungguhan; tidak ada nilai mustahil lain. Setelah `999` dibuang, sebaran umur wajar (rata-rata 38,8; rentang 0–90).

**Keputusan:** `999` diganti `NaN` (bukan dihapus barisnya). **Baris tidak boleh dibuang**, terutama di test, karena setiap sampel test harus tetap mendapat prediksi. Model berbasis pohon menangani `NaN` secara native. Jika `999` dibiarkan, model menganggap ada pasien berusia 999 tahun dan memotong pohon di sekitar angka itu.

### Sel 7 · Audit income

```python
inc = patients["income_at_visit"].astype("string").str.strip()
jenis = pd.Series("lainnya", index=inc.index)
POLA = {
    "angka polos (12679.0)": r"\d+(\.\d+)?",
    "singkatan k (115.6k)": r"(?i)[\d.,]+k",
    "dolar ($34,487)": r"\$[\d,.]+",
    "USD (USD 1,234)": r"(?i)usd\s*[\d,.]+",
}
for nama, pola in POLA.items():
    jenis[inc.str.fullmatch(pola).fillna(False).astype(bool)] = nama
jenis[inc.isna()] = "kosong"
print("format income:")
print(pd.DataFrame({"jumlah": jenis.value_counts(), "contoh": inc.groupby(jenis).first()}).to_string())
n_campur = int((~jenis.isin(["kosong", "angka polos (12679.0)"])).sum())
AUDIT.append(("income berformat non-polos (USD/$/k/lainnya)", n_campur, "angka(): buang simbol, 'k' x1000, koma ribuan"))
AUDIT.append(("income kosong", int((jenis == "kosong").sum()), "dibiarkan NaN"))
```

**Alasan — Sel 7**

**Apa:** mengklasifikasikan setiap nilai `income_at_visit` ke dalam pola penulisannya dengan regex.

**Mengapa:** kolom yang bertipe teks hampir pasti campur format. Daripada menebak, formatnya dihitung satu per satu.

**Temuan:** hanya **3.855** nilai yang berupa angka polos; **690 nilai** memakai format lain — `$51,777` (232), `36,523` tanpa simbol (225), `0.0k`/`115.6k` (162), `USD 122,462` (71) — dan 455 kosong. Kalau dipaksa `pd.to_numeric`, 690 nilai itu menjadi `NaN` dan informasi pendapatan hilang diam-diam.

**Keputusan:** semua format diselesaikan oleh `angka()` (buang simbol, `k`×1000, koma ribuan). Yang benar-benar kosong dibiarkan `NaN`.

### Sel 8 · Audit status pernikahan dan kolom kategori

```python
mar = patients["marital_at_visit"].dropna().astype(str)
print("marital (mentah):", mar.value_counts().to_dict())
kata_penuh = mar[mar.str.len() > 1]                          # "Married", "Single", ... (bukan kode 1 huruf)
bentrok = kata_penuh.groupby(kata_penuh.str[0]).nunique()    # huruf awal yang dipakai >1 kata berbeda
print("huruf awal yang bentrok (>1 status berbeda):", bentrok[bentrok > 1].to_dict() or "tidak ada")

KAT = ["gender", "race", "ethnicity", "county"]
print("\nkardinalitas:", patients[KAT].nunique().to_dict())
for c in KAT:
    print(f"{c:10s}", patients[c].value_counts(dropna=False).head(5).to_dict())
```

**Alasan — Sel 8**

**Apa:** memeriksa penulisan status pernikahan dan kardinalitas kolom kategori.

**Mengapa:** kode status bercampur antara huruf (`M`, `S`, `D`, `W`) dan kata penuh (`Married`, `Single`, `Divorced`, `Widowed`). Cara paling sederhana menyatukannya adalah mengambil huruf pertama. Cara ini hanya aman jika **tidak ada dua status berbeda yang berawalan sama** — itu yang dicek oleh `bentrok`.

**Temuan:** tidak ada huruf awal yang bentrok, jadi `str[0]` aman. Kolom `county` berkardinalitas tinggi (58), `race` 6, `ethnicity` 2, `gender` 2.

**Catatan:** kardinalitas `county` yang tinggi berisiko overfit bila dipakai sebagai kategori; model berbasis pohon cukup tahan, tetapi kontribusi kolom demografi sebaiknya diuji (lihat ablasi di bagian 6.4 untuk vital dan riwayat).

## 2.3 Audit triase

### Sel 9 · Audit tanda vital

```python
v = pd.read_csv(TRIASE / "triage_vitals.csv")
n_enc = len(train) + len(test)                                  # jumlah kunjungan seluruhnya
mentah = v["value"].astype(str)
n_koma_desimal = int(mentah.str.contains(r",\d{1,2}$").sum())   # '28,02' -> koma = desimal
n_koma_ambigu = int(mentah.str.contains(r",\d{3}$").sum())      # ',ddd' -> ribuan ATAU desimal 3 digit?
v["nilai"] = angka(v["value"])
n_gagal = int(v["nilai"].isna().sum() - v["value"].isna().sum())  # nilai ada, tetapi tidak terbaca angka
print(f"baris {len(v)} | kunjungan {v[KEY].nunique()} dari {n_enc} | koma desimal {n_koma_desimal} | koma ambigu {n_koma_ambigu} | gagal parse {n_gagal}")

# satuan beda skala: median satuan itu vs median vital itu (mis. 10,5 kPa vs 79 mmHg)
med_vital = v.groupby("vital_name")["nilai"].transform("median")
med_unit = v.groupby(["vital_name", "unit"])["nilai"].transform("median")
rasio = med_unit / med_vital
beda_skala = rasio.notna() & ~rasio.between(0.5, 2)
dobel = int(v.duplicated([KEY, "vital_name"], keep=False).sum())   # baris yang berbagi (kunjungan, vital) yang sama
print(f"Jumlah data dengan perbedaan skala: {beda_skala.sum()} | baris terindikasi duplikat: {dobel}")

ringkasan = (v.assign(median_unit=med_unit, median_vital=med_vital, rasio=rasio, beda_skala=beda_skala)
               .groupby(["vital_name", "unit"])
               .agg(jumlah_data=("nilai", "count"), median_unit=("median_unit", "first"),
                    median_vital=("median_vital", "first"), rasio=("rasio", "first"), beda_skala=("beda_skala", "first"))
               .reset_index())
print(ringkasan.to_string(index=False))
AUDIT.append(("vital: koma desimal ('28,02')", n_koma_desimal, "angka(): koma -> titik"))
AUDIT.append(("vital: satuan beda skala (kPa, fraction)", int(beda_skala.sum()), "KONVERSI: kPa x7.50062, fraction x100"))
```

**Alasan — Sel 9**

**Apa:** tiga pemeriksaan pada tabel vital: (1) format angka (koma desimal), (2) satuan yang berbeda skala, (3) duplikasi pengukuran.

**Mengapa & temuan:**
- **Koma desimal:** 112 nilai bertulis seperti `28,02`. Tidak ada koma "ambigu" (`,ddd`), sehingga aturan di `angka()` tidak salah tafsir.
- **Satuan beda skala:** tekanan darah ada yang dalam **kPa** (sistolik median 15,9 vs 118 mmHg; rasio 0,13) dan SpO2 ada yang berupa **fraction** (0,94 vs 94%; rasio 0,01). Total **1.049 baris** berbeda skala. Rasio median per satuan terhadap median vital dipakai sebagai detektor: bila rasionya di luar 0,5–2, satuan itu pasti beda skala. Satuan `/min` vs `beats/min` (dan `/min` vs `breaths/min`) hanya beda nama, skalanya sama, jadi tidak perlu konversi.
- **Duplikasi:** 0 baris ganda per (kunjungan, vital), jadi memakai rata-rata saat pivot tidak mengubah apa pun.

**Keputusan:** kPa × 7,50062 → mmHg dan fraction × 100 → persen (`KONVERSI`).

**Catatan jujur:** *outlier* lain (mis. denyut jantung 990, yang tampak seperti salah ketik ×10) tidak ditangani di pipeline ini. Alasannya ada di bagian 6.4: ablasi menunjukkan vital tidak menambah sinyal, sehingga membersihkannya lebih jauh tidak membantu.

### Sel 10 · Audit gejala dan severity

```python
g = pd.read_csv(TRIASE / "triage_symptoms.csv")
g["nama"] = norm(g["symptom_concept"])              # samakan huruf besar/kecil & simbol
n_mentah, n_norm = g["symptom_concept"].nunique(), g["nama"].nunique()
per_kunjungan = g.groupby(KEY).size()               # berapa baris gejala tiap kunjungan
n_tanpa = n_enc - per_kunjungan.size                # kunjungan tanpa satu pun baris gejala
print(f"nama gejala: mentah {n_mentah} -> setelah norm() {n_norm} | baris {len(g)} | kunjungan tanpa gejala {n_tanpa}")
print("gejala per kunjungan:", per_kunjungan.describe().round(2).to_dict())
print("\n40 nama gejala teratas (setelah norm):")
print(g["nama"].value_counts().head(40).to_string())

sev = pd.read_csv(TRIASE / "triage_symptom_severity.csv")
print("\nseverity_score:", sev["severity_score"].describe().round(2).to_dict())
print(f"baris severity {len(sev)} vs baris gejala {len(g)} | kunjungan dgn severity {sev[KEY].nunique()}")
AUDIT.append(("variasi penulisan gejala (mentah - setelah norm)", n_mentah - n_norm, "norm() lalu kamus sinonim"))
```

**Alasan — Sel 10**

**Apa:** memeriksa nama gejala, banyaknya gejala per kunjungan, dan tabel severity.

**Mengapa & temuan:**
- **Variasi penulisan:** 118 nama mentah menyusut menjadi **70** setelah `norm()` (48 duplikat hanya beda huruf besar/simbol).
- **Sinonim:** dari 70 nama itu, sebagian besar ternyata *parafrase* dari beberapa konsep yang sama (`throat pain`, `sore throat`, `painful swallowing`, `throat discomfort`, `throat symptoms` semuanya "nyeri tenggorokan"). `norm()` saja tidak cukup; dibutuhkan kamus sinonim (bagian 3.1).
- **Cakupan:** semua 5.000 kunjungan punya ≥1 gejala (rata-rata 3,81; maksimum 12), jadi tidak ada kunjungan tanpa gejala yang perlu ditangani khusus.
- **Severity:** skor 0–100, tetapi baris severity (16.594) lebih sedikit daripada baris gejala (19.065), jadi **sebagian gejala tidak punya skor** (mis. kelompok `dyspnea` sama sekali tidak punya baris severity; lihat heatmap bagian 4), dan 765 baris severity yang ada skornya kosong. Itu murni nilai kosong, bukan nol.

**Untuk write-up:** angka 118 → 70 → 8 konsep adalah ringkasan yang baik untuk menjelaskan "penyatuan variasi penulisan gejala".

### Sel 11 · Kebocoran dan kemiripan train–test

```python
semua = pd.concat([train.assign(bagian="train"), test.assign(bagian="test")], ignore_index=True)
semua = semua.merge(waktu[[KEY, "prediction_time"]], on=KEY, how="left")

print("pasien di train DAN test   :", len(set(train[PASIEN]) & set(test[PASIEN])))
print("kunjungan di train DAN test:", len(set(train[KEY]) & set(test[KEY])))
print("rentang waktu prediksi per bagian (train vs test harus mirip):")
print(semua.groupby("bagian")["prediction_time"].agg(["min", "max"]).to_string())

hasil = []
for nama, kol in RIWAYAT.items():                   # tiap tabel riwayat: adakah catatan SETELAH waktu prediksi?
    t = pd.read_csv(RIWAYAT_DIR / f"{nama}.csv")
    n_awal = len(t)
    t["_t"] = pd.to_datetime(t[kol], utc=True, format="mixed")
    t = t.merge(semua[[PASIEN, "prediction_time"]], on=PASIEN, how="inner")   # hanya pasien yang menjadi sampel
    hasil.append({"tabel": nama, "baris": n_awal, "pasien_di_luar_sampel": n_awal - len(t),
                  "tanggal_kosong": int(t["_t"].isna().sum()),
                  "sesudah_waktu_prediksi": int((t["_t"] >= t["prediction_time"]).sum())})
hasil = pd.DataFrame(hasil)
print("\nriwayat vs waktu prediksi:")
print(hasil.to_string(index=False))
AUDIT.append(("riwayat sesudah waktu prediksi (kebocoran waktu)", int(hasil["sesudah_waktu_prediksi"].sum()), "difilter: hanya t < prediction_time"))

pd.DataFrame(AUDIT, columns=["temuan", "jumlah", "penanganan"])     # tabel temuan kualitas data untuk write-up
```

**Alasan — Sel 11**

**Apa:** tiga pemeriksaan kebocoran/ketidakcocokan: (1) apakah pasien/kunjungan yang sama muncul di train dan test, (2) apakah rentang waktu train dan test sepadan, (3) apakah ada catatan riwayat yang bertanggal **setelah** `prediction_time`.

**Mengapa:** ini tiga sumber utama skor validasi yang menyesatkan.
1. Jika pasien yang sama ada di train dan test (atau di fold berbeda), model bisa "menghafal" pasien. **Temuan: 0 irisan**, dan karena 1 pasien = 1 baris, `StratifiedKFold` biasa sudah aman (tidak perlu `GroupKFold`).
2. Jika test berasal dari periode yang berbeda, validasi acak terlalu optimis. **Temuan:** train (Mei 2024–Okt 2026) dan test (Mar 2024–Okt 2026) tumpang-tindih penuh, jadi splitnya acak, bukan temporal → CV acak sah.
3. Jika riwayat memuat kejadian setelah waktu prediksi, model memakai informasi masa depan. **Temuan: 0 baris** di ketujuh tabel.

**Keputusan:** meskipun hasil audit (3) nol, filter `t < prediction_time` **tetap dipasang** di pipeline (bagian 3.5). Alasan: audit adalah foto kondisi data saat ini, sedangkan filter adalah jaminan. Bila data diganti, pipeline tetap benar.

**Untuk write-up:** tabel `AUDIT` di sel ini bisa langsung disalin sebagai "tabel temuan kualitas data dan penanganannya".

# 3. Preprocessing dan Feature Engineering

**Prinsip yang dipakai di seluruh bagian ini:**
1. **Satu baris per sampel.** Semua tabel dilebur ke tabel utama sehingga model menerima satu matriks fitur biasa.
2. **Train dan test diproses bersama** dalam satu tabel (`base`) dengan fungsi yang sama → tidak mungkin ada perbedaan perlakuan.
3. **Label tidak pernah disentuh** saat membuat fitur → tidak ada kebocoran target.
4. **Riwayat difilter waktu** (`< prediction_time`) sebelum diringkas → tidak ada kebocoran masa depan.

## 3.1 Kamus dan fungsi pembantu

### Sel 12 · Kamus sinonim gejala, konversi satuan, daftar kategori

```python
SINONIM = {                                         # kamus: kelompok kanonik -> variasi nama (hasil norm())
    "throat": ["painful swallowing", "sore throat", "throat discomfort", "throat pain", "throat symptoms"],
    "cough": ["cough", "cough productive", "cough symptom", "coughing", "persistent cough"],
    "nasal": ["blocked nose", "nasal complaints", "nasal congestion", "nasal symptoms", "stuffy nose"],
    "fever": ["febrile feeling", "fever", "feverish", "raised temperature", "temperature complaint"],
    "systemic": ["body malaise", "feeling unwell", "general fatigue symptoms", "general malaise", "systemic symptoms"],
    "secretion": ["airway secretions", "mucus discharge", "mucus complaint", "respiratory secretions", "sputum production"],
    "dyspnea": ["breathing discomfort", "difficulty breathing", "dyspnea wheeze", "shortness of breath wheeze", "wheezy feeling"],
    "head": ["head and facial symptoms"],
}
KANON = {s: k for k, daftar in SINONIM.items() for s in daftar}     # dibalik: nama -> kelompok
KONVERSI = {"kPa": 7.50062, "fraction": 100}        # satuan minoritas -> satuan mayoritas (mmHg, persen)
KATEGORI = ["gender", "race", "ethnicity", "marital", "county"]

# validasi kamus: semua nama gejala yang tidak masuk kamus akan jadi "lain" -> lihat isinya
nama_semua = sorted(norm(g["symptom_concept"]).unique())
di_luar = [n for n in nama_semua if n not in KANON]
print(f"{len(nama_semua)} nama gejala | {len(KANON)} masuk kamus | {len(di_luar)} masuk 'lain': {di_luar}")
```

**Alasan — Sel 12**

**Apa:** tiga "kamus" konfigurasi — pengelompokan sinonim gejala, faktor konversi satuan vital, dan daftar kolom kategori.

**Mengapa:**
- **Kamus sinonim** dibuat dari daftar `sorted(norm(gejala).unique())`: lima ungkapan "nyeri tenggorokan" disatukan menjadi `throat`, dan seterusnya untuk `cough`, `nasal`, `fever`, `systemic`, `secretion`, `dyspnea`, `head`. Tujuannya mengurangi 70 nama menjadi 8 konsep klinis sehingga model belajar "ada batuk atau tidak", bukan "ada kata `coughing` atau tidak".
- **Validasi kamus** (baris `di_luar`): nama gejala yang tidak ada di kamus tidak dibuang tetapi dimasukkan ke grup `lain`. Isinya **34 nama** gejala di luar infeksi saluran napas atas (mis. `anxiety`, `joint pain`, `thirst`, `frequent urination`, `blurred vision`, `memory loss`, `sweating`) — tampaknya *distraktor* (di bagian 4 terlihat muncul ±11% di setiap kelas, sehingga tidak membedakan kelas).
- **Konversi satuan:** hanya dua yang diperlukan (kPa → mmHg dan fraction → %), sesuai audit bagian 2.3.

**Validitas kamus (diperiksa terpisah):** untuk tiap varian dalam satu kelompok, distribusi kelasnya mirip dengan anggota kelompoknya (selisih rata-rata ±0,04, setara noise sampel) dan rata-rata severity-nya seragam, jadi pengelompokan manual ini tidak "memaksa" gejala yang berbeda menjadi satu.

**Untuk write-up:** ini contoh *domain knowledge + data audit* yang dipadukan: nama gejala dikelompokkan berdasarkan makna klinis, lalu divalidasi dengan data.

### Sel 13 · Fungsi pembangun fitur per tabel

```python
def fitur_pasien(patients):
    """Tabel 1-ke-1: satu baris per pasien (indeks = PASIEN)."""
    p = patients.copy()
    p["age"] = p["age_at_visit"].where(p["age_at_visit"].between(0, 120))   # 999 & nilai ngawur -> NaN
    p["income"] = angka(p["income_at_visit"])
    p["marital"] = p["marital_at_visit"].str[0]     # aman karena audit 2.2 melaporkan tidak ada huruf awal yang bentrok
    return p[[PASIEN, "age", "income"] + KATEGORI].set_index(PASIEN)


def fitur_vital(v):
    """Tabel long -> lebar: satu baris per kunjungan (indeks = KEY), satu kolom per vital."""
    v = v.copy()
    v["nilai"] = angka(v["value"]) * v["unit"].map(KONVERSI).fillna(1)      # samakan satuan
    tabel = v.pivot_table(index=KEY, columns="vital_name", values="nilai", aggfunc="mean")   # duplikat -> rata-rata
    tabel.columns = "vital_" + norm(pd.Series(tabel.columns)).str.replace(" ", "_")
    return tabel


def fitur_gejala(g, sev):
    """Tabel long -> lebar: jumlah gejala per kelompok (ada_*), severity maks per kelompok (sev_*), ringkasan severity."""
    g = g.copy()
    g["kanon"] = norm(g["symptom_concept"]).map(KANON).fillna("lain")      # di luar kamus -> "lain"
    ada = pd.crosstab(g[KEY], g["kanon"]).add_prefix("ada_")
    s = sev.copy()
    s["kanon"] = norm(s["symptom_concept"]).map(KANON).fillna("lain")
    per_kanon = s.pivot_table(index=KEY, columns="kanon", values="severity_score", aggfunc="max").add_prefix("sev_")
    ringkas = s.groupby(KEY)["severity_score"].agg(sev_mean="mean", sev_max="max", sev_n="count")
    return pd.concat([ada, per_kanon, ringkas], axis=1)


def fitur_teks(g, keluhan, kunci):
    """Teks per kunjungan: nama gejala dalam URUTAN BARIS FILE + ' || ' + teks keluhan utama.
    PENTING: g tidak boleh diurutkan/diacak sebelum fungsi ini (urutan baris = sumber sinyal, lihat bagian 4.3)."""
    urut = norm(g["symptom_concept"]).groupby(g[KEY]).agg(" | ".join)
    return kunci.map(urut).fillna("") + " || " + norm(kunci.map(keluhan).fillna(""))


def fitur_riwayat(base):
    """Tujuh tabel riwayat -> jumlah kejadian (n_*) dan jarak hari ke kejadian terakhir (hari_*), HANYA sebelum prediction_time."""
    out = base[[KEY]].copy()
    for nama, kol in RIWAYAT.items():
        t = pd.read_csv(RIWAYAT_DIR / f"{nama}.csv")
        t["_t"] = pd.to_datetime(t[kol], utc=True, format="mixed")
        t = t.merge(base[[PASIEN, KEY, "prediction_time"]], on=PASIEN)
        sebelum = t["_t"] < t["prediction_time"]
        print(f"{nama:20s} baris {len(t):6d} | sesudah waktu prediksi (dibuang): {(~sebelum).sum()}")
        t = t[sebelum]
        out[f"n_{nama}"] = out[KEY].map(t.groupby(KEY).size()).fillna(0)
        out[f"hari_{nama}"] = out[KEY].map((t["prediction_time"] - t["_t"]).dt.days.groupby(t[KEY]).min())
    return out
```

**Alasan — Sel 13**

**Apa:** lima fungsi, masing-masing mengubah satu jenis tabel menjadi kolom fitur per sampel.

**Mengapa dibuat sebagai fungsi:** (1) logika yang sama dijamin berlaku untuk train dan test; (2) tiap fungsi bisa diuji dan diubah terpisah; (3) write-up lebih mudah: "tabel tipe X ditangani dengan fungsi Y".

**Alasan tiap fungsi:**
- `fitur_pasien` — tabel 1-ke-1: bersihkan umur (`999`→NaN), income (`angka`), dan status pernikahan (huruf pertama).
- `fitur_vital` — tabel *long* (6 vital per kunjungan) → *wide*: `pivot_table` dengan `mean`. Satuan diseragamkan **sebelum** dirata-rata; kalau dibalik, rata-rata mencampur mmHg dan kPa.
- `fitur_gejala` — tiga jenis fitur: **berapa baris gejala** per kelompok (`ada_*`), **severity maksimum** per kelompok (`sev_*`; maks dipilih karena parafrase dalam satu kelompok mewakili satu gejala yang sama), dan ringkasan severity keseluruhan. Gejala tanpa baris severity menghasilkan `NaN`, bukan 0.
- `fitur_teks` — menyusun satu string per kunjungan dari nama gejala **sesuai urutan baris di file**, digandeng teks keluhan utama. Ini bahan untuk TF-IDF (3.6). Urutan baris sengaja dipertahankan; alasan dan konsekuensinya dijelaskan di bagian 4.3.
- `fitur_riwayat` — untuk tiap tabel riwayat: (a) ubah tanggal ke waktu (`format="mixed"` karena ada tanggal berformat berbeda), (b) gabungkan ke `prediction_time` pasien, (c) **buang yang `>= prediction_time`**, (d) hitung jumlah kejadian (`n_*`) dan hari sejak kejadian terakhir (`hari_*`). Bila pasien tak punya riwayat, `n_*` = 0 dan `hari_*` = `NaN` (benar-benar "tidak ada kejadian").

**Mengapa riwayat hanya dihitung/jaraknya, bukan teks diagnosisnya:** pada percobaan sebelumnya TF-IDF atas nama penyakit/obat riwayat justru menurunkan skor CV; ringkasan jumlah dan recency lebih stabil dan terbukti masih bermanfaat (ablasi 6.4).

## 3.2 Master table dan pemeriksaan pemetaan 1-ke-1

### Sel 14 · Master table: gabungkan train + test dan waktu prediksi

```python
base = pd.concat([train, test], ignore_index=True)                  # fitur dibangun sekali untuk train+test
base = base.merge(waktu[[KEY, "prediction_time"]], on=KEY, how="left")

assert base[KEY].is_unique and base["prediction_time"].notna().all()   # tiap kunjungan tepat satu baris & punya waktu
assert base[PASIEN].is_unique                                          # 1 pasien = 1 baris sampel
assert patients[PASIEN].is_unique                                      # tabel pasien: 1 pasien = 1 baris
assert not set(train[PASIEN]) & set(test[PASIEN])                      # pasien tidak boleh ada di train DAN test
cek = waktu.merge(base[[KEY, PASIEN]], on=KEY, suffixes=("_ie", "_base"))
assert (cek[PASIEN + "_ie"] == cek[PASIEN + "_base"]).all()            # encounter -> pasien konsisten di kedua tabel
print("sampel:", len(base), "| train:", base[TARGET].notna().sum(), "| test:", base[TARGET].isna().sum())
```

**Alasan — Sel 14**

**Apa:** membentuk tabel utama (`base`) dari train + test, menempelkan `prediction_time`, lalu menegakkan lima *assertion* tentang struktur kunci.

**Mengapa:**
- Menggabung train dan test **sebelum** feature engineering memastikan kedua bagian diproses identik. Ini aman karena tidak ada langkah yang memakai label: kolom `target_class` di test kosong (`NaN`), dan fitur tidak pernah menghitung apa pun dari label.
- *Assertion* adalah "asuransi" terhadap kesalahan join yang paling umum: baris menggandakan diri (many-to-many), kunci tidak unik, atau encounter menunjuk pasien yang salah. Jika asumsi 1-ke-1 salah, jumlah baris berubah diam-diam dan skor validasi menjadi palsu; dengan `assert`, notebook berhenti seketika.

**Temuan:** 5.000 sampel = 3.600 train + 1.400 test; seluruh asersi lulus.

**Catatan perbaikan:** versi draf memakai variabel `index_enc` yang belum didefinisikan di sel ini; di sini diganti `waktu`.

## 3.3 Membangun fitur per tabel

### Sel 15 · Fitur pasien (tabel 1-ke-1)

```python
feat = base.merge(fitur_pasien(patients), left_on=PASIEN, right_index=True, how="left")
print(feat[["age", "income"]].describe().round(1))
print("NaN:", feat[["age", "income"] + KATEGORI].isna().sum().to_dict())
```

**Alasan — Sel 15**

**Apa:** menempelkan fitur pasien ke tabel utama dengan `merge` kiri (`how="left"`).

**Mengapa `left`:** jumlah dan urutan baris tabel utama tidak boleh berubah. `inner` bisa membuang sampel yang datanya tidak lengkap (termasuk di test!), sedangkan `left` mengisi `NaN`.

**Temuan:** setelah pembersihan, umur punya 79 `NaN` di 5.000 sampel (60 kosong asli + 19 sentinel `999`, sesuai audit) dan income 455 `NaN`; `marital`, `county`, dan `ethnicity` punya `NaN` kategorikal. Rentang income 0–±998 ribu menunjukkan sebaran sangat miring (median 68.833 vs rata-rata 116.146) — model pohon tidak terganggu oleh skala/kemiringan ini, sehingga tidak perlu transformasi log.

### Sel 16 · Fitur vital (tabel long → wide)

```python
vital = fitur_vital(pd.read_csv(TRIASE / "triage_vitals.csv"))
feat = feat.merge(vital, left_on=KEY, right_index=True, how="left")
print(feat[[c for c in feat.columns if c.startswith("vital_")]].median().round(2).to_dict())
print("NaN per vital:", feat[[c for c in feat.columns if c.startswith("vital_")]].isna().sum().to_dict())
```

**Alasan — Sel 16**

**Apa:** memivot enam jenis vital menjadi enam kolom (`vital_bmi`, `vital_heart_rate`, …) dengan satuan seragam.

**Mengapa dicek mediannya:** sanity check hasil konversi. Setelah konversi, median tekanan darah sistolik/diastolik ±118/79 mmHg dan SpO2 ±94% — angka yang wajar secara klinis. Jika konversi salah, mediannya akan tampak ganjil (mis. SpO2 0,94 atau tekanan darah 15).

**Temuan:** `vital_bmi` paling banyak kosong (±38%); vital lain ±6%. Kosongnya acak, proporsinya serupa antar kelas (bukan pola informatif), sehingga dibiarkan `NaN`.

### Sel 17 · Fitur gejala, severity, dan teks

```python
g = pd.read_csv(TRIASE / "triage_symptoms.csv")                    # JANGAN diurutkan/diacak: urutan baris dipakai fitur_teks
sev = pd.read_csv(TRIASE / "triage_symptom_severity.csv")
keluhan = pd.read_csv(TRIASE / "presenting_complaints.csv").set_index(KEY)["presenting_complaints_text"]

feat = feat.merge(fitur_gejala(g, sev), left_on=KEY, right_index=True, how="left")
feat["teks"] = fitur_teks(g, keluhan, feat[KEY])

print(g.assign(kanon=norm(g["symptom_concept"]).map(KANON).fillna("lain"))["kanon"].value_counts().to_dict())
print("contoh teks:", feat["teks"].iloc[0])
```

**Alasan — Sel 17**

**Apa:** membangun tiga kelompok fitur dari gejala: jumlah per konsep (`ada_*`), severity (`sev_*`), dan satu string teks per kunjungan.

**Mengapa:**
- `ada_*` dan `sev_*` adalah representasi **klinis**: gejala apa yang muncul dan seberapa parah. Ablasi (6.4) menunjukkan severity gejala adalah sinyal terkuat dalam data.
- `teks` menyimpan gejala **berurutan** + keluhan utama untuk diubah menjadi TF-IDF di 3.6. Contoh nilai: `"fever | mucus complaint || reports fever"` (gejala berurutan, lalu teks keluhan).

**Peringatan yang harus ada di write-up:** `teks` mempertahankan **urutan baris** file. Bagian 4.3 dan ablasi 6.4 menunjukkan bahwa urutan ini membawa sinyal tambahan yang bukan sinyal medis, melainkan jejak cara data dibangkitkan. File `g` dibaca ulang di sini dan **tidak diurutkan**, supaya urutannya identik dengan file asli.

### Sel 18 · Fitur riwayat (difilter waktu)

```python
feat = feat.merge(fitur_riwayat(feat), on=KEY, how="left")
print(feat[[c for c in feat.columns if c.startswith(("n_", "hari_"))]].describe().loc[["mean", "max"]].round(2).T.to_string())
```

**Alasan — Sel 18**

**Apa:** menjalankan `fitur_riwayat` untuk semua tabel riwayat, dengan filter `waktu < prediction_time` di dalamnya.

**Mengapa:** inilah titik di mana kebocoran waktu paling mungkin terjadi. Setiap baris yang tercetak (`sesudah waktu prediksi (dibuang): 0`) membuktikan bahwa untuk data ini tidak ada catatan masa depan; bila ada, jumlahnya akan terlihat dan dibuang otomatis.

**Mengapa dua kolom per tabel:** `n_*` (seberapa banyak riwayat) dan `hari_*` (seberapa baru) menangkap dua hal berbeda dengan biaya hanya 14 kolom. Tabel `allergies` hanya 2 baris sehingga `hari_allergies` konstan dan dibuang otomatis di 3.6.

**Temuan:** pada ablasi (6.4) menghapus seluruh riwayat menurunkan AUC dari 0,9513 menjadi 0,9474 dan F1 setelah bobot dari 0,7357 menjadi 0,7273 [run referensi]. Sinyalnya **tersebar** di banyak tabel (tidak ada satu tabel pun yang menonjol), tetapi nyata.

## 3.4 Matriks fitur akhir

### Sel 19 · TF-IDF, kolom kategori, pembuangan kolom konstan, pemisahan train/test

```python
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(feat["teks"])
X_teks = pd.DataFrame(tfidf.transform(feat["teks"]).toarray(), columns="tf_" + tfidf.get_feature_names_out())
for c in KATEGORI:
    feat[c] = feat[c].astype("category")
BUANG = [KEY, PASIEN, TARGET, ID, "prediction_time", "teks"]
X = pd.concat([feat.drop(columns=BUANG), X_teks], axis=1)
X.columns = X.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)   # LightGBM menolak simbol di nama kolom
konstan = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]      # mis. hari_allergies (hanya 2 baris), token sangat langka
X = X.drop(columns=konstan); print("kolom konstan dibuang:", len(konstan), konstan[:6])

latih = feat[TARGET].notna().to_numpy()
KELAS = sorted(train[TARGET].unique())
y = feat.loc[latih, TARGET].map({k: i for i, k in enumerate(KELAS)}).to_numpy()
X_tr, X_te = X[latih].reset_index(drop=True), X[~latih].reset_index(drop=True)
print("X_tr", X_tr.shape, "| X_te", X_te.shape, "| kolom TF-IDF:", X_teks.shape[1], "| kelas:", KELAS)
```

**Alasan — Sel 19**

**Apa:** mengubah teks menjadi fitur numerik (TF-IDF), menandai kolom kategori, membuang kolom yang tak berguna, lalu memisahkan kembali train dan test.

**Mengapa:**
- **TF-IDF unigram + bigram** (`ngram_range=(1,2)`): unigram menangkap *gejala apa yang ada*, bigram menangkap *dua kata berurutan*, termasuk gejala-gejala yang berurutan di teks. `min_df=2` membuang token yang hanya muncul sekali (tak bisa digeneralisasi); `sublinear_tf=True` meredam efek kata yang berulang.
- **Kategori sebagai `category`:** model pohon dapat memakainya langsung tanpa one-hot, sehingga tidak menambah ratusan kolom (`county` punya 58 nilai).
- **Nama kolom dibersihkan:** LightGBM menolak karakter seperti spasi dan tanda baca.
- **Kolom konstan dibuang:** tidak punya daya pembeda dan hanya memperlambat model.
- **`KELAS = sorted(...)`** menetapkan urutan kelas tunggal yang dipakai di mana-mana (probabilitas, bobot kelas, laporan), mencegah salah tukar kelas.
- **Pemisahan** memakai mask `latih` (baris yang punya target), bukan pemotongan posisi, sehingga urutan baris tidak berpengaruh.

**Temuan:** matriks akhir **3.600 × 707** untuk train dan **1.400 × 707** untuk test: 46 kolom fitur terstruktur + 661 kolom TF-IDF. Ini ukuran yang sama persis dengan notebook final (diperiksa: matriksnya identik sel demi sel).

**Catatan:** vocabulary TF-IDF di-*fit* pada teks train+test (tanpa label). Ini bersifat transduktif tetapi tidak memakai target, jadi bukan kebocoran label.

### Sel 20 · Pemeriksaan kemiripan train–test pada matriks fitur (adversarial validation)

```python
adv_y = (~latih).astype(int)                     # 1 = baris test
oof_adv = np.zeros(len(X))
for a, b in StratifiedKFold(N_FOLD, shuffle=True, random_state=SEED).split(X, adv_y):
    m = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, max_leaf_nodes=8,
                                       categorical_features="from_dtype", random_state=SEED).fit(X.iloc[a], adv_y[a])
    oof_adv[b] = m.predict_proba(X.iloc[b])[:, 1]
print(f"AUC pembeda train-vs-test: {roc_auc_score(adv_y, oof_adv):.3f}   (0,5 = tidak bisa dibedakan)")
```

**Alasan — Sel 20**

**Apa:** melatih model untuk menebak apakah suatu baris berasal dari train atau test, hanya dari fitur. Jika model bisa menebaknya dengan baik, berarti distribusi test berbeda dari train.

**Mengapa:** skor validasi hanya bermakna untuk test jika test berasal dari distribusi yang sama. Ini pengganti formal dari pemeriksaan "kemiripan" secara visual.

**Temuan:** AUC ≈ **0,494** (0,5 = tidak bisa dibedakan) → **tidak ada pergeseran distribusi** pada 707 fitur. Konsekuensinya: kenaikan/penurunan skor di leaderboard lebih mungkin disebabkan oleh *noise pengambilan sampel* (test hanya 1.400 baris) daripada perbedaan data; lihat bagian 6.3.

# 4. EDA Tahap 2 — Hubungan Fitur dengan Kelas (pada data bersih)

EDA tahap 1 (bagian 2) memeriksa **kualitas** data. Tahap 2 ini memeriksa **sinyal**: fitur mana yang berhubungan dengan kelas.
Sengaja dilakukan *setelah* preprocessing karena perbandingan antar-kelas pada data mentah menyesatkan (mis. rata-rata tekanan darah yang mencampur kPa dan mmHg tidak bermakna).
Semua analisis memakai **train saja** (hanya train yang punya label).

### Sel 21 · Gejala dan severity per kelas (heatmap)

```python
def heat(ax, tabel, judul, fmt="{:.2f}", cmap="Blues"):
    """Heatmap sederhana dengan angka di tiap sel (tanpa seaborn)."""
    im = ax.imshow(tabel.values.astype(float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(tabel.shape[1])); ax.set_xticklabels([c.replace("_", "\n") for c in tabel.columns], fontsize=8)
    ax.set_yticks(range(tabel.shape[0])); ax.set_yticklabels(tabel.index, fontsize=8)
    batas = np.nanmax(tabel.values.astype(float)) / 2
    for i in range(tabel.shape[0]):
        for j in range(tabel.shape[1]):
            nilai = tabel.values[i, j]
            if pd.notna(nilai):
                ax.text(j, i, fmt.format(nilai), ha="center", va="center", fontsize=8, color="white" if nilai > batas else "black")
    ax.set_title(judul, fontsize=10)

tr = feat[latih].copy()
ada = [c for c in feat.columns if c.startswith("ada_")]
sev_kol = [c for c in feat.columns if c.startswith("sev_") and c not in ("sev_mean", "sev_max", "sev_n")]
p_ada = (tr[ada] > 0).groupby(tr[TARGET]).mean().T              # proporsi kunjungan yang punya gejala itu
m_sev = tr[sev_kol].groupby(tr[TARGET]).mean().T                # rata-rata severity maks per gejala

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
heat(ax[0], p_ada, "Proporsi kunjungan yang memiliki gejala")
heat(ax[1], m_sev, "Rata-rata severity (0-100)", fmt="{:.0f}", cmap="Oranges")
plt.tight_layout(); plt.show()
print("kelompok gejala TANPA kolom severity:", sorted(set(c.replace("ada_", "") for c in ada) - set(c.replace("sev_", "") for c in sev_kol)))
```

**Alasan — Sel 21**

**Apa:** dua heatmap — (kiri) seberapa sering tiap kelompok gejala muncul di tiap kelas; (kanan) rata-rata tingkat keparahannya.

**Mengapa:** ini cara tercepat melihat apakah fitur gejala benar-benar membedakan kelas dan apakah polanya masuk akal secara klinis.

**Temuan (semuanya sesuai logika klinis, sehingga fitur dapat dipercaya):**
- **`acute_bronchitis`:** batuk (71%), sesak/mengi `dyspnea` (61% — hanya ±8% di kelas lain), dan sekret/dahak (70%) dominan.
- **`viral_sinusitis`:** gejala kepala/wajah (`head`, 63% vs ≤10% di kelas lain) dan keparahan `head` tinggi (61); hampir tidak ada gejala sistemik (5%).
- **`strep_pharyngitis` vs `viral_pharyngitis`** — pasangan tersulit: keduanya nyeri tenggorokan dan gejala sistemik. Pembedanya: **batuk hampir tidak ada pada strep (5%)** vs 50% pada viral; dan **keparahan nyeri tenggorokan lebih tinggi pada strep (56 vs 45)**.
- Kelompok **`lain`** muncul ±11% di *setiap* kelas → tidak membedakan kelas (bukti bahwa ia memang distraktor).
- Kelompok **`dyspnea` tidak punya kolom severity** (tidak pernah dilaporkan skornya), terlihat dari baris terakhir keluaran.

**Untuk write-up:** gunakan heatmap kiri sebagai bukti bahwa fitur gejala berbasis kamus sinonim bermakna klinis; severity tenggorokan dan batuk adalah dasar pemisahan strep vs viral pharyngitis.

### Sel 22 · Umur, jumlah gejala, dan riwayat per kelas

```python
urut = sorted(tr[TARGET].unique())
tr["n_gejala"] = tr[ada].sum(axis=1)                                    # jumlah baris gejala per kunjungan

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].boxplot([tr.loc[tr[TARGET] == k, "age"].dropna() for k in urut]); ax[0].set_xticklabels([k.replace("_", "\n") for k in urut], fontsize=8)
ax[0].set_title("Umur per kelas")
ax[1].boxplot([tr.loc[tr[TARGET] == k, "n_gejala"] for k in urut]); ax[1].set_xticklabels([k.replace("_", "\n") for k in urut], fontsize=8)
ax[1].set_title("Jumlah baris gejala per kunjungan")
plt.tight_layout(); plt.show()

print(tr.groupby(TARGET)[["age", "n_gejala"]].agg(["mean", "median"]).round(1).to_string())
print("\nriwayat (rata-rata per kelas):")
print(tr[[c for c in tr.columns if c.startswith("n_") and c != "n_gejala"]].groupby(tr[TARGET]).mean().round(2).T.to_string())
```

**Alasan — Sel 22**

**Apa:** membandingkan umur, jumlah gejala, dan jumlah riwayat antar kelas.

**Temuan:**
- **Umur:** pasien `strep_pharyngitis` lebih muda (median ±27 tahun vs ±37–39 pada kelas lain; rata-rata 33,2 vs 38,5–39,3). Ini sesuai pengetahuan klinis bahwa radang tenggorokan strep lebih sering pada anak dan remaja, jadi umur adalah fitur yang sah untuk memisahkan strep dari viral pharyngitis.
- **Jumlah gejala:** `acute_bronchitis` punya jauh lebih banyak baris gejala (rata-rata ±5,7 vs ±3,0–3,5). Karena itu jumlah gejala per kelompok (`ada_*`) berguna, bukan hanya ada/tidaknya.
- **Riwayat:** perbedaan antar-kelas kecil dan tidak punya penjelasan klinis yang jelas (mis. `n_observations` ±14 pada bronkitis vs ±9–11 pada kelas lain). Ini konsisten dengan hasil ablasi: riwayat memberi sinyal **tersebar dan lemah**, bukan satu pola kuat.

**Untuk write-up:** hindari klaim klinis tentang riwayat; cukup nyatakan "kontribusi kecil tetapi konsisten".

### Sel 23 · Tanda vital per kelas dan nilai tak wajar

```python
vital_kol = [c for c in feat.columns if c.startswith("vital_")]
print("median vital per kelas (median dipakai karena ada outlier yang belum dibersihkan):")
print(tr[vital_kol].groupby(tr[TARGET]).median().round(1).T.to_string())

WAJAR = {"vital_heart_rate": (30, 220), "vital_respiratory_rate": (5, 70), "vital_systolic_bp": (60, 260),
         "vital_diastolic_bp": (30, 150), "vital_spo2": (50, 100), "vital_bmi": (10, 70)}
print("\njumlah nilai di luar rentang wajar (seluruh 5.000 kunjungan):",
      {k: int(((feat[k] < lo) | (feat[k] > hi)).sum()) for k, (lo, hi) in WAJAR.items()})
```

**Alasan — Sel 23**

**Apa:** membandingkan vital antar kelas memakai **median**, dan menghitung nilai yang di luar rentang fisiologis wajar.

**Mengapa median:** rata-rata vital per kelas terdistorsi oleh outlier yang tidak dibersihkan: nilai maksimum di data adalah denyut jantung **990**, tekanan sistolik **1.390**, dan SpO2 **990** (tampak seperti salah ketik ×10). Median kebal terhadap hal itu.

**Temuan:** median tiap vital hampir sama di keempat kelas (mis. denyut jantung ±80, laju napas ±14), dan baris terakhir menunjukkan 9–42 nilai tak wajar per vital (total ±130 dari 30.000 nilai, ≈0,4%). Vital **tidak membedakan kelas**; bila ada perbedaan (mis. BMI strep lebih rendah) itu mencerminkan umur yang lebih muda, bukan sinyal vital itu sendiri.

**Konsekuensi dan keputusan:** karena vital tidak informatif, membersihkan outlier-nya lebih jauh tidak akan menaikkan skor. Ablasi (6.4) mengonfirmasi: menghapus semua kolom `vital_*` tidak menurunkan skor, bahkan sedikit menaikkannya (AUC 0,9513 → 0,9522) [run referensi]. Kolom vital tetap dipertahankan di pipeline final agar hasil identik dengan model yang sudah memberi skor terbaik di leaderboard (jangan mengubah resep yang sudah terbukti tanpa bukti yang kuat; selisih ini masih dalam noise).

**Untuk write-up:** laporkan ini sebagai *temuan negatif*: "kami membersihkan satuan vital, tetapi vital tidak berkontribusi prediktif".

### Sel 24 · Urutan baris gejala: bukti dan pengecekan konsistensi train–test

```python
gg = pd.read_csv(TRIASE / "triage_symptoms.csv")                       # urutan baris file asli
gg["kanon"] = norm(gg["symptom_concept"]).map(KANON).fillna("lain")
pertama = gg.groupby(KEY)["kanon"].first()                             # gejala di baris PERTAMA tiap kunjungan
feat["gejala_pertama"] = feat[KEY].map(pertama)
tr["gejala_pertama"] = tr[KEY].map(pertama)

p_pertama = pd.crosstab(tr["gejala_pertama"], tr[TARGET], normalize="columns")      # proporsi per kelas
print("proporsi gejala PERTAMA per kelas:"); print(p_pertama.round(2).to_string())

print("\nPERBANDINGAN: gejala ada kapan saja  vs  gejala ada di baris PERTAMA")
for k in ["fever", "nasal", "head", "cough"]:
    print(f"{k:6s} ada  :", {c: round(float(p_ada.loc[f'ada_{k}', c]), 2) for c in p_ada.columns})
    print(f"{k:6s} pertama:", {c: round(float(p_pertama.loc[k, c]), 2) if k in p_pertama.index else 0 for c in p_pertama.columns})

bagian = np.where(feat[TARGET].notna(), "train", "test")
tabel_tt = pd.crosstab(feat["gejala_pertama"], bagian)
print("\nchi-kuadrat distribusi gejala pertama train vs test: p =", round(chi2_contingency(tabel_tt)[1], 3))

fig, ax = plt.subplots(figsize=(7.5, 3.8)); heat(ax, p_pertama, "Proporsi gejala di baris pertama, per kelas"); plt.tight_layout(); plt.show()
feat = feat.drop(columns="gejala_pertama")          # hanya untuk EDA; bukan fitur model
```

**Alasan — Sel 24**

**Apa:** memeriksa apakah *urutan* baris gejala dalam satu kunjungan membawa informasi yang tidak ada pada sekadar *kehadiran* gejala.

**Mengapa ini penting dijelaskan:** TF-IDF atas teks gejala menaikkan skor jauh (bagian 6.4), padahal fitur `ada_*` sudah memuat gejala apa saja yang ada. Selisihnya harus punya sumber, dan sumber itu perlu dipahami dan dilaporkan dengan jujur.

**Temuan:**
- Distribusi **gejala pertama** sangat berbeda antar kelas meski kehadiran gejalanya mirip. Contoh paling jelas: demam *ada* pada 56% strep dan 50% viral pharyngitis (hampir sama), tetapi demam menjadi gejala **pertama** pada **49%** strep vs **25%** viral pharyngitis. Gejala `head` ada pada 6–10% non-sinusitis, namun **tidak pernah** menjadi gejala pertama pada kelas tersebut (0%) dan menjadi pertama pada 19% sinusitis. Batuk menjadi pertama pada 70% bronkitis.
- Pola ini **konsisten di test**: distribusi gejala pertama train vs test tidak berbeda nyata (χ², p ≈ 0,22), sehingga model yang memakainya dapat menggeneralisasi ke test.

**Interpretasi yang benar:** urutan baris mengikuti urutan pembangkitan gejala oleh generator data (tiap penyakit membangkitkan gejala dalam urutan khasnya). Ini **sidik jari cara data dibuat**, bukan pengetahuan medis: pada pasien sungguhan, urutan baris di file tidak memiliki arti diagnostik.

**Untuk write-up (wajib):** jangan klaim "model menemukan pola klinis dari urutan gejala". Tulis terbuka bahwa (1) model memanfaatkan urutan baris sebagai artefak dataset, (2) kontribusinya terukur lewat ablasi (6.4), dan (3) model tanpa fitur urutan tetap kompeten. Menampilkan kedua hasil lebih kredibel daripada menyembunyikan salah satunya.

# 5. Modeling

### Sel 25 · Konfigurasi model

```python
PARAM = dict(n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", random_state=SEED, verbose=-1)
# cadangan bila LightGBM tidak ada: konfigurasi yang dipilih lewat CV (probabilitas lebih terkalibrasi); ketidakseimbangan kelas
# ditangani di bagian 6.2, jadi tanpa class_weight
PARAM_HGB = dict(learning_rate=0.03, max_iter=250, max_leaf_nodes=8, min_samples_leaf=30, l2_regularization=3.0,
                 max_features=0.7, early_stopping=False, categorical_features="from_dtype")

def latih_fold(Xa, ya, Xb, yb, seed):
    """Latih satu model pada fold; kembalikan (model, jumlah iterasi)."""
    if HAS_LGB:
        m = lgb.LGBMClassifier(**{**PARAM, "random_state": seed})
        m.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, m.best_iteration_
    return HistGradientBoostingClassifier(**PARAM_HGB, random_state=seed).fit(Xa, ya), PARAM_HGB["max_iter"]

print("model:", "LightGBM" if HAS_LGB else "HistGradientBoosting (cadangan)")
```

**Alasan — Sel 25**

**Apa:** mendefinisikan model gradient boosting (LightGBM utama; `HistGradientBoostingClassifier` sebagai cadangan) dan fungsi pelatihan satu fold.

**Mengapa gradient boosting berbasis pohon:**
- Data tabular dengan campuran fitur numerik, kategorikal, banyak `NaN`, dan ratusan kolom TF-IDF yang jarang (*sparse*): pohon menangani `NaN`, skala berbeda, dan interaksi tanpa pra-pemrosesan tambahan.
- Hanya 3.600 sampel: model berkapasitas moderat dengan regularisasi (`num_leaves=15`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.5`) lebih aman daripada model sangat besar.

**Mengapa parameter ini:** `learning_rate=0.03` kecil + `n_estimators=3000` dengan *early stopping* 100 putaran → jumlah pohon ditentukan data, bukan ditebak. `colsample_bytree=0.5` memaksa tiap pohon hanya melihat setengah fitur sehingga tidak terlalu bergantung pada satu kolom TF-IDF. `class_weight="balanced"` memperbesar pengaruh kelas kecil saat pelatihan.

**Mengapa ada model cadangan:** agar notebook tetap berjalan tanpa LightGBM. Pada cadangan, `class_weight` dimatikan karena imbalance sengaja ditangani di langkah bobot kelas (6.2), dan probabilitasnya lebih terkalibrasi.

**Catatan untuk write-up (keterbatasan):** *early stopping* memakai fold validasi yang sama dengan yang dipakai untuk menilai OOF, sehingga skor OOF LightGBM sedikit optimistis. Itu salah satu alasan skor leaderboard lebih tepercaya sebagai estimasi akhir daripada OOF.

### Sel 26 · Cross-validation berulang → prediksi OOF dan prediksi test

```python
oof = np.zeros((len(y), len(KELAS)))                 # prediksi out-of-fold (tiap sampel diprediksi model yang TIDAK melihatnya)
pred_test = np.zeros((len(X_te), len(KELAS)))
penting = pd.Series(0.0, index=X.columns)
for r in range(N_REPEAT):
    for f, (a, b) in enumerate(StratifiedKFold(N_FOLD, shuffle=True, random_state=SEED + r).split(X_tr, y)):
        m, it = latih_fold(X_tr.iloc[a], y[a], X_tr.iloc[b], y[b], SEED + r)
        pb = m.predict_proba(X_tr.iloc[b])
        oof[b] += pb / N_REPEAT
        pred_test += m.predict_proba(X_te) / (N_FOLD * N_REPEAT)
        if HAS_LGB:
            penting += m.feature_importances_ / (N_FOLD * N_REPEAT)
        print(f"ulangan {r} fold {f}: iter {it}, macro-F1 {f1_score(y[b], pb.argmax(1), average='macro'):.4f}")
print("OOF macro-F1 (rata-rata ulangan):", round(f1_score(y, oof.argmax(1), average="macro"), 4))
if HAS_LGB:
    print(penting.sort_values(ascending=False).head(20).round(0))
```

**Alasan — Sel 26**

**Apa:** `StratifiedKFold` 5-fold diulang 3 kali dengan pengacakan berbeda (15 model). Tiap sampel train mendapat prediksi dari model yang **tidak pernah melihatnya** (OOF), dirata-rata atas 3 ulangan. Prediksi test adalah rata-rata dari ke-15 model.

**Mengapa strategi ini:**
- **Stratified:** kelas `strep` hanya 195 sampel; tanpa stratifikasi, sebuah fold bisa kebetulan hanya berisi 25 atau 55 sampel strep.
- **Tidak perlu `GroupKFold`:** satu pasien = satu baris (terbukti di 2.3 dan 3.2), jadi tidak ada kebocoran antar-fold lewat pasien yang sama.
- **Diulang 3×:** dengan 39 sampel strep per fold, skor satu kali CV sangat berisik (skor per fold bervariasi 0,67–0,76 pada keluaran referensi). Merata-ratakan ulangan menstabilkan OOF dan, nanti, bobot kelas.
- **OOF (bukan skor train):** satu-satunya estimasi jujur performa pada data baru.
- **Rata-rata 15 model untuk test:** memangkas varians tanpa melatih ulang pada seluruh data; tiap model sudah dilatih pada 80% data.

**Temuan [run referensi, HistGradientBoosting]:** OOF macro-F1 **0,712** sebelum bobot kelas; skor per fold bervariasi 0,67–0,76 (menunjukkan seberapa berisik evaluasi pada data sekecil ini).

**Untuk write-up:** sebutkan "5-fold stratified, 3 ulangan, 15 model per prediksi" dan jelaskan alasannya (kelas minoritas 195 sampel; 1 pasien = 1 baris).

# 6. Evaluasi

## 6.1 Kinerja OOF sebelum penyesuaian bobot kelas

### Sel 27 · Laporan klasifikasi dan confusion matrix (argmax probabilitas)

```python
pred_raw = oof.argmax(1)
print("OOF macro-F1:", round(f1_score(y, pred_raw, average="macro"), 4), "| AUC macro (OvR):", round(roc_auc_score(y, oof, multi_class="ovr", average="macro"), 4))
print(classification_report(y, pred_raw, target_names=KELAS, digits=3))

fig, ax = plt.subplots(figsize=(5.8, 5))
ConfusionMatrixDisplay(confusion_matrix(y, pred_raw), display_labels=[k.replace("_", "\n") for k in KELAS]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion matrix OOF (sebelum bobot kelas)"); plt.tight_layout(); plt.show()
```

**Alasan — Sel 27**

**Apa:** metrik per kelas (precision, recall, F1) dan confusion matrix dari prediksi OOF.

**Mengapa:** macro-F1 tunggal menyembunyikan *di mana* model salah. Laporan per kelas menunjukkan bahwa performa sangat berbeda antar kelas.

**Temuan [run referensi]:**
- `viral_sinusitis` (F1 ±0,92) dan `acute_bronchitis` (±0,88) mudah — gejalanya khas (kepala/wajah dan batuk/sesak).
- `viral_pharyngitis` sedang (±0,77) dan `strep_pharyngitis` sangat rendah (F1 ±0,27; **recall hanya ±19%**): dari 195 kasus strep, hanya sekitar 37 yang terdeteksi; sebagian besar disalahklasifikasikan sebagai `viral_pharyngitis`.
- Secara klinis ini masuk akal: strep dan viral pharyngitis memang hampir identik pada gejala (keduanya nyeri tenggorokan + gejala sistemik) dan hanya dibedakan oleh batuk, keparahan nyeri tenggorokan, dan umur.
- AUC macro tinggi (0,952) sementara F1 strep rendah: model **mengurutkan** kasus strep dengan baik, tetapi **ambang keputusan** argmax terlalu konservatif untuk kelas 5%. Inilah motivasi langkah berikutnya.

## 6.2 Penyesuaian bobot kelas untuk macro-F1

### Sel 28 · Mencari bobot kelas pada OOF + uji kestabilan

```python
MIN_GAIN = 0.001                       # kenaikan <= ini dianggap noise -> bobot diabaikan
KANDIDAT = np.round(np.arange(0.5, 2.51, 0.05), 2)

def f1_bobot(yy, PP, w):
    return f1_score(yy, (PP * w).argmax(1), average="macro")

def cari_bobot(yy, PP, passes=3):
    """Koordinat demi koordinat: ubah bobot satu kelas, ambil yang memaksimalkan macro-F1."""
    w = np.ones(PP.shape[1])
    for _ in range(passes):
        for k in range(len(w)):
            skor = [f1_bobot(yy, PP, np.where(np.arange(len(w)) == k, c, w)) for c in KANDIDAT]
            w[k] = KANDIDAT[int(np.argmax(skor))]
    return w

w = cari_bobot(y, oof)
f1_awal, f1_akhir = f1_bobot(y, oof, np.ones(len(KELAS))), f1_bobot(y, oof, w)
gain = f1_akhir - f1_awal
w_pakai = w if gain > MIN_GAIN else np.ones(len(KELAS))
print("bobot:", {k: float(round(b, 2)) for k, b in zip(KELAS, w)})
print(f"macro-F1 OOF: {f1_awal:.4f} -> {f1_akhir:.4f} (gain {gain:+.4f}) | dipakai: {'ya' if gain > MIN_GAIN else 'TIDAK (gain terlalu kecil)'}")

# uji kestabilan: bobot disetel di separuh data, dinilai di separuh lain (gain harus tetap positif)
rng, gains = np.random.RandomState(0), []
for _ in range(10):
    idx = rng.permutation(len(y)); i1, i2 = idx[: len(y) // 2], idx[len(y) // 2:]
    w1 = cari_bobot(y[i1], oof[i1])
    gains.append(f1_bobot(y[i2], oof[i2], w1) - f1_bobot(y[i2], oof[i2], np.ones(len(KELAS))))
print(f"gain pada data yang tidak dipakai menyetel: rata-rata {np.mean(gains):+.4f}, terendah {np.min(gains):+.4f}")
```

**Alasan — Sel 28**

**Apa:** mengalikan probabilitas tiap kelas dengan sebuah bobot sebelum `argmax`, lalu mencari bobot yang memaksimalkan macro-F1 pada OOF (satu kelas demi satu kelas, 3 putaran).

**Mengapa:** `argmax` pada probabilitas yang tidak dibobot mengoptimalkan *akurasi*, bukan macro-F1. Untuk kelas 5%, probabilitas strep jarang melampaui kelas lain meski kasusnya sebenarnya strep. Menaikkan bobot strep memindahkan ambang keputusan sehingga lebih banyak kasus strep tertangkap, dengan mengorbankan sedikit presisi. Karena macro-F1 menimbang kelas kecil sama dengan kelas besar, pertukaran ini menguntungkan.

**Mengapa ada dua pengaman (ini yang membuat langkah ini dapat dipertanggungjawabkan):**
1. **`MIN_GAIN`:** bila kenaikan ≤ 0,001, bobot dianggap noise dan diabaikan.
2. **Uji kestabilan split-half:** bobot disetel di separuh data lalu dinilai di separuh lain, diulang 10 kali. Penyetelan bobot pada seluruh OOF *selalu* terlihat bagus pada data yang dipakai menyetel; angka yang jujur adalah gain pada data yang tidak dipakai menyetel.

**Temuan [run referensi, HGB tanpa class_weight]:** bobot `acute_bronchitis` 0,85; `strep` **1,70**; `viral_pharyngitis` 1,00; `viral_sinusitis` **0,50**. Macro-F1 OOF naik **0,7120 → 0,7412 (+0,029)**; pada data yang tidak dipakai menyetel gain tetap **+0,021 rata-rata dan tidak pernah negatif (minimum +0,008)** → peningkatannya nyata, bukan sekadar overfit.

**Catatan penting untuk write-up:** langkah ini efektif karena model dasar **tidak** dibobot. Pada eksperimen lain di proyek ini, model yang sudah memakai bobot kelas lunak sejak pelatihan tidak mendapat gain tambahan (uji split-half-nya negatif). Jadi bobot di sini bukan "trik ajaib", tetapi cara menangani imbalance di *satu tempat* (keputusan) alih-alih dua. Bila kamu memakai LightGBM dengan `class_weight="balanced"`, bobot hasilnya akan berbeda dan gain-nya lebih kecil; biarkan uji kestabilan yang memutuskan.

### Sel 29 · Hasil setelah bobot kelas

```python
pred_w = (oof * w_pakai).argmax(1)
print("OOF macro-F1 setelah bobot:", round(f1_score(y, pred_w, average="macro"), 4))
print(classification_report(y, pred_w, target_names=KELAS, digits=3))

banding = pd.DataFrame({
    "F1 sebelum": f1_score(y, pred_raw, average=None), "F1 sesudah": f1_score(y, pred_w, average=None),
    "recall sebelum": [(pred_raw[y == k] == k).mean() for k in range(len(KELAS))],
    "recall sesudah": [(pred_w[y == k] == k).mean() for k in range(len(KELAS))]}, index=KELAS).round(3)
print(banding.to_string())

fig, ax = plt.subplots(figsize=(5.8, 5))
ConfusionMatrixDisplay(confusion_matrix(y, pred_w), display_labels=[k.replace("_", "\n") for k in KELAS]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion matrix OOF (setelah bobot kelas)"); plt.tight_layout(); plt.show()
```

**Alasan — Sel 29**

**Apa:** laporan klasifikasi dan confusion matrix setelah bobot, plus tabel perbandingan F1 dan recall tiap kelas sebelum/sesudah.

**Mengapa:** peningkatan macro-F1 harus dijelaskan *dari mana*, bukan hanya diumumkan. Tabel perbandingan menunjukkan pertukaran yang terjadi.

**Temuan [run referensi]:** recall strep naik dari **19% menjadi 38%** dan F1 strep dari **0,269 menjadi 0,399** (presisinya turun dari 0,46 ke 0,42). Sebagai gantinya, kelas besar turun sedikit: recall `viral_sinusitis` dari 94,5% ke 89,6% (F1 0,922 → 0,915), sedangkan `acute_bronchitis` dan `viral_pharyngitis` hampir tidak berubah. Karena macro-F1 menimbang keempat kelas sama, kenaikan +0,13 pada F1 strep lebih besar daripada penurunan kecil di kelas besar — itulah sumber gain +0,029.

**Untuk write-up:** tampilkan tabel perbandingan ini; ia menjelaskan *mekanisme* penyesuaian bobot lebih baik daripada satu angka macro-F1.

## 6.3 Seberapa pasti skornya?

### Sel 30 · Ketidakpastian skor pada test sebesar 1.400 baris

```python
P_final = oof * w_pakai
rng = np.random.RandomState(0)
skor_sub = np.array([f1_score(y[i], P_final[i].argmax(1), average="macro")
                     for i in (rng.choice(len(y), 1400, replace=False) for _ in range(300))])
print(f"macro-F1 OOF seluruh train      : {f1_score(y, P_final.argmax(1), average='macro'):.4f}")
print(f"pada subset acak 1.400 baris    : rata-rata {skor_sub.mean():.4f} | simpangan baku {skor_sub.std():.4f} | rentang 5-95% [{np.percentile(skor_sub, 5):.3f}, {np.percentile(skor_sub, 95):.3f}]")
print(f"jumlah strep yang diharapkan di test (1.400 baris): ±{(y == 1).mean() * 1400:.0f}")
```

**Alasan — Sel 30**

**Apa:** mensimulasikan "skor pada test sebesar 1.400 baris" dengan mengambil 300 subset acak 1.400 baris dari OOF, lalu mengukur sebaran macro-F1-nya.

**Mengapa:** leaderboard memakai test berukuran 1.400 dengan hanya ±76 kasus strep. Skor di ukuran sekecil itu **berfluktuasi** secara alami, sehingga selisih kecil antar-submission tidak otomatis berarti satu model lebih baik.

**Temuan [run referensi]:** simpangan baku macro-F1 pada ukuran test ≈ **0,013** (rentang 5–95% sekitar ±0,02). Artinya selisih skor leaderboard di bawah ±0,02 antara dua submission **tidak dapat dibedakan dari keberuntungan sampel**. Validasi adversarial (3.4) menunjukkan tidak ada pergeseran distribusi train–test, jadi sumber fluktuasinya adalah ukuran sampel.

**Untuk write-up:** ini jawaban jika ditanya "mengapa skor leaderboard berbeda dari OOF?": sertakan simpangan baku ini sebagai bentuk kejujuran statistik, dan pilih model berdasarkan CV yang berulang, bukan satu angka leaderboard.

## 6.4 Ablasi: apa yang sebenarnya memberi sinyal?

### Sel 31 · Ablasi blok fitur (opsional, `RUN_ABLATION`)

```python
REFERENSI = pd.DataFrame([   # hasil run verifikasi: HistGradientBoosting, 5-fold, 1 ulangan (SEED), bobot kelas disetel pada OOF yang sama
    ("A  final (TF-IDF urutan asli)",         707, 0.7178, 0.7357, 0.9513),
    ("B  tanpa TF-IDF sama sekali",            46, 0.6898, 0.7177, 0.9455),
    ("C  TF-IDF, URUTAN gejala diacak",      1043, 0.6906, 0.7237, 0.9459),
    ("D  TF-IDF hanya teks keluhan",           64, 0.6924, 0.7115, 0.9452),
    ("E  final tanpa vital_*",                701, 0.7113, 0.7460, 0.9522),
    ("F  final tanpa riwayat (n_*, hari_*)",  694, 0.7039, 0.7273, 0.9474),
], columns=["varian", "n_kolom", "F1_mentah", "F1_setelah_bobot", "AUC"]).set_index("varian")

if RUN_ABLATION:
    def bangun_X(teks, buang_awalan=(), pakai_tfidf=True):
        f = feat.copy(); f["teks"] = teks
        bagian = [f.drop(columns=BUANG)]
        if pakai_tfidf:
            tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(f["teks"])
            bagian.append(pd.DataFrame(tf.transform(f["teks"]).toarray(), columns="tf_" + tf.get_feature_names_out()))
        Xa = pd.concat(bagian, axis=1); Xa.columns = Xa.columns.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        Xa = Xa.drop(columns=[c for c in Xa.columns if Xa[c].nunique(dropna=True) <= 1])
        if buang_awalan:
            Xa = Xa.drop(columns=[c for c in Xa.columns if c.startswith(tuple(buang_awalan))])
        return Xa[latih].reset_index(drop=True)

    g_acak = g.sample(frac=1, random_state=0)                                  # urutan baris gejala diacak
    VARIAN = {
        "A  final (TF-IDF urutan asli)":  lambda: bangun_X(feat["teks"]),
        "B  tanpa TF-IDF sama sekali":    lambda: bangun_X(feat["teks"], pakai_tfidf=False),
        "C  TF-IDF, URUTAN gejala diacak": lambda: bangun_X(fitur_teks(g_acak, keluhan, feat[KEY])),
        "D  TF-IDF hanya teks keluhan":   lambda: bangun_X(" || " + norm(feat[KEY].map(keluhan).fillna(""))),
        "E  final tanpa vital_*":         lambda: bangun_X(feat["teks"], buang_awalan=("vital_",)),
        "F  final tanpa riwayat (n_*, hari_*)": lambda: bangun_X(feat["teks"], buang_awalan=("n_", "hari_")),
    }
    baris = []
    for nama, bangun in VARIAN.items():
        Xa = bangun(); o = np.zeros((len(y), len(KELAS)))
        for a, b in StratifiedKFold(N_FOLD, shuffle=True, random_state=SEED).split(Xa, y):
            o[b] = HistGradientBoostingClassifier(**PARAM_HGB, random_state=SEED).fit(Xa.iloc[a], y[a]).predict_proba(Xa.iloc[b])
        wa = cari_bobot(y, o)
        baris.append((nama, Xa.shape[1], f1_score(y, o.argmax(1), average="macro"), f1_bobot(y, o, wa),
                      roc_auc_score(y, o, multi_class="ovr", average="macro")))
        print(f"{nama:40s} kolom={Xa.shape[1]:4d} F1={baris[-1][2]:.4f} F1+bobot={baris[-1][3]:.4f} AUC={baris[-1][4]:.4f}")
    REFERENSI = pd.DataFrame(baris, columns=REFERENSI.reset_index().columns).set_index("varian")
else:
    print("RUN_ABLATION=False -> menampilkan hasil run referensi (set True untuk menjalankan ulang, ±6 menit).")
REFERENSI.round(4)
```

**Alasan — Sel 31**

**Apa:** enam varian pipeline yang masing-masing menghilangkan/mengubah satu blok fitur, dievaluasi dengan CV 5-fold (HGB, 1 ulangan). Dijalankan opsional (`RUN_ABLATION=True`) karena memakan waktu; tabel hasil run referensi selalu ditampilkan.

**Mengapa ablasi:** untuk write-up, "model memakai fitur X" tidak cukup; pembaca perlu tahu fitur mana yang *benar-benar* menyumbang. Ablasi menjawabnya secara eksperimental. **AUC** disertakan di samping F1 karena AUC jauh lebih stabil terhadap noise ambang keputusan.

**Temuan [run referensi] dan cara membacanya:**
| Perbandingan | Kesimpulan |
|---|---|
| **A vs B** (ada/tidak TF-IDF): AUC 0,9513 vs 0,9455; F1+bobot 0,7357 vs 0,7177 | TF-IDF memberi peningkatan nyata (±+0,006 AUC, ±+0,018 F1) |
| **A vs C** (urutan asli vs diacak): AUC 0,9513 vs 0,9459 | Jika urutan baris gejala diacak, seluruh keuntungan TF-IDF **hilang** (C ≈ B). Keuntungannya berasal dari **urutan**, bukan dari kata-katanya |
| **A vs D** (TF-IDF hanya dari keluhan): AUC 0,9452 | Teks keluhan utama tidak menyumbang apa-apa (D ≈ B) |
| **A vs E** (tanpa vital): AUC 0,9522 vs 0,9513 | Vital **tidak berkontribusi**; menghapusnya sama saja atau sedikit lebih baik |
| **A vs F** (tanpa riwayat): AUC 0,9474 vs 0,9513; F1+bobot 0,7273 vs 0,7357 | Riwayat memberi sinyal **nyata namun kecil**, tersebar di banyak tabel |

**Untuk write-up:** ablasi ini adalah bagian terkuat dari analisis kamu karena mengukur kontribusi tiap komponen dengan eksperimen. Presentasikan **A dan B berdampingan**: B (tanpa fitur urutan) adalah kinerja "bersih" model pada informasi medis; A adalah kinerja yang juga memanfaatkan artefak urutan data. Menjelaskan perbedaannya secara terbuka adalah bentuk integritas ilmiah.

**Catatan (batasan):** satu ulangan, satu split; selisih AUC < 0,001 (mis. A vs E) tidak bermakna. Kamu dapat menaikkan keandalannya dengan mengulang ablasi pada beberapa `SEED`.

# 7. Submission

### Sel 32 · Membuat file submission

```python
contoh = pd.read_csv(DATA / "sample_submission.csv")
hasil = pd.DataFrame({ID: test[ID], TARGET: np.array(KELAS)[(pred_test * w_pakai).argmax(1)]})
sub_df = contoh[[ID]].merge(hasil, on=ID, how="left")
assert len(sub_df) == len(contoh) and sub_df[TARGET].notna().all(), "ID test tidak cocok dengan sample_submission"
sub_df.to_csv(OUT / "submission.csv", index=False)
pd.DataFrame(oof, columns=KELAS).assign(**{ID: train[ID].to_numpy()}).to_csv(OUT / "oof_model.csv", index=False)

banding_prior = pd.DataFrame({"proporsi test (prediksi)": sub_df[TARGET].value_counts(normalize=True),
                              "proporsi train (asli)": train[TARGET].value_counts(normalize=True)}).round(3)
print(banding_prior.to_string()); print("tersimpan:", OUT / "submission.csv")
```

**Alasan — Sel 32**

**Apa:** menerapkan bobot kelas pada prediksi test, memetakan indeks kelas kembali ke nama, lalu menyimpan `submission.csv` (dan OOF untuk analisis lanjutan).

**Mengapa:**
- `merge` ke `sample_submission` menjamin urutan baris sama dengan yang diminta kompetisi, dan `assert` memastikan tidak ada `sample_id` yang hilang atau kosong; kesalahan format di sini membuat submission ditolak atau, lebih buruk, diterima dengan skor salah.
- Prediksi test memakai **bobot yang sama** dengan yang disetel pada OOF; menerapkan bobot berbeda ke test akan memutus hubungan antara skor OOF dan skor test.
- Perbandingan **proporsi prediksi test vs proporsi train** adalah sanity check terakhir: bila proporsi prediksi sangat menyimpang (mis. strep 30%), ada yang salah. Pada run referensi, strep diprediksi 5,1% (prior 5,4%); `viral_sinusitis` turun ke 36,8% (prior 41,1%) dan `viral_pharyngitis` naik ke 32,6% (prior 27,4%) — pergeseran ini wajar karena bobot kelas sengaja menurunkan sinusitis (0,50) dan menaikkan strep (1,70). Proporsi prediksi tidak harus sama dengan prior, tetapi tidak boleh ekstrem.
- `oof_model.csv` disimpan agar analisis ensemble/ablasi dapat dilakukan tanpa melatih ulang.

# 8. Catatan untuk Write-up

**Alur ringkas pipeline:** audit kualitas data → pembersihan (umur, income, vital, gejala) → filter waktu pada riwayat → fitur per tabel + TF-IDF → LightGBM/HGB dengan `StratifiedKFold` berulang → bobot kelas pada OOF → submission.

**Apa yang perlu dilaporkan secara terbuka (poin kejujuran):**
1. **Artefak urutan baris:** sebagian kenaikan skor berasal dari *urutan baris gejala* di file, yaitu jejak cara data dibangkitkan, bukan fenomena medis (bagian 4.3, ablasi 6.4). Laporkan model dengan dan tanpa fitur ini.
2. **Temuan negatif:** vital (sudah dibersihkan satuannya) dan teks keluhan tidak berkontribusi; riwayat berkontribusi kecil.
3. **Bobot kelas:** efektif karena model dasar tidak dibobot; gain-nya diuji dengan split-half (bukan hanya in-sample).
4. **Ketidakpastian:** skor pada test 1.400 baris punya simpangan baku ≈0,013; selisih antar-submission < 0,02 tidak bermakna (bagian 6.3).
5. **Keterbatasan OOF:** *early stopping* LightGBM memakai fold validasi, jadi OOF sedikit optimistis.

**Bagian yang sebaiknya muncul di write-up (saran struktur):** (a) EDA & tabel temuan kualitas data (`AUDIT`), (b) desain fitur per tipe tabel + alasan anti-kebocoran, (c) skema validasi, (d) hasil + ablasi, (e) keterbatasan dan etika penggunaan artefak data.

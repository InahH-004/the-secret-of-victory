"""
relational_features.py
======================
TEMPLATE REUSABLE: fitur dari banyak tabel relasional dengan aturan point-in-time.

Tidak ada logika khusus GDGoC di file ini. Semua keputusan dataset-spesifik
(rentang nilai, tabel mana pakai join apa, dsb.) ada di file konfigurasi
(contoh: run_gdgoc.py). Cara adaptasi: lihat TEMPLATE_GUIDE.md.
"""
import glob                      # mencari file dengan pola (termasuk subfolder)
import os                        # manipulasi path dan nama file
import re                        # regex untuk membersihkan nama kolom

import numpy as np               # operasi numerik (NaN, log, dll.)
import pandas as pd              # tabel dan agregasi

# Resolusi prediction_time (detik penuh). Event yang terjadi di detik yang sama
# dengan prediction_time dianggap sudah tersedia saat prediksi dibuat.
ONE_SECOND = pd.Timedelta(seconds=1)


# =============================================================================
# 1. LOADING
# =============================================================================
def load_tables(root, exclude=("DATA_DICTIONARY",)):
    """Baca semua CSV di root secara rekursif. Kunci dict = nama file tanpa .csv."""
    tables = {}                                                    # wadah hasil: nama tabel -> DataFrame
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):  # cari semua .csv, termasuk subfolder
        name = os.path.splitext(os.path.basename(path))[0]         # ambil nama file tanpa folder dan ekstensi
        if name in exclude:                                        # lewati file metadata
            continue                                               # lanjut ke file berikutnya
        tables[name] = pd.read_csv(path)                           # baca CSV ke DataFrame
    return tables                                                  # kembalikan semua tabel


# =============================================================================
# 2. PARSING (satu fungsi per jenis format angka/teks/waktu)
# =============================================================================
def parse_ts(series):
    """Ubah string waktu campuran (dengan/tanpa 'Z', dengan/tanpa milidetik) menjadi datetime UTC."""
    return pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")  # gagal parse -> NaT


def to_float_decimal_comma(series):
    """Angka dengan koma desimal ('18,33') atau titik ('18.33') menjadi float. Selain itu NaN."""
    s = series.astype("string").str.strip()                        # jadikan string dan buang spasi tepi
    s = s.str.replace(",", ".", regex=False)                       # koma -> titik sebagai desimal
    return pd.to_numeric(s, errors="coerce").astype(float)         # angka valid -> float, sisanya NaN


def parse_amount(x):
    """Pendapatan dengan format campuran: 'USD 122,462', '13.1k', '20,982' (koma = pemisah ribuan)."""
    if pd.isna(x):                                                 # nilai kosong langsung NaN
        return np.nan                                              # kembalikan NaN
    s = str(x).upper().replace("USD", "").replace("$", "").replace(",", "").strip()  # buang USD/$, koma ribuan, spasi
    mult = 1.0                                                     # pengali default
    if s.endswith("K"):                                            # akhiran K berarti ribuan
        mult, s = 1_000.0, s[:-1]                                  # set pengali 1000 dan buang K
    try:                                                           # coba ubah ke angka
        return float(s) * mult                                     # kalikan pengali
    except ValueError:                                             # jika bukan angka
        return np.nan                                              # tandai NaN


def clean_col_names(df):
    """LightGBM menolak karakter seperti spasi, [ ], :, koma pada nama kolom. Ganti semuanya dengan _."""
    df = df.copy()                                                 # jangan ubah DataFrame asli
    seen, names = {}, []                                           # seen: hitung nama yang sudah dipakai
    for c in df.columns:                                           # proses tiap nama kolom
        base = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")   # regex ganti karakter ilegal dengan _
        n = seen.get(base, 0)                                      # sudah berapa kali nama dasar ini muncul
        seen[base] = n + 1                                         # tambah hitungan
        names.append(base if n == 0 else f"{base}_{n + 1}")        # tabrakan -> beri akhiran _2, _3, ...
    df.columns = names                                             # pasang nama baru
    assert df.columns.is_unique                                    # pengaman terakhir: pasti unik
    return df                                                      # kembalikan DataFrame bersih


# =============================================================================
# 3. CUTOFF (point-in-time)
# =============================================================================
def attach_cutoff(df, index_df, key):
    """Tambah kolom 'cutoff' (prediction_time) ke tiap baris lewat join pada key."""
    lookup = index_df[[key, "cutoff"]]                             # ambil hanya kunci dan cutoff dari tabel indeks
    return df.merge(lookup, on=key, how="inner", validate="many_to_one")  # join; validate memastikan cutoff unik per key


def apply_cutoff(df, time_col, log, name, strict=False):
    """
    Simpan hanya baris yang sudah tersedia pada cutoff.
      strict=False : tersedia jika ts < cutoff + 1 detik  (dipakai untuk pipeline)
      strict=True  : tersedia jika ts < cutoff            (hanya untuk diagnosis)
    Setiap pemanggilan menambah satu catatan ke `log` (list of dict).
    """
    ts = parse_ts(df[time_col])                                    # ubah kolom waktu ke datetime UTC
    limit = df["cutoff"] if strict else df["cutoff"] + ONE_SECOND  # pilih batas: cutoff atau cutoff+1 detik
    keep = ts < limit                                              # True jika event sudah tersedia (NaT -> False)
    log.append({                                                   # catat keputusan agar bisa ditulis di laporan
        "table": name,                                             # nama tabel
        "time_col": time_col,                                      # kolom waktu yang dipakai
        "mode": "strict(<cutoff)" if strict else "same-second(<cutoff+1s)",  # aturan yang dipakai
        "rows_in": int(len(df)),                                   # jumlah baris sebelum filter
        "kept": int(keep.sum()),                                   # baris yang lolos
        "dropped": int((~keep).sum()),                             # baris yang dibuang
        "ts_unparsed": int(ts.isna().sum()),                       # waktu yang gagal diparse
    })
    out = df.loc[keep].copy()                                      # ambil baris yang lolos
    out["_ts"] = ts[keep]                                          # simpan waktu event sebagai datetime
    out["_days_before"] = ((out["cutoff"] - out["_ts"]).dt.total_seconds() / 86400).clip(lower=0)  # jarak hari ke cutoff (>=0)
    return out                                                     # kembalikan tabel yang sudah dipangkas


def check_no_future(df, log, name):
    """Pengaman: catat berapa baris yang waktunya >= cutoff+1 detik. Idealnya 0 setelah apply_cutoff."""
    ts = df["_ts"]                                                 # waktu event yang tersimpan
    bad = int((ts >= df["cutoff"] + ONE_SECOND).sum())             # hitung yang melewati batas
    log.append({"table": name, "check": "future_rows", "count": bad})  # catat hasil pemeriksaan
    assert bad == 0, f"{name}: {bad} baris melewati cutoff"        # hentikan pipeline jika ada kebocoran


# =============================================================================
# 4. RENTANG NILAI (aturan klinis / domain)
# =============================================================================
def apply_valid_range(df, name_col, value_col, ranges, log, name):
    """Nilai di luar [min, max] untuk nama tertentu dijadikan NaN. Nilai tidak diperbaiki/ditebak."""
    df = df.copy()                                                 # jangan ubah input
    for label, (lo, hi) in ranges.items():                         # tiap konsep punya batas sendiri
        m = (df[name_col] == label) & df[value_col].notna()        # baris dengan label ini dan nilai ada
        out = m & ((df[value_col] < lo) | (df[value_col] > hi))    # yang berada di luar batas
        log.append({"table": name, "check": f"range:{label}", "out_of_range_set_nan": int(out.sum()), "bounds": (lo, hi)})
        df.loc[out, value_col] = np.nan                            # buang nilai tidak masuk akal
    return df                                                      # kembalikan tabel dengan nilai valid


# =============================================================================
# 5. AGREGASI (per key = patient_id atau encounter_id)
# =============================================================================
def vocab_from(df, code_col, min_count):
    """Daftar kode yang muncul minimal min_count kali. Dihitung dari fitur (tanpa label), aman untuk train+test."""
    vc = df[code_col].value_counts()                               # hitung kemunculan tiap kode
    return set(vc[vc >= min_count].index)                          # simpan kode yang cukup sering


def agg_recency(df, key, prefix):
    """Jumlah event sebelum cutoff, dan jarak hari ke event terbaru."""
    g = df.groupby(key)                                            # kelompokkan per pasien/encounter
    return pd.DataFrame({
        f"{prefix}_n": g.size(),                                   # jumlah baris event
        f"{prefix}_days_since_last": g["_days_before"].min(),      # event terbaru = jarak hari terkecil
    })


def agg_code_bag(df, key, code_col, prefix, vocab):
    """Satu kolom biner per kode: 1 jika kode itu muncul untuk key tersebut sebelum cutoff."""
    sub = df[df[code_col].isin(vocab)]                             # ambil hanya kode yang masuk vocab
    if sub.empty:                                                  # jika kosong, kembalikan tabel kosong
        return pd.DataFrame(index=pd.Index([], name=key))
    ct = pd.crosstab(sub[key], sub[code_col])                      # tabel silang: baris key, kolom kode, isi jumlah
    ct = (ct > 0).astype(np.int8)                                  # ubah jumlah menjadi 0/1 (ada/tidak)
    ct.columns = [f"{prefix}__{c}" for c in ct.columns]            # beri awalan supaya nama kolom jelas asal tabelnya
    return ct                                                      # kembalikan matriks biner


def agg_numeric_by_code(df, key, code_col, value_col, prefix, vocab, stats=("last", "mean")):
    """Nilai numerik per kode. stats: last (terakhir sebelum cutoff), mean, min, max, dll."""
    sub = df[df[code_col].isin(vocab) & df[value_col].notna()]     # hanya kode di vocab dan nilai tidak kosong
    sub = sub.sort_values("_ts")                                   # urutkan waktu supaya 'last' benar
    parts = []                                                     # wadah hasil tiap statistik
    for st in stats:                                               # ulangi untuk tiap jenis statistik
        grouped = sub.groupby([key, code_col])[value_col]          # kelompok (key, kode)
        w = getattr(grouped, st)().unstack(code_col)               # hitung statistik lalu jadikan kolom per kode
        w.columns = [f"{prefix}__{c}__{st}" for c in w.columns]    # beri nama: prefix__kode__statistik
        parts.append(w)                                            # simpan
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()   # gabungkan semua statistik secara kolom


def explode_tokens(df, key, col, sep=",", strip_prefix=None):
    """Teks bebas 'Reports: Fever, Throat' menjadi satu baris per token (key, token)."""
    s = df[col].astype("string")                                   # pastikan tipe string
    if strip_prefix:                                               # buang awalan tetap jika ada
        s = s.str.replace(strip_prefix, "", regex=False)
    tmp = df[[key]].copy()                                         # ambil kolom key saja
    tmp["token"] = s.str.split(sep)                                # pecah teks berdasarkan pemisah
    tmp = tmp.explode("token").dropna(subset=["token"]).reset_index(drop=True)  # satu token per baris; reset index agar unik
    tmp["token"] = tmp["token"].astype("string").str.strip().str.lower()  # rapikan spasi dan huruf kecil
    return tmp[tmp["token"].fillna("") != ""]                      # buang token kosong


def to_entity_level(part, universe, level):
    """Ubah fitur level pasien menjadi level encounter (satu pasien = satu encounter di tabel indeks)."""
    if level == "encounter":                                       # sudah level encounter
        return part                                                # kembalikan apa adanya
    aligned = part.reindex(universe["patient_id"].values)          # urutkan baris mengikuti urutan universe
    aligned.index = universe["encounter_id"].values                # ganti index menjadi encounter_id
    return aligned                                                 # kembalikan level encounter

"""
run_gdgoc.py
============
Pipeline fitur untuk GDGoC UGM Academic Hacker 2026 (4 kelas, Macro F1).
Semua logika umum ada di relational_features.py. File ini hanya berisi
KEPUTUSAN spesifik dataset (lihat LOG_KEPUTUSAN.md untuk alasannya).

Cara jalankan:
    python run_gdgoc.py --root "/home/claude/data" --out "/mnt/user-data/outputs/hasil"
"""
import argparse                       # membaca argumen dari command line
import json                           # menyimpan log keputusan ke file
import os                             # membuat folder output

import numpy as np                    # operasi numerik
import pandas as pd                   # tabel
import relational_features as rf      # template fitur relasional

# =============================================================================
# KONFIGURASI DATASET (keputusan spesifik, dicatat di LOG_KEPUTUSAN.md)
# =============================================================================
MIN_COUNT = 20                        # kode minimal muncul 20 kali agar jadi kolom (hindari kolom jarang)

# Rentang fisiologis: nilai di luar rentang dianggap salah input dan dijadikan NaN.
VITAL_RANGES = {                      # nama dari triage_vitals.vital_name
    "Heart rate": (20, 250),          # detak jantung per menit
    "Respiratory rate": (4, 60),      # napas per menit
    "Systolic BP": (50, 260),         # tekanan sistolik mmHg
    "Diastolic BP": (30, 160),        # tekanan diastolik mmHg
    "SpO2": (50, 100),                # saturasi oksigen %
    "BMI": (10, 80),                  # indeks massa tubuh kg/m2
}
OBS_RANGES = {                        # nama dari observations.observation_name (dicocokkan persis)
    "Systolic Blood Pressure": (50, 260),
    "Diastolic Blood Pressure": (30, 160),
    "Heart rate": (20, 250),
    "Respiratory rate": (4, 60),
    "Body mass index (BMI) [Ratio]": (10, 80),
    "Body Weight": (2, 300),          # kg
    "Pain severity - 0-10 verbal numeric rating [Score] - Reported": (0, 10),
    "Patient Health Questionnaire 2 item (PHQ-2) total score [Reported]": (0, 6),
}
# Body Height tidak diberi rentang karena satuannya campur (0.16 vs 1860). Lihat log.

# Tabel riwayat: dikaitkan lewat patient_id, dipotong pada cutoff.
HIST = {                              # nama tabel: (kolom kode, awalan nama fitur)
    "conditions": ("condition_code", "cond"),
    "medications": ("medication_code", "med"),
    "procedures": ("procedure_code", "proc"),
    "immunizations": ("immunization_code", "imm"),
    "allergies": ("allergy_code", "alg"),
}

CATEGORICAL = ["gender", "race", "ethnicity", "state", "marital_at_visit"]  # kolom kategori untuk LightGBM
TRIAGE_PREFIXES = ("vital__", "sym__", "sym_all_", "sev__", "sev_all_", "cc__")  # fitur dari triase (untuk uji ablasi)


# =============================================================================
# LANGKAH 1: UNIVERSE (satu baris per encounter yang diprediksi)
# =============================================================================
def build_universe(t):
    """Ambil tabel indeks dan turunkan cutoff = prediction_time dalam resolusi detik."""
    idx = t["index_encounters"][["patient_id", "encounter_id", "prediction_time"]].copy()  # kolom yang dibutuhkan
    idx["cutoff"] = rf.parse_ts(idx["prediction_time"]).dt.floor("s")                     # potong ke detik penuh
    assert idx["cutoff"].notna().all(), "ada prediction_time yang gagal diparse"            # pengaman
    assert idx["encounter_id"].is_unique and idx["patient_id"].is_unique                  # satu encounter per pasien
    return idx                                                                            # kembalikan universe


# =============================================================================
# LANGKAH 2: DEMOGRAFI (level pasien, tanpa timestamp)
# =============================================================================
def demographics(t, u, log):
    """Usia, gender, ras, dll. Pendapatan di-parse. city & county dibuang (kardinalitas tinggi)."""
    p = t["patients"].copy()                                                # salinan tabel pasien
    p["income"] = p["income_at_visit"].map(rf.parse_amount)                  # parse format campuran menjadi angka
    p["log_income"] = np.log1p(p["income"])                                  # log agar skala besar tidak mendominasi
    log.append({"table": "patients", "check": "income_unparsed", "count": int(p["income"].isna().sum() - p["income_at_visit"].isna().sum())})
    log.append({"table": "patients", "decision": "city & county dibuang: kardinalitas tinggi, rawan overfit"})
    keep = ["patient_id", "age_at_visit", "log_income"] + CATEGORICAL     # kolom yang dipakai
    p = p[keep].merge(u[["patient_id"]], on="patient_id")                    # batasi ke pasien di universe
    return p.set_index("patient_id"), "patient", "other"                     # level pasien, jenis fitur 'other'


# =============================================================================
# LANGKAH 3: TRIASE (level encounter, timestamp hampir sama dengan cutoff)
# =============================================================================
def convert_vital_units(v, log):
    """Samakan satuan: SpO2 'fraction' -> persen, tekanan darah 'kPa' -> mmHg."""
    v = v.copy()                                                              # jangan ubah input
    m = (v["vital_name"] == "SpO2") & (v["unit"] == "fraction")               # saturasi yang ditulis pecahan
    log.append({"table": "triage_vitals", "check": "SpO2 fraction -> persen", "rows": int(m.sum())})
    v.loc[m, "value"] = v.loc[m, "value"] * 100                               # 0.96 -> 96
    m = v["vital_name"].isin(["Systolic BP", "Diastolic BP"]) & (v["unit"] == "kPa")  # tekanan dalam kPa
    log.append({"table": "triage_vitals", "check": "BP kPa -> mmHg", "rows": int(m.sum())})
    v.loc[m, "value"] = v.loc[m, "value"] * 7.50062                           # 1 kPa = 7,50062 mmHg
    return v


def triage(t, u, log):
    """Fitur triase: vital, gejala, severity, dan keluhan utama. Semuanya level encounter."""
    out = []                                                                  # wadah hasil (df, level, jenis)

    # --- vital signs ---
    v = rf.attach_cutoff(t["triage_vitals"], u, "encounter_id")               # pasang cutoff lewat encounter
    v = rf.apply_cutoff(v, "measured_at", log, "triage_vitals")               # simpan yang tersedia pada cutoff
    v["value"] = rf.to_float_decimal_comma(v["value"])                        # '18,33' -> 18.33
    v = convert_vital_units(v, log)                                           # samakan satuan
    v = rf.apply_valid_range(v, "vital_name", "value", VITAL_RANGES, log, "triage_vitals")  # buang nilai mustahil
    out.append((rf.agg_numeric_by_code(v, "encounter_id", "vital_name", "value", "vital",
                                       set(v["vital_name"].unique()), stats=("last",)), "encounter", "num"))

    # --- gejala (ada/tidak) ---
    s = rf.attach_cutoff(t["triage_symptoms"], u, "encounter_id")
    s = rf.apply_cutoff(s, "observed_at", log, "triage_symptoms")
    vocab = rf.vocab_from(s, "symptom_concept", MIN_COUNT)                    # gejala yang cukup sering
    out.append((rf.agg_code_bag(s, "encounter_id", "symptom_concept", "sym", vocab), "encounter", "bag"))
    out.append((rf.agg_recency(s, "encounter_id", "sym_all"), "encounter", "count"))

    # --- severity (skor 0-100 per gejala) ---
    sv = rf.attach_cutoff(t["triage_symptom_severity"], u, "encounter_id")
    sv = rf.apply_cutoff(sv, "observed_at", log, "triage_symptom_severity")
    sv["severity_score"] = pd.to_numeric(sv["severity_score"], errors="coerce")
    vocab = rf.vocab_from(sv[sv["severity_score"].notna()], "symptom_concept", MIN_COUNT)
    out.append((rf.agg_numeric_by_code(sv, "encounter_id", "symptom_concept", "severity_score",
                                       "sev", vocab, stats=("max",)), "encounter", "num"))
    g = sv.groupby("encounter_id")["severity_score"]                          # agregat keseluruhan
    out.append((pd.DataFrame({"sev_all_max": g.max(), "sev_all_mean": g.mean()}), "encounter", "num"))

    # --- keluhan utama (teks bebas) ---
    pc = t["presenting_complaints"]                                           # tidak punya timestamp
    pc = pc[pc["encounter_id"].isin(u["encounter_id"])]                       # pastikan encounter ada di universe
    log.append({"table": "presenting_complaints",
                "decision": "tanpa timestamp: dianggap dicatat saat triase (encounter indeks). Asumsi, lihat log."})
    tok = rf.explode_tokens(pc, "encounter_id", "presenting_complaints_text",
                            sep=",", strip_prefix="Reports:")                 # 'Reports: Fever, Throat' -> token
    vocab = rf.vocab_from(tok, "token", MIN_COUNT)
    out.append((rf.agg_code_bag(tok, "encounter_id", "token", "cc", vocab), "encounter", "bag"))
    return out


# =============================================================================
# LANGKAH 4: RIWAYAT MEDIS (level pasien, dipotong pada cutoff)
# =============================================================================
def history(t, u, log):
    """Kondisi, obat, prosedur, imunisasi, alergi: jumlah, jarak waktu, dan kode yang ada."""
    out = []
    for name, (code, pre) in HIST.items():                                    # ulangi untuk tiap tabel riwayat
        h = rf.attach_cutoff(t[name], u, "patient_id")                        # pasang cutoff lewat pasien
        h = rf.apply_cutoff(h, "recorded_at", log, name)                      # buang yang setelah cutoff
        vocab = rf.vocab_from(h, code, MIN_COUNT)                             # kode yang cukup sering
        out.append((rf.agg_recency(h, "patient_id", pre), "patient", "count"))
        out.append((rf.agg_code_bag(h, "patient_id", code, pre, vocab), "patient", "bag"))
    return out


def previous_encounters(t, u, log):
    """Riwayat kunjungan sebelumnya: jumlah, jarak waktu, durasi rata-rata, dan jenisnya."""
    pe = rf.attach_cutoff(t["previous_encounter"], u, "patient_id")
    pe = rf.apply_cutoff(pe, "history_start", log, "previous_encounter")      # aturan utama: waktu mulai
    stop = rf.parse_ts(pe["history_stop"])                                    # waktu selesai
    log.append({"table": "previous_encounter", "check": "history_stop melewati cutoff",
                "count": int((stop >= pe["cutoff"] + rf.ONE_SECOND).sum())})  # pengaman kedua
    pe["dur_days"] = (stop[pe.index] - pe["_ts"]).dt.total_seconds() / 86400   # durasi kunjungan dalam hari
    out = [(rf.agg_recency(pe, "patient_id", "prev"), "patient", "count")]
    out.append((pe.groupby("patient_id")["dur_days"].mean().rename("prev_dur_days_mean").to_frame(), "patient", "num"))
    vocab = rf.vocab_from(pe, "history_type_code", MIN_COUNT)
    out.append((rf.agg_code_bag(pe, "patient_id", "history_type_code", "prev", vocab), "patient", "bag"))
    return out


# =============================================================================
# LANGKAH 5: OBSERVASI (lab dan survei, level pasien)
# =============================================================================
def observations(t, u, log):
    """Numerik per kode (last, mean) dan teks sebagai token 'kode=nilai'."""
    out = []
    ob = rf.attach_cutoff(t["observations"], u, "patient_id")
    ob = rf.apply_cutoff(ob, "observed_at", log, "observations")
    out.append((rf.agg_recency(ob, "patient_id", "obs"), "patient", "count"))

    num = ob[ob["value_type"] == "numeric"].copy()                            # bagian numerik
    num["value"] = rf.to_float_decimal_comma(num["value"])                    # koma desimal -> titik
    num = rf.apply_valid_range(num, "observation_name", "value", OBS_RANGES, log, "observations")
    numok = num[num["value"].notna()]                                         # hanya nilai yang valid
    vocab = rf.vocab_from(numok, "observation_code", MIN_COUNT)
    out.append((rf.agg_numeric_by_code(numok, "patient_id", "observation_code", "value", "obsnum",
                                       vocab, stats=("last", "mean")), "patient", "num"))

    txt = ob[(ob["value_type"] == "text") & ob["value"].notna()].copy()       # bagian teks
    txt["token"] = txt["observation_code"].astype(str) + "=" + txt["value"].astype(str).str.strip().str.lower()
    vocab = rf.vocab_from(txt, "token", MIN_COUNT)
    out.append((rf.agg_code_bag(txt, "patient_id", "token", "obstxt", vocab), "patient", "bag"))
    return out


# =============================================================================
# LANGKAH 6: RAKIT, VALIDASI, DAN SANITY CHECK
# =============================================================================
def assemble(u, parts):
    """Gabungkan semua bagian ke level encounter, lalu isi nilai kosong sesuai jenis fitur."""
    frames, bag_or_count = [], []                                             # daftar bagian dan kolom yang diisi 0
    for df, level, kind in parts:
        aligned = rf.to_entity_level(df, u, level)                            # semua ke level encounter
        frames.append(aligned)
        if kind in ("bag", "count"):                                          # bag/count: tidak ada = 0
            bag_or_count.extend(aligned.columns)
    feat = pd.concat(frames, axis=1)                                          # gabung kolom berdasarkan index
    feat = feat.loc[:, ~feat.columns.duplicated()]                            # buang kolom kembar
    feat = feat.reindex(u["encounter_id"].values)                             # urutkan dan pastikan semua encounter ada
    feat[[c for c in bag_or_count if c in feat.columns]] = feat[[c for c in bag_or_count if c in feat.columns]].fillna(0)
    feat.index.name = "encounter_id"
    return feat.reset_index().copy()                       # reset index jadi kolom encounter_id


def sanity_cv(train_feat, drop_prefixes=()):
    """5-fold CV pada train. Hanya untuk sanity check, BUKAN untuk memilih model final."""
    from sklearn.model_selection import StratifiedKFold   # pembagian fold yang menjaga proporsi kelas
    from sklearn.metrics import f1_score                  # Macro F1
    import lightgbm as lgb                                # model boosting pohon

    cols = [c for c in train_feat.columns
            if c not in ("sample_id", "patient_id", "encounter_id", "target_class")
            and not c.startswith(tuple(drop_prefixes))]   # buang fitur sesuai ablasi
    X = train_feat[cols].copy()                           # matriks fitur
    for c in X.columns:                                   # kolom non-angka (object/string) -> category
        if not pd.api.types.is_numeric_dtype(X[c]):       # cek tipe: angka dibiarkan, teks jadi kategori
            X[c] = X[c].astype("category")
    y = train_feat["target_class"].astype("category")     # label sebagai kategori
    yc = y.cat.codes                                      # ubah label ke angka 0..3
    oof = np.zeros((len(X), len(y.cat.categories)))       # prediksi out-of-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5 fold, acak tapi bisa direproduksi
    for tr, va in skf.split(X, yc):                       # ulangi untuk tiap fold
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               class_weight="balanced", random_state=42, verbose=-1)  # bobot kelas seimbang
        m.fit(X.iloc[tr], yc.iloc[tr])                    # latih di fold train
        oof[va] = m.predict_proba(X.iloc[va])             # prediksi di fold validasi
    return f1_score(yc, oof.argmax(1), average="macro"), len(cols)  # Macro F1 dan jumlah fitur


def main(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)                   # buat folder output jika belum ada
    log = []                                              # semua keputusan dan angka dikumpulkan di sini
    t = rf.load_tables(root)                              # 1. baca semua tabel
    u = build_universe(t)                                 # 2. universe + cutoff

    parts = [demographics(t, u, log)]                     # 3. kumpulkan semua bagian fitur
    parts += triage(t, u, log)
    parts += history(t, u, log)
    parts += previous_encounters(t, u, log)
    parts += observations(t, u, log)

    for name, col in [("triage_vitals", "measured_at"), ("triage_symptoms", "observed_at")]:  # diagnostik: dampak mode strict
        d = rf.attach_cutoff(t[name], u, "encounter_id")  # pasang cutoff
        rf.apply_cutoff(d, col, log, name, strict=True)    # hanya catat ke log, tidak dipakai untuk fitur

    feat = assemble(u, parts)                             # 4. rakit tabel fitur per encounter
    feat = rf.clean_col_names(feat.copy())                # 5. nama kolom aman untuk LightGBM (copy: hindari fragmentasi)
    feat = feat.assign(patient_id=u.set_index("encounter_id").loc[feat["encounter_id"], "patient_id"].values)  # tambah patient_id sekali

    for name in ("train", "test"):                        # 6. pasang fitur ke train dan test
        df = t[name].merge(feat, on="encounter_id", how="left", validate="one_to_one")
        assert (df["patient_id_x"] == df["patient_id_y"]).all(), f"{name}: patient_id tidak cocok"
        df = df.drop(columns=["patient_id_y"]).rename(columns={"patient_id_x": "patient_id"})
        df.to_csv(os.path.join(out_dir, f"{name}_features.csv"), index=False)
        if name == "train":
            train_feat = df
        log.append({"table": name, "check": "baris", "rows": int(len(df)), "missing_feature_rows": int(df.drop(columns=["sample_id","patient_id","encounter_id"],errors="ignore").isna().all(axis=1).sum())})

    feat.to_csv(os.path.join(out_dir, "features_all.csv"), index=False)  # tabel fitur lengkap

    # 7. sanity check: dengan triase vs tanpa triase (membuktikan dampak keputusan triase)
    f_full, n_full = sanity_cv(train_feat)
    f_notri, n_notri = sanity_cv(train_feat, drop_prefixes=TRIAGE_PREFIXES)
    log.append({"check": "sanity_cv_macro_f1", "with_triage": round(f_full, 4), "features_with": n_full,
                "without_triage": round(f_notri, 4), "features_without": n_notri})

    with open(os.path.join(out_dir, "log_keputusan.json"), "w") as f:  # simpan log
        json.dump(log, f, indent=2, default=str)
    print("\n".join(str(x) for x in log))                  # tampilkan log di layar
    print(f"\nFitur per encounter: {feat.shape[1]-2} fitur (+ encounter_id dan patient_id), {feat.shape[0]} baris")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()                        # parser argumen
    ap.add_argument("--root", required=True)              # folder data hasil ekstrak
    ap.add_argument("--out", required=True)               # folder output
    a = ap.parse_args()
    main(a.root, a.out)                                   # jalankan pipeline

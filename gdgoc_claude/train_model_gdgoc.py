"""
train_model_gdgoc.py
====================
Tahap pemodelan GDGoC UGM Academic Hacker 2026 di atas hasil/train_features.csv.

Urutan kerja:
  1. Siapkan matriks fitur (kolom kategori disamakan antara train dan test).
  2. Baseline Logistic Regression (pembanding).
  3. Pencarian hyperparameter LightGBM dengan 5-fold CV, dipilih berdasarkan log-loss.
  4. Tuning bobot probabilitas per kelas di atas out-of-fold untuk memaksimalkan Macro F1,
     divalidasi dengan split-half agar tidak optimistik.
  5. Latih ulang di seluruh train (3 seed), prediksi test, buat submission.

Jalankan:
    python train_model_gdgoc.py --feat "hasil" --data "/home/claude/data" --out "model"
"""
import argparse                                      # membaca argumen command line
import json                                          # menyimpan log dan parameter
import os                                            # membuat folder output
import time                                          # mengukur lama pencarian

import lightgbm as lgb                               # model boosting pohon
import numpy as np                                   # operasi numerik
import pandas as pd                                  # tabel
from sklearn.impute import SimpleImputer             # isi nilai kosong untuk LR
from sklearn.linear_model import LogisticRegression  # baseline linear
from sklearn.metrics import classification_report, f1_score, log_loss  # metrik
from sklearn.model_selection import StratifiedKFold  # CV yang menjaga proporsi kelas
from sklearn.pipeline import make_pipeline           # rangkai imputasi, skala, model
from sklearn.preprocessing import StandardScaler     # standarisasi untuk LR

ID_COLS = ["sample_id", "patient_id", "encounter_id"]  # kolom identitas, bukan fitur
TARGET = "target_class"                                # nama kolom label
SEED = 42                                              # seed utama agar hasil bisa direproduksi
N_SPLITS = 5                                           # jumlah fold CV
N_TRIALS = 12                                          # jumlah konfigurasi hyperparameter yang dicoba


# =============================================================================
# 1. PERSIAPAN DATA
# =============================================================================
def prepare(train, test):
    """Pisahkan fitur dan label, lalu samakan tipe kolom kategori di train dan test."""
    X = train.drop(columns=ID_COLS + [TARGET])                  # matriks fitur train
    Xt = test.drop(columns=ID_COLS)                             # matriks fitur test (tanpa label)
    classes = sorted(train[TARGET].unique())                    # urutan kelas tetap dan alfabetis
    y = train[TARGET].map({c: i for i, c in enumerate(classes)})  # label diubah ke angka 0..K-1
    cats = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("str")]  # kolom teks
    for c in cats:                                              # setiap kolom teks
        levels = sorted(set(X[c].dropna()) | set(Xt[c].dropna()))  # kategori gabungan train dan test
        cat_type = pd.CategoricalDtype(levels)                  # tipe kategori dengan daftar yang sama
        X[c] = X[c].astype(cat_type)                            # terapkan ke train
        Xt[c] = Xt[c].astype(cat_type)                          # terapkan ke test agar kode sama
    assert set(X.columns) == set(Xt.columns), "kolom fitur train dan test berbeda"  # pengaman: kolom harus sama
    Xt = Xt[X.columns]                                          # samakan urutan kolom test dengan train
    return X, Xt, y.values, classes, cats


def class_weights(y, K):
    """Bobot seimbang per baris: kelas jarang mendapat bobot lebih besar. Rumus: n / (K * n_kelas)."""
    counts = np.bincount(y, minlength=K)                        # jumlah baris tiap kelas
    per_class = len(y) / (K * counts)                           # bobot per kelas
    return per_class[y]                                         # bobot per baris


# =============================================================================
# 2. BASELINE LOGISTIC REGRESSION
# =============================================================================
def logistic_baseline(X, y, cats, K):
    """LR dengan one-hot untuk kategori, imputasi median, dan standarisasi. Dipakai sebagai pembanding."""
    Xd = pd.get_dummies(X, columns=cats, dummy_na=True, dtype=float)   # kategori -> kolom biner
    model = make_pipeline(                                              # rangkai langkah secara berurutan
        SimpleImputer(strategy="median"),                               # isi NaN dengan median (di dalam fold)
        StandardScaler(),                                               # samakan skala fitur
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000),  # linear dengan bobot kelas seimbang
    )
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)  # fold yang sama dengan LightGBM
    oof = np.zeros((len(Xd), K))                                        # prediksi out-of-fold
    for tr, va in skf.split(Xd, y):                                     # tiap fold
        model.fit(Xd.iloc[tr], y[tr])                                   # latih di fold train
        oof[va] = model.predict_proba(Xd.iloc[va])                      # prediksi di fold validasi
    return oof


# =============================================================================
# 3. LIGHTGBM: CV DENGAN EARLY STOPPING
# =============================================================================
def cv_lgb(X, y, cfg, K, cats):
    """5-fold CV. Setiap fold dihentikan dini dengan log-loss validasi. Mengembalikan OOF dan iterasi terbaik."""
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)  # pembagian fold tetap
    oof = np.zeros((len(X), K))                                         # wadah prediksi out-of-fold
    best_iters = []                                                     # iterasi terbaik tiap fold
    params = dict(objective="multiclass", num_class=K, metric="multi_logloss",
                  verbosity=-1, seed=SEED, **cfg)                       # parameter model
    for tr, va in skf.split(X, y):                                      # tiap fold
        dtr = lgb.Dataset(X.iloc[tr], y[tr], weight=class_weights(y[tr], K),  # data latih dengan bobot kelas
                          categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr,             # validasi tanpa bobot: mencerminkan distribusi asli
                          categorical_feature=cats, free_raw_data=False)
        booster = lgb.train(params, dtr, num_boost_round=800,           # maksimal 800 pohon
                            valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50, verbose=False)])  # berhenti jika 50 putaran tanpa membaik
        oof[va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)  # prediksi pada iterasi terbaik
        best_iters.append(booster.best_iteration)                       # simpan iterasi terbaik
    return oof, best_iters


def random_configs(n, rng):
    """Ambil n konfigurasi acak dari ruang pencarian. Konfigurasi pertama selalu default yang wajar."""
    space = {
        "num_leaves": [15, 31, 63],                  # kompleksitas pohon
        "learning_rate": [0.03, 0.05, 0.1],          # langkah belajar
        "min_child_samples": [10, 20, 40],           # sampel minimum per daun, melawan overfit
        "feature_fraction": [0.5, 0.7, 0.9],         # bagian fitur yang dipakai tiap pohon
        "bagging_fraction": [0.6, 0.8, 1.0],         # bagian baris yang dipakai tiap pohon
        "lambda_l2": [0.0, 1.0, 5.0],                # regularisasi L2
    }
    configs = [dict(num_leaves=31, learning_rate=0.05, min_child_samples=20,
                    feature_fraction=0.7, bagging_fraction=0.8, lambda_l2=1.0, bagging_freq=1)]
    for _ in range(n - 1):                           # sisanya acak
        c = {k: v[rng.integers(len(v))] for k, v in space.items()}  # pilih satu nilai per parameter
        c["bagging_freq"] = 1                        # bagging aktif
        configs.append({k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()})
    return configs


# =============================================================================
# 4. TUNING BOBOT KELAS UNTUK MACRO F1
# =============================================================================
GRID = np.exp(np.linspace(np.log(0.5), np.log(2.0), 21))  # kandidat pengali per kelas (0,5 sampai 2)


def macro_f1(P, y, w):
    """Macro F1 dari probabilitas P yang dikalikan bobot w, lalu argmax."""
    return f1_score(y, (P * w).argmax(1), average="macro")


def tune_weights(P, y, K, passes=3):
    """Coordinate ascent: ubah satu bobot kelas dalam grid, pertahankan jika Macro F1 naik."""
    w = np.ones(K)                                   # mulai dari bobot seragam
    best = macro_f1(P, y, w)                         # skor awal
    for _ in range(passes):                          # beberapa putaran agar saling memengaruhi
        for k in range(K):                           # tiap kelas
            for g in GRID:                           # tiap kandidat pengali
                w2 = w.copy()                        # salinan supaya w asli tidak rusak
                w2[k] = g                            # ganti bobot kelas k
                s = macro_f1(P, y, w2)               # hitung skor
                if s > best + 1e-9:                  # hanya terima jika benar-benar naik
                    best, w = s, w2
    return w, best


def split_half_check(P, y, K, seed=SEED):
    """Validasi bobot: tune di satu separuh OOF, uji di separuh lain, lalu tukar. Mengukur apakah tuning itu nyata."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))                    # acak urutan baris
    halves = [idx[: len(y) // 2], idx[len(y) // 2:]]  # dua separuh
    results = []                                     # hasil tiap arah
    for a, b in [(0, 1), (1, 0)]:                    # tune di a, uji di b; lalu sebaliknya
        w, _ = tune_weights(P[halves[a]], y[halves[a]], K)   # bobot dari separuh a
        base = macro_f1(P[halves[b]], y[halves[b]], np.ones(K))  # tanpa tuning, di separuh b
        tuned = macro_f1(P[halves[b]], y[halves[b]], w)           # dengan tuning, di separuh b
        results.append({"tuned_on": int(a), "test_on": int(b), "untuned_f1": round(base, 4), "tuned_f1": round(tuned, 4)})
    return results


# =============================================================================
# 5. PIPELINE UTAMA
# =============================================================================
def main(feat_dir, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)              # buat folder output
    log = {}                                         # semua angka dan keputusan disimpan di sini
    train = pd.read_csv(os.path.join(feat_dir, "train_features.csv"))  # fitur train
    test = pd.read_csv(os.path.join(feat_dir, "test_features.csv"))    # fitur test
    X, Xt, y, classes, cats = prepare(train, test)   # 1. siapkan data
    K = len(classes)                                 # jumlah kelas (4)
    log["classes"] = classes                         # urutan kelas untuk submission
    log["n_features"] = X.shape[1]                   # jumlah fitur
    log["categorical"] = cats                        # kolom kategori

    # --- 2. baseline LR ---
    t0 = time.time()
    oof_lr = logistic_baseline(X, y, cats, K)        # prediksi out-of-fold LR
    log["baseline_lr"] = {"macro_f1": round(f1_score(y, oof_lr.argmax(1), average="macro"), 4),
                          "logloss": round(log_loss(y, oof_lr), 4)}

    # --- 3. pencarian hyperparameter LightGBM ---
    rng = np.random.default_rng(SEED)                # generator acak untuk pencarian
    rows = []                                        # hasil tiap konfigurasi
    best = None                                      # konfigurasi terbaik
    for i, cfg in enumerate(random_configs(N_TRIALS, rng)):   # coba konfigurasi satu per satu
        oof, iters = cv_lgb(X, y, cfg, K, cats)      # CV untuk konfigurasi ini
        ll = log_loss(y, oof)                        # log-loss OOF (pemilih utama)
        f1 = f1_score(y, oof.argmax(1), average="macro")  # Macro F1 OOF (hanya dicatat)
        rows.append({**cfg, "trial": i, "oof_logloss": round(ll, 5), "oof_macro_f1": round(f1, 4),
                     "mean_best_iter": int(np.mean(iters))})
        print(f"trial {i:2d} logloss={ll:.4f} f1={f1:.4f} iter={int(np.mean(iters))}", flush=True)
        if best is None or ll < best["ll"]:          # simpan yang log-loss-nya terkecil
            best = {"cfg": cfg, "ll": ll, "oof": oof, "iters": iters, "f1": f1}
    pd.DataFrame(rows).sort_values("oof_logloss").to_csv(os.path.join(out_dir, "cv_search_results.csv"), index=False)
    log["search_time_sec"] = round(time.time() - t0, 1)
    log["best_params"] = best["cfg"]
    log["best_oof_logloss"] = round(best["ll"], 5)
    log["best_oof_macro_f1_argmax"] = round(best["f1"], 4)
    log["best_mean_iter"] = int(np.mean(best["iters"]))

    # --- 4. bobot kelas untuk Macro F1 ---
    P = best["oof"]                                  # probabilitas OOF terbaik
    w, tuned_f1 = tune_weights(P, y, K)              # tuning bobot di seluruh OOF
    log["class_weights"] = dict(zip(classes, np.round(w, 4).tolist()))
    log["oof_macro_f1_tuned_in_sample"] = round(tuned_f1, 4)  # optimistik, hanya untuk referensi
    log["split_half_check"] = split_half_check(P, y, K)      # uji yang lebih jujur

    # --- 5. latih ulang di seluruh train, 3 seed, lalu prediksi test ---
    n_rounds = int(np.mean(best["iters"]) * 1.1)     # iterasi final: rata-rata fold dinaikkan 10% karena data lebih banyak
    log["final_rounds"] = n_rounds
    Pt = np.zeros((len(Xt), K))                      # prediksi test rata-rata dari 3 seed
    imp = np.zeros(X.shape[1])                       # kepentingan fitur rata-rata
    params = dict(objective="multiclass", num_class=K, verbosity=-1, **best["cfg"])
    for seed in [1, 2, 3]:                           # tiga seed mengurangi variasi acak
        d = lgb.Dataset(X, y, weight=class_weights(y, K), categorical_feature=cats, free_raw_data=False)
        m = lgb.train({**params, "seed": seed}, d, num_boost_round=n_rounds)  # latih di seluruh train
        Pt += m.predict(Xt) / 3                      # rata-rata probabilitas
        imp += m.feature_importance("gain") / 3      # rata-rata kepentingan fitur
    pd.DataFrame(Pt, columns=classes).assign(sample_id=test["sample_id"].values)[["sample_id"] + classes] \
        .to_csv(os.path.join(out_dir, "test_probabilities.csv"), index=False)  # probabilitas test per kelas
    pred = np.array(classes)[(Pt * w).argmax(1)]     # kelas dengan probabilitas tertimbang tertinggi
    sub = pd.DataFrame({"sample_id": test["sample_id"].values, "target_class": pred})
    if data_dir:                                                      # jika folder data diberikan, ikuti urutan sample_submission
        ref = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))[["sample_id"]]
        sub = ref.merge(sub, on="sample_id", how="left", validate="one_to_one")  # urutan sama persis dengan contoh Kaggle
        assert sub["target_class"].notna().all(), "ada sample_id yang tidak punya prediksi"
    sub.to_csv(os.path.join(out_dir, "submission.csv"), index=False)  # file untuk diunggah ke Kaggle
    log["test_pred_distribution"] = sub["target_class"].value_counts().to_dict()
    log["train_distribution"] = train[TARGET].value_counts(normalize=True).round(4).to_dict()
    pd.DataFrame({"feature": X.columns, "gain": imp}).sort_values("gain", ascending=False) \
        .to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    # --- ringkasan OOF untuk laporan ---
    pred_oof = (P * w).argmax(1)                     # prediksi OOF dengan bobot
    log["oof_classification_report"] = classification_report(y, pred_oof, target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame({"sample_id": train["sample_id"].values, "target_class": np.array(classes)[y],
                  **{f"p_{c}": P[:, i] for i, c in enumerate(classes)}}).to_csv(os.path.join(out_dir, "oof_probabilities.csv"), index=False)

    with open(os.path.join(out_dir, "model_log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)     # simpan semua hasil
    print(json.dumps({k: log[k] for k in ["baseline_lr", "best_params", "best_oof_logloss",
                                          "best_oof_macro_f1_argmax", "class_weights",
                                          "oof_macro_f1_tuned_in_sample", "split_half_check",
                                          "final_rounds", "test_pred_distribution"]}, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()                   # parser argumen
    ap.add_argument("--feat", required=True)         # folder berisi train_features.csv dan test_features.csv
    ap.add_argument("--data", required=False, default="")  # folder data asli (tidak dipakai, disiapkan untuk tahap lanjut)
    ap.add_argument("--out", required=True)          # folder output model
    a = ap.parse_args()
    main(a.feat, a.data, a.out)

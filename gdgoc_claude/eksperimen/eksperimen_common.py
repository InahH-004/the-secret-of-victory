"""
eksperimen_common.py
====================
Helper bersama untuk eksperimen ensemble dan hyperparameter.

Fitur TIDAK dibangun ulang di sini. Sel-sel kode dari notebook utama
(GDGoC_UGM_Academic_Hacker_2026.ipynb) dijalankan sampai tepat sebelum
bagian pemodelan, sehingga fiturnya selalu identik dengan notebook.

Taruh folder ini di dalam folder yang sama dengan notebook, misalnya:
    gdgoc_claude/
        GDGoC_UGM_Academic_Hacker_2026.ipynb
        eksperimen/
            eksperimen_common.py
            exp_ensemble.py
            exp_features_hp.py
"""
import os
import matplotlib
matplotlib.use("Agg")               # tidak membuka jendela plot saat sel notebook dijalankan

import nbformat
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
NB_NAME = "GDGoC_UGM_Academic_Hacker_2026.ipynb"
STOP_MARKER = "X = train.drop(columns=ID_COLS + [TARGET])"   # sel pemodelan pertama di notebook

CFG_BASE = dict(num_leaves=31, learning_rate=0.03, min_child_samples=20, feature_fraction=0.5,
                bagging_fraction=1.0, lambda_l2=5.0, bagging_freq=1)
GRID = np.exp(np.linspace(np.log(0.5), np.log(2.0), 21))


def find_notebook():
    """Cari notebook di folder eksperimen, lalu di folder induknya."""
    for p in [os.path.join(HERE, NB_NAME), os.path.join(HERE, "..", NB_NAME), os.path.join(os.getcwd(), NB_NAME)]:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"{NB_NAME} tidak ditemukan. Pastikan folder eksperimen berada di samping notebook.")


def load_features():
    """Jalankan sel notebook sampai sebelum pemodelan. Mengembalikan namespace berisi train, test, sv, dll."""
    nb_path = find_notebook()
    nb = nbformat.read(nb_path, as_version=4)
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(nb_path))       # notebook mencari folder data berdasarkan cwd
    ns = {"__name__": "__notebook__"}
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            if STOP_MARKER in cell.source:
                break
            exec(compile(cell.source, "<cell>", "exec"), ns)
    finally:
        os.chdir(old_cwd)
    return ns


def encode_target(train, classes):
    return train["target_class"].map({c: i for i, c in enumerate(classes)}).values


def prep_matrix(train, test, extra_cols_train=None, extra_cols_test=None):
    """Matriks fitur X (train) dan Xt (test), dengan kolom kategori yang disamakan."""
    id_cols = ["sample_id", "patient_id", "encounter_id", "target_class"]
    X = train.drop(columns=id_cols).copy()
    Xt = test.drop(columns=["sample_id", "patient_id", "encounter_id"]).copy()
    cats = [c for c in X.columns if str(X[c].dtype) == "object" or str(X[c].dtype).startswith("str")]
    for c in cats:
        dt = pd.CategoricalDtype(sorted(set(X[c].dropna()) | set(Xt[c].dropna())))
        X[c] = X[c].astype(dt)
        Xt[c] = Xt[c].astype(dt)
    Xt = Xt[X.columns]
    return X, Xt, cats


def oof_lgb(X, y, cats, K, cfg=CFG_BASE, seed=42, max_rounds=2000, early_stop=100):
    """OOF probabilitas LightGBM dengan 5-fold. Mengembalikan (oof, daftar iterasi terbaik)."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    oof = np.zeros((len(X), K))
    iters = []
    params = dict(objective="multiclass", num_class=K, metric="multi_logloss", verbosity=-1,
                  seed=42, **cfg)
    for tr, va in folds:
        cw = len(tr) / (K * np.bincount(y[tr], minlength=K))           # bobot kelas seimbang
        dtr = lgb.Dataset(X.iloc[tr], y[tr], weight=cw[y[tr]], categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr, categorical_feature=cats, free_raw_data=False)
        b = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stop, verbose=False)])
        oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration)
        iters.append(b.best_iteration)
    return oof, iters


def tune_weights(P, y, K, passes=3):
    """Coordinate ascent untuk bobot per kelas yang memaksimalkan Macro F1."""
    w = np.ones(K)
    best = f1_score(y, (P * w).argmax(1), average="macro")
    for _ in range(passes):
        for k in range(K):
            for g in GRID:
                w2 = w.copy()
                w2[k] = g
                s = f1_score(y, (P * w2).argmax(1), average="macro")
                if s > best + 1e-9:
                    best, w = s, w2
    return w, best


def split_half_f1(P, y, K, seed=42):
    """Rata-rata Macro F1 pada separuh yang TIDAK dipakai untuk menuning bobot (jujur)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    h = [perm[: len(y) // 2], perm[len(y) // 2:]]
    scores = []
    for a, b in [(0, 1), (1, 0)]:
        w, _ = tune_weights(P[h[a]], y[h[a]], K)
        scores.append(f1_score(y[h[b]], (P[h[b]] * w).argmax(1), average="macro"))
    return float(np.mean(scores))


def metrics(P, y, K, seed=42):
    return {"logloss": log_loss(y, P),
            "f1_argmax": f1_score(y, P.argmax(1), average="macro"),
            "f1_split": split_half_f1(P, y, K, seed=seed)}

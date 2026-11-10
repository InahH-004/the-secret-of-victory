"""
exp_ensemble.py
===============
Membandingkan LightGBM tunggal dengan kombinasi LightGBM + Logistic Regression + HistGradientBoosting.
Setiap model dievaluasi dengan 3 kali pengacakan fold (repeated CV), bukan dengan public leaderboard.

Jalankan (dari dalam folder eksperimen):
    python exp_ensemble.py            # penuh, sekitar 10 sampai 20 menit
    python exp_ensemble.py --quick    # cek cepat, 1 pengulangan
Hasil: ensemble_summary.csv dan oof_ensemble.npz di folder ini.
"""
import sys, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eksperimen_common import load_features, encode_target, prep_matrix, oof_lgb, metrics, HERE

QUICK = "--quick" in sys.argv
REPEATS = 1 if QUICK else 3

t0 = time.time()
print("Membangun fitur dari notebook...", flush=True)
ns = load_features()
train, test = ns["train"], ns["test"]
classes = sorted(train["target_class"].unique())
K = len(classes)
y = encode_target(train, classes)
X, Xt, cats = prep_matrix(train, test)
print(f"Fitur siap: X={X.shape}, kolom kategori={len(cats)} ({time.time()-t0:.0f} detik)", flush=True)

# Versi numerik (one-hot) untuk model sklearn
Xd = pd.get_dummies(X, columns=cats, dummy_na=True, dtype=float)
Xd = Xd.loc[:, Xd.nunique(dropna=True) > 1]          # buang kolom konstan (HistGB menolaknya)


def oof_sklearn(make_model, Xm, seed):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xm, y))
    oof = np.zeros((len(Xm), K))
    for tr, va in folds:
        m = make_model()
        m.fit(Xm.iloc[tr], y[tr])
        oof[va] = m.predict_proba(Xm.iloc[va])
    return oof


def make_lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))


def make_hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced", random_state=42)


rows, saved = [], {}
for r in range(REPEATS):
    seed = 42 + r
    print(f"\n--- pengulangan {r+1}/{REPEATS} (seed {seed}) ---", flush=True)
    P_lgb, iters = oof_lgb(X, y, cats, K, seed=seed)
    print(f"LightGBM selesai, iterasi rata-rata {np.mean(iters):.0f} ({time.time()-t0:.0f} detik)", flush=True)
    P_lr = oof_sklearn(make_lr, Xd, seed)
    print("Logistic Regression selesai", flush=True)
    P_hgb = oof_sklearn(make_hgb, Xd, seed)
    print(f"HistGradientBoosting selesai ({time.time()-t0:.0f} detik)", flush=True)

    candidates = {
        "LGBM": P_lgb,
        "LR": P_lr,
        "HistGB": P_hgb,
        "LGBM 0.7 + LR 0.3": 0.7 * P_lgb + 0.3 * P_lr,
        "LGBM 0.5 + HistGB 0.5": 0.5 * P_lgb + 0.5 * P_hgb,
        "LGBM 0.6 + HistGB 0.2 + LR 0.2": 0.6 * P_lgb + 0.2 * P_hgb + 0.2 * P_lr,
    }
    for name, P in candidates.items():
        rows.append({"repeat": r, "model": name, **metrics(P, y, K, seed=seed)})
        saved[f"r{r}_" + name.replace(" ", "_").replace(".", "").replace("+", "p")] = P

df = pd.DataFrame(rows)
summary = df.groupby("model")[["logloss", "f1_argmax", "f1_split"]].agg(["mean", "std"]).round(4)
summary = summary.sort_values(("f1_split", "mean"), ascending=False)
print("\n=== RINGKASAN (urut dari Macro F1 split-half terbaik) ===")
print(summary.to_string())
summary.to_csv(os.path.join(HERE, "ensemble_summary.csv"))
np.savez(os.path.join(HERE, "oof_ensemble.npz"), y=y, **saved)
print(f"\nSelesai dalam {time.time()-t0:.0f} detik. File: ensemble_summary.csv, oof_ensemble.npz")

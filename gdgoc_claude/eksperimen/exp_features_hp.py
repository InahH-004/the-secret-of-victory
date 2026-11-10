"""
exp_features_hp.py
==================
1. Menguji fitur tambahan dari tingkat keparahan gejala (jumlah gejala parah, jumlah total skor).
2. Mencari hyperparameter LightGBM dengan ruang yang lebih luas (termasuk learning rate lebih kecil,
   regularisasi lebih kuat, dan extra_trees), dievaluasi dengan repeated CV.

Hasil setiap konfigurasi disimpan segera ke hp_search.csv, jadi kalau dihentikan, progres tidak hilang.

Jalankan (dari dalam folder eksperimen):
    python exp_features_hp.py            # penuh, bisa 20 sampai 40 menit
    python exp_features_hp.py --quick    # cek cepat, 2 konfigurasi
"""
import sys, os, time
import numpy as np
import pandas as pd

from eksperimen_common import load_features, encode_target, prep_matrix, oof_lgb, metrics, CFG_BASE, HERE

QUICK = "--quick" in sys.argv
N_CONFIGS = 2 if QUICK else 14
REPEATS = 1 if QUICK else 2
t0 = time.time()

print("Membangun fitur dari notebook...", flush=True)
ns = load_features()
train, test, sv = ns["train"], ns["test"], ns["sv"]
classes = sorted(train["target_class"].unique())
K = len(classes)
y = encode_target(train, classes)

# --- Fitur tambahan dari tingkat keparahan (level encounter) ---
g = sv.groupby("encounter_id")["severity_score"]
extra = pd.DataFrame({
    "sev_n_ge50": sv[sv["severity_score"] >= 50].groupby("encounter_id").size(),
    "sev_n_ge75": sv[sv["severity_score"] >= 75].groupby("encounter_id").size(),
    "sev_sum": g.sum(),
    "sev_n_rows": g.size(),
})


def add_extra(df):
    out = df.merge(extra, left_on="encounter_id", right_index=True, how="left")
    for c in extra.columns:
        out[c] = out[c].fillna(0)
    return out


# --- Bandingkan: baseline vs baseline + fitur keparahan ---
results = []
for name, (tr_, te_) in {"baseline": (train, test), "baseline + severity": (add_extra(train), add_extra(test))}.items():
    X, Xt, cats = prep_matrix(tr_, te_)
    oof, iters = oof_lgb(X, y, cats, K, cfg=CFG_BASE, seed=42)
    m = metrics(oof, y, K, seed=42)
    results.append({"set": name, "n_fitur": X.shape[1], **m})
    print(f"{name:22s} logloss={m['logloss']:.5f} f1_argmax={m['f1_argmax']:.4f} f1_split={m['f1_split']:.4f}", flush=True)
    if name == "baseline + severity":
        X_best, cats_best = X, cats
pd.DataFrame(results).to_csv(os.path.join(HERE, "fitur_severity.csv"), index=False)
print(f"Fitur selesai ({time.time()-t0:.0f} detik)\n", flush=True)

# --- Pencarian hyperparameter (random search, repeated CV) ---
rng = np.random.default_rng(7)
SPACE = {"learning_rate": [0.01, 0.02, 0.03, 0.05], "num_leaves": [7, 15, 31, 63],
         "min_child_samples": [20, 40, 80], "feature_fraction": [0.2, 0.3, 0.5, 0.7],
         "bagging_fraction": [0.7, 0.9, 1.0], "lambda_l2": [1.0, 5.0, 10.0, 20.0],
         "extra_trees": [False, True]}
out_path = os.path.join(HERE, "hp_search.csv")
rows = []
for t in range(N_CONFIGS):
    cfg = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
    cfg["extra_trees"] = bool(cfg["extra_trees"])
    cfg["bagging_freq"] = 1
    cfg = {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}
    logl, f1a, f1s, its = [], [], [], []
    for r in range(REPEATS):
        oof, iters = oof_lgb(X_best, y, cats_best, K, cfg=cfg, seed=42 + r)
        m = metrics(oof, y, K, seed=42 + r)
        logl.append(m["logloss"]); f1a.append(m["f1_argmax"]); f1s.append(m["f1_split"]); its.append(np.mean(iters))
    rows.append({**cfg, "logloss": np.mean(logl), "f1_argmax": np.mean(f1a),
                 "f1_split": np.mean(f1s), "iterasi": int(np.mean(its))})
    pd.DataFrame(rows).sort_values("logloss").to_csv(out_path, index=False)   # simpan progres
    print(f"[{t+1:2d}/{N_CONFIGS}] logloss={np.mean(logl):.5f} f1_argmax={np.mean(f1a):.4f} "
          f"f1_split={np.mean(f1s):.4f} iter={int(np.mean(its))} | {cfg}", flush=True)

print(f"\nSelesai dalam {time.time()-t0:.0f} detik. Hasil: hp_search.csv (diurutkan dari log-loss terkecil)")
print("Konfigurasi terbaik menurut log-loss:")
print(pd.read_csv(out_path).head(3).to_string())

# Penjelasan Kode: `train_model_gdgoc.py`

Dokumen ini menjelaskan setiap fungsi di `train_model_gdgoc.py` statement demi statement. Alasan di balik setiap keputusan ada di `LOG_KEPUTUSAN_MODEL.md`.

---

## Gambaran alur

1. **`prepare`**: pisahkan fitur dan label, lalu samakan kolom kategori antara train dan test.
2. **`logistic_baseline`**: LR sebagai pembanding.
3. **`random_configs` + `cv_lgb`**: coba 12 konfigurasi LightGBM dengan 5-fold CV, pilih yang log-loss-nya terkecil.
4. **`tune_weights` + `split_half_check`**: tuning bobot probabilitas per kelas untuk Macro F1, lalu uji jujur dengan split-half.
5. **Latih final**: di seluruh train, 3 seed, prediksi test, lalu tulis submission.

---

## 1. `prepare(train, test)`

1. `X = train.drop(columns=ID_COLS + [TARGET])`: buang kolom identitas dan label, sisanya fitur.
2. `Xt = test.drop(columns=ID_COLS)`: fitur test. Test tidak punya label, jadi `target_class` memang tidak ada.
3. `classes = sorted(train[TARGET].unique())`: urutan kelas dibuat alfabetis dan tetap. Urutan ini dipakai untuk submission.
4. `y = train[TARGET].map({c: i ...})`: label diubah menjadi angka 0 sampai 3 sesuai urutan `classes`.
5. `cats = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("str")]`: cari kolom teks. Pandas versi baru memakai tipe `str`, jadi cek `object` saja tidak cukup.
6. Loop `for c in cats:`:
   - `levels = sorted(set(X[c].dropna()) | set(Xt[c].dropna()))`: gabungkan level dari train dan test. Ini tidak melihat label, jadi aman.
   - `cat_type = pd.CategoricalDtype(levels)`: tipe kategori dengan daftar level yang sama.
   - `X[c] = ...` dan `Xt[c] = ...`: terapkan tipe yang sama ke train dan test. Dengan begitu kode kategori identik di keduanya.
7. `assert set(X.columns) == set(Xt.columns)`: pengaman. Kolom harus sama.
8. `Xt = Xt[X.columns]`: urutkan kolom test mengikuti train. LightGBM mencocokkan kolom berdasarkan posisi, jadi urutan ini penting.
9. `return X, Xt, y.values, classes, cats`.

## 2. `class_weights(y, K)`

1. `counts = np.bincount(y, minlength=K)`: jumlah baris tiap kelas. `minlength=K` memastikan semua kelas ada, walau ada kelas yang kosong di fold tertentu.
2. `per_class = len(y) / (K * counts)`: rumus bobot seimbang. Kelas yang lebih jarang mendapat bobot lebih besar.
3. `return per_class[y]`: ubah menjadi bobot per baris dengan indexing.

## 3. `logistic_baseline(X, y, cats, K)`

1. `Xd = pd.get_dummies(X, columns=cats, dummy_na=True, dtype=float)`: kategori menjadi kolom biner. `dummy_na=True` membuat kolom terpisah untuk nilai kosong, sehingga informasi "tidak tercatat" tidak hilang.
2. `make_pipeline(...)`: rangkai tiga langkah. Pipeline penting karena imputasi dan standarisasi dihitung ulang di setiap fold, sehingga tidak ada kebocoran statistik dari fold validasi.
   - `SimpleImputer(strategy="median")`: isi NaN dengan median kolom.
   - `StandardScaler()`: skala setiap kolom ke rata-rata 0 dan simpangan 1. LR sensitif terhadap skala.
   - `LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)`: `C` adalah kekuatan regularisasi, `class_weight` menyeimbangkan kelas, dan `max_iter` memberi cukup iterasi untuk konvergen.
3. `skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)`: fold yang sama dengan LightGBM, supaya perbandingannya adil.
4. `for tr, va in skf.split(Xd, y):`: latih di `tr`, prediksi di `va`.
5. `oof[va] = model.predict_proba(Xd.iloc[va])`: simpan probabilitas out-of-fold.

## 4. `cv_lgb(X, y, cfg, K, cats)`

1. `params = dict(objective="multiclass", num_class=K, metric="multi_logloss", verbosity=-1, seed=SEED, **cfg)`: parameter LightGBM. `**cfg` menyisipkan hyperparameter konfigurasi.
2. `dtr = lgb.Dataset(X.iloc[tr], y[tr], weight=class_weights(y[tr], K), categorical_feature=cats, free_raw_data=False)`: data latih dengan bobot kelas. `categorical_feature` memberi tahu LightGBM kolom mana yang kategori. `free_raw_data=False` menyimpan data asli supaya bisa dipakai lagi.
3. `dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr, ...)`: data validasi **tanpa bobot**. Ini disengaja: log-loss validasi harus mencerminkan distribusi kelas asli, bukan distribusi yang sudah diubah. `reference=dtr` memastikan kode kategorinya sama.
4. `booster = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])`: latih maksimal 800 pohon. Jika log-loss validasi tidak membaik selama 50 putaran, berhenti dan kembalikan model pada titik terbaik.
5. `oof[va] = booster.predict(X.iloc[va], num_iteration=booster.best_iteration)`: prediksi memakai iterasi terbaik, bukan iterasi terakhir.
6. `best_iters.append(booster.best_iteration)`: simpan iterasi terbaik fold ini. Angka ini dipakai untuk jumlah iterasi final.
7. `return oof, best_iters`.

## 5. `random_configs(n, rng)`

1. `space = {...}`: daftar nilai yang boleh dipilih untuk setiap hyperparameter. Rentangnya dipilih supaya masuk akal untuk data kecil (3.600 baris).
2. `configs = [dict(...)]`: konfigurasi pertama adalah default yang wajar. Ini selalu dicoba sebagai titik pembanding.
3. Loop `for _ in range(n - 1):`: sisanya acak.
   - `c = {k: v[rng.integers(len(v))] for k, v in space.items()}`: pilih satu nilai per parameter secara acak.
   - `c["bagging_freq"] = 1`: aktifkan bagging. Tanpa ini, `bagging_fraction` tidak berpengaruh.
   - `{k: (v.item() if hasattr(v, "item") else v) ...}`: ubah tipe numpy menjadi tipe Python biasa. Beberapa versi LightGBM menolak tipe numpy di parameter.
4. `return configs`.

## 6. `macro_f1(P, y, w)`

`f1_score(y, (P * w).argmax(1), average="macro")`. Probabilitas dikali bobot per kelas, lalu kelas dengan nilai tertinggi dipilih. `average="macro"` memberi bobot sama pada setiap kelas.

## 7. `tune_weights(P, y, K, passes)`

Ini coordinate ascent: mengubah satu bobot pada satu waktu.

1. `w = np.ones(K)`: mulai dari bobot seragam, yaitu tanpa penyesuaian.
2. `best = macro_f1(P, y, w)`: skor awal.
3. `for _ in range(passes):`: ulangi beberapa putaran, karena bobot satu kelas memengaruhi optimum kelas lain.
4. `for k in range(K):` dan `for g in GRID:`: coba setiap kandidat untuk setiap kelas.
5. `w2 = w.copy(); w2[k] = g`: buat salinan dengan satu bobot diubah. Salinan dibuat supaya `w` asli tidak rusak selama pencarian.
6. `if s > best + 1e-9: best, w = s, w2`: terima perubahan hanya jika skor naik. `1e-9` mencegah penerimaan karena selisih numerik yang tidak berarti.
7. `return w, best`.

## 8. `split_half_check(P, y, K, seed)`

Ini uji untuk melihat apakah tuning bobot benar-benar membantu, atau hanya menghafal OOF.

1. `idx = rng.permutation(len(y))`: acak urutan baris.
2. `halves = [idx[:half], idx[half:]]`: bagi menjadi dua separuh.
3. `for a, b in [(0, 1), (1, 0)]:`: dua arah. Bobot dituning di separuh `a`, lalu diuji di separuh `b`.
4. `w, _ = tune_weights(P[halves[a]], y[halves[a]], K)`: tuning hanya dengan data separuh `a`.
5. `base = macro_f1(P[halves[b]], y[halves[b]], np.ones(K))`: skor tanpa tuning di separuh `b`.
6. `tuned = macro_f1(P[halves[b]], y[halves[b]], w)`: skor dengan bobot dari separuh `a`, diuji di separuh `b`. Separuh `b` tidak pernah dipakai untuk menentukan bobot ini.
7. `results.append(...)`: catat hasilnya.

## 9. `main(feat_dir, data_dir, out_dir)`

**Persiapan:**
1. `train = pd.read_csv(...)` dan `test = pd.read_csv(...)`: baca fitur dari tahap sebelumnya.
2. `X, Xt, y, classes, cats = prepare(train, test)`: siapkan data.
3. `K = len(classes)`: jumlah kelas, yaitu 4.

**Baseline:**
4. `oof_lr = logistic_baseline(X, y, cats, K)`: prediksi LR.
5. `log["baseline_lr"] = {...}`: simpan Macro F1 dan log-loss LR.

**Pencarian LightGBM:**
6. `for i, cfg in enumerate(random_configs(N_TRIALS, rng)):`: coba setiap konfigurasi.
7. `oof, iters = cv_lgb(X, y, cfg, K, cats)`: CV untuk konfigurasi ini.
8. `ll = log_loss(y, oof)`: log-loss OOF sebagai pemilih utama (M1).
9. `f1 = f1_score(...)`: Macro F1 argmax, hanya dicatat untuk referensi.
10. `if best is None or ll < best["ll"]: best = {...}`: simpan konfigurasi dengan log-loss terkecil.
11. `pd.DataFrame(rows).sort_values("oof_logloss").to_csv(...)`: simpan tabel semua trial, diurutkan dari yang terbaik.

**Bobot dan validasi:**
12. `P = best["oof"]`: probabilitas OOF konfigurasi terbaik.
13. `w, tuned_f1 = tune_weights(P, y, K)`: tuning bobot di seluruh OOF.
14. `log["oof_macro_f1_tuned_in_sample"] = ...`: skor ini optimistik karena dituning dan diukur pada data yang sama. Dicatat hanya sebagai referensi.
15. `log["split_half_check"] = split_half_check(P, y, K)`: perkiraan jujur.

**Latih final:**
16. `n_rounds = int(np.mean(best["iters"]) * 1.1)`: iterasi final dari rata-rata fold dinaikkan 10% (M7).
17. `Pt = np.zeros((len(Xt), K))` dan `imp = np.zeros(X.shape[1])`: wadah untuk rata-rata prediksi dan kepentingan fitur dari 3 seed.
18. `params = dict(objective="multiclass", num_class=K, verbosity=-1, **best["cfg"])`: parameter model final.
19. `for seed in [1, 2, 3]:`: latih tiga model dengan seed berbeda.
20. `d = lgb.Dataset(X, y, weight=class_weights(y, K), ...)`: seluruh train, dengan bobot kelas.
21. `m = lgb.train({**params, "seed": seed}, d, num_boost_round=n_rounds)`: latih tanpa early stopping, karena tidak ada validasi. Jumlah iterasi sudah ditentukan dari CV.
22. `Pt += m.predict(Xt) / 3`: rata-rata probabilitas dari tiga seed.
23. `imp += m.feature_importance("gain") / 3`: rata-rata kepentingan fitur, dipakai untuk analisis.

**Output:**
24. Simpan `test_probabilities.csv`: probabilitas per kelas untuk setiap sampel test. Berguna untuk analisis atau ensemble nanti.
25. `pred = np.array(classes)[(Pt * w).argmax(1)]`: prediksi akhir. Probabilitas dikali bobot, lalu argmax.
26. Blok `if data_dir:`: urutkan submission mengikuti `sample_submission.csv`. `validate="one_to_one"` memastikan tidak ada duplikasi. `assert ... notna()` memastikan setiap ID punya prediksi.
27. `sub.to_csv(...)`: file submission.
28. `log["test_pred_distribution"]`: distribusi prediksi test. Dibandingkan dengan distribusi train untuk mendeteksi keanehan.
29. `feature_importance.csv`: kepentingan fitur, diurutkan dari yang tertinggi.
30. `oof_classification_report`: precision, recall, dan F1 per kelas pada OOF, dengan bobot.
31. `oof_probabilities.csv`: probabilitas OOF per sampel train, untuk analisis kesalahan.
32. `json.dump(log, ...)`: simpan semua hasil.

## 10. Blok `if __name__ == "__main__":`

`argparse` membaca `--feat`, `--data`, dan `--out`. Argumen `--data` opsional. Tanpa `--data`, urutan submission tidak diubah, tetapi ini tidak disarankan.

---

## Cara membaca `model/`

| File | Isi |
|---|---|
| `submission.csv` | File untuk diunggah ke Kaggle. Format `sample_id,target_class`, 1.400 baris, urutan sama dengan contoh. |
| `model_log.json` | Semua angka dan keputusan: baseline, parameter terbaik, bobot, split-half, distribusi prediksi. |
| `cv_search_results.csv` | Hasil 12 trial, diurutkan dari log-loss terkecil. |
| `feature_importance.csv` | Kepentingan fitur (gain), diurutkan. Berguna untuk laporan. |
| `oof_probabilities.csv` | Probabilitas out-of-fold per sampel train. Berguna untuk mencari kesalahan. |
| `test_probabilities.csv` | Probabilitas test per kelas. |

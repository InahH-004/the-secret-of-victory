# Log Keputusan: Pemodelan GDGoC UGM Academic Hacker 2026

Lanjutan dari `LOG_KEPUTUSAN.md` (yang membahas fitur). Dokumen ini membahas keputusan di tahap model. Setiap keputusan punya alternatif, alasan, dan bukti dari hasil run `train_model_gdgoc.py`.

---

## M1. Hyperparameter dipilih berdasarkan log-loss OOF, bukan Macro F1 argmax

**Keputusan.** Dari 12 konfigurasi, dipilih yang log-loss OOF-nya terkecil (trial 10).

**Alternatif.** Memilih berdasarkan Macro F1 argmax (trial 0).

**Alasan.** Macro F1 argmax dari 3.600 sampel bersifat berisik. Satu prediksi yang berpindah kelas bisa menggeser skor cukup banyak. Log-loss mengukur kualitas probabilitas di seluruh baris, jadi lebih stabil untuk memilih model. Aturan ini ditetapkan sebelum melihat perbandingan F1 konfigurasi.

**Bukti.** Trial 0 punya F1 argmax tertinggi (0,6803) tetapi log-loss lebih buruk (0,5143). Trial 10 punya F1 argmax 0,6665 dan log-loss 0,5084. Setelah bobot kelas dituning (M4), kedua konfigurasi berada di kisaran yang sama (0,6953 vs 0,6962). Namun peningkatan split-half trial 10 lebih besar (+0,016 vs +0,007). Jadi keputusan awal tetap didukung.

**Risiko.** Selisih antar konfigurasi kecil, sekitar 0,01. Untuk Macro F1 pada test, perbedaan ini bisa tidak terlihat.

**Kode.** `main()` pada blok pencarian, dan `cv_lgb()`.

---

## M2. Early stopping memakai fold validasi, sehingga OOF sedikit optimistik

**Keputusan.** Setiap fold dihentikan dini berdasarkan log-loss pada fold validasi itu sendiri.

**Alternatif.** Menyisihkan validation set terpisah yang tidak pernah dipakai untuk early stopping.

**Alasan.** Dengan 3.600 sampel, memisahkan validasi lagi akan membuat data latih terlalu kecil. Dampak optimisme ini dikurangi dengan dua cara: jumlah iterasi final ditentukan dari rata-rata fold (M7), dan skor yang dilaporkan adalah perkiraan kasar, bukan angka pasti.

**Risiko.** Skor OOF sedikit lebih tinggi daripada skor test yang sebenarnya.

**Kode.** `cv_lgb()`, `callbacks=[lgb.early_stopping(50, ...)]`.

---

## M3. Dua mekanisme untuk kelas minoritas: class weight saat latih dan bobot probabilitas pasca-latih

**Keputusan.** Saat melatih, setiap baris diberi bobot seimbang `n / (K * n_kelas)`. Setelah itu, probabilitas dikali bobot per kelas (M4) sebelum argmax.

**Alternatif.** Hanya salah satu dari keduanya.

**Alasan.**
- Bobot saat latih membantu model belajar pola kelas minoritas, terutama `strep_pharyngitis` yang hanya 5,4%.
- Bobot pasca-latih menyesuaikan titik keputusan langsung untuk Macro F1, yang tidak sama dengan log-loss yang dioptimalkan model.

**Bukti.** Baseline tanpa tuning bobot memberi F1 0,6665 pada OOF. Setelah tuning, perkiraan naik sekitar 0,016 (lihat M5).

**Risiko.** Dua mekanisme ini saling menumpuk. Bobot pasca-latih yang besar bisa membuat kelas minoritas terlalu sering diprediksi. Hasil test menunjukkan `strep_pharyngitis` diprediksi 10,1%, padahal proporsi di train 5,4%. Ini konsisten dengan tujuan Macro F1, tetapi perlu dijelaskan di write-up.

**Kode.** `class_weights()`, `tune_weights()`.

---

## M4. Grid bobot dibatasi 0,5 sampai 2,0, dan bobot `strep_pharyngitis` mentok di batas atas

**Keputusan.** Pengali bobot per kelas hanya boleh di antara 0,5 dan 2,0 (21 nilai dalam skala log).

**Alternatif.** Grid yang lebih lebar.

**Alasan.** Grid yang lebih lebar memberi lebih banyak ruang untuk overfit pada OOF.

**Bukti.** Bobot akhir: `acute_bronchitis` 0,574, `strep_pharyngitis` 2,000 (batas atas), `viral_pharyngitis` 0,871, `viral_sinusitis` 1,149.

**Risiko.** Karena `strep_pharyngitis` mentok di batas, nilai optimum mungkin lebih tinggi. Ini belum diuji. Memperluas grid perlu validasi tambahan agar tidak overfit. Ditandai sebagai kandidat tindak lanjut.

**Kode.** `GRID` di `train_model_gdgoc.py`.

---

## M5. Bobot diuji dengan split-half, bukan hanya dilaporkan dari in-sample

**Keputusan.** Bobot dituning pada satu separuh OOF lalu diuji pada separuh lain, dan sebaliknya.

**Alternatif.** Melaporkan Macro F1 setelah tuning langsung pada OOF yang sama.

**Alasan.** Tuning bobot pada data yang sama dengan evaluasinya akan menghasilkan skor yang terlalu bagus. Split-half memberi perkiraan yang lebih jujur.

**Bukti.**

| Arah | F1 tanpa tuning | F1 dengan tuning | Peningkatan |
|---|---|---|---|
| Tune di separuh A, uji di B | 0,6613 | 0,6697 | +0,0084 |
| Tune di separuh B, uji di A | 0,6714 | 0,6948 | +0,0234 |
| Rata-rata | | | +0,0159 |

**Catatan.** Skor in-sample setelah tuning adalah 0,6962. Perkiraan jujur lebih rendah, yaitu sekitar 0,6665 + 0,016 ≈ 0,68. Ini perkiraan kasar, dengan ketidakpastian sekitar ±0,02.

**Kode.** `split_half_check()`.

---

## M6. Logistic Regression dipakai sebagai baseline

**Keputusan.** LR dengan one-hot untuk kategori, imputasi median, dan standarisasi, dilatih dengan fold yang sama.

**Alasan.** Tanpa baseline, kita tidak tahu apakah model non-linear memberi manfaat. Ini juga memberi argumen untuk laporan.

**Bukti.**

| Model | Macro F1 OOF (argmax) | Log-loss |
|---|---|---|
| Logistic Regression | 0,6285 | 1,3096 |
| LightGBM (trial 10) | 0,6665 | 0,5084 |

LightGBM unggul sekitar 0,04 pada Macro F1. Log-loss LR sangat tinggi karena `class_weight="balanced"` membuat probabilitasnya tidak terkalibrasi, sehingga perbandingan log-loss ini tidak adil. Perbandingan F1 lebih layak.

**Kode.** `logistic_baseline()`.

---

## M7. Model final dilatih ulang di seluruh train dengan 327 iterasi, rata-rata 3 seed

**Keputusan.** Jumlah iterasi final = rata-rata iterasi terbaik fold × 1,1 = 327. Dilatih dengan 3 seed, probabilitas test dirata-ratakan.

**Alasan.**
- Data latih final (3.600) lebih banyak daripada data per fold (2.880), sehingga sedikit lebih banyak iterasi wajar. Faktor 1,1 adalah heuristik sederhana.
- Tiga seed mengurangi variasi acak. Konfigurasi terpilih memakai `feature_fraction = 0,5`, jadi setiap pohon hanya melihat setengah fitur dan hasilnya bisa sedikit berbeda antar seed.

**Risiko.** Faktor 1,1 tidak diuji secara terpisah. Ini bisa diperiksa dengan membandingkan beberapa nilai pengali.

**Kode.** `main()`, bagian latih final.

---

## M8. Kategori memakai level gabungan train dan test

**Keputusan.** Kolom teks (gender, race, dll.) dikonversi ke kategori dengan daftar level gabungan dari train dan test.

**Alasan.** Jika level train dan test tidak sama, LightGBM akan memberi kode berbeda untuk kategori yang sama. Daftar level gabungan tidak melihat label, jadi aman.

**Kode.** `prepare()`.

---

## M9. Submission mengikuti urutan `sample_submission.csv`

**Keputusan.** Baris submission disusun ulang sesuai urutan contoh dari Kaggle, dengan `validate="one_to_one"`.

**Alasan.** Meskipun Kaggle biasanya mencocokkan berdasarkan `sample_id`, urutan yang sama persis menghindari risiko salah format.

**Bukti.** Urutan `sample_id` sama persis dengan `sample_submission.csv`. Tidak ada NaN dan tidak ada duplikat.

**Kode.** `main()`, bagian menulis submission.

---

## M10. Tidak ada seleksi fitur

**Keputusan.** Semua 515 fitur dipakai.

**Alasan.** LightGBM sudah menangani fitur tidak relevan dengan baik, dan seleksi fitur berdasarkan CV bisa menambah optimisme. Seleksi fitur adalah kandidat tindak lanjut.

**Catatan.** Dari gain importance, 10 fitur teratas menyumbang sekitar 47% dari total gain. Ini menunjukkan sinyal terkonsentrasi, tetapi belum diuji apakah membuang fitur lain membantu.

---

## M11. Pemeriksaan pergeseran distribusi train vs test

**Keputusan.** Tidak ada koreksi, hanya pemeriksaan.

**Bukti.** Rentang waktu prediksi train dan test hampir sama. Standardized mean difference untuk fitur utama semuanya di bawah 0,07 (usia -0,024, gejala +0,028, batuk +0,068). Split tampaknya acak terhadap waktu dan karakteristik pasien, sehingga skor CV seharusnya mewakili test dengan baik.

---

## Hal yang belum diverifikasi

1. Batas bobot kelas `strep_pharyngitis` (M4). Perlu diuji dengan grid yang lebih luas dengan validasi tambahan.
2. Faktor 1,1 untuk jumlah iterasi (M7).
3. Seleksi fitur (M10).
4. Pengecekan apakah skor leaderboard sesuai dengan perkiraan OOF.

---

## Ringkasan angka

| Metrik | Nilai |
|---|---|
| Baseline LR, Macro F1 OOF | 0,6285 |
| LightGBM, Macro F1 OOF (argmax) | 0,6665 |
| LightGBM, Macro F1 OOF setelah tuning bobot (in-sample) | 0,6962 |
| LightGBM, peningkatan split-half rata-rata | +0,0159 |
| Perkiraan kasar skor test | sekitar 0,68 (±0,02) |
| Log-loss OOF terbaik | 0,5084 |

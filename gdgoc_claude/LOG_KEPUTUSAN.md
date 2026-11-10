# Log Keputusan: Pipeline Fitur GDGoC UGM Academic Hacker 2026

Setiap keputusan di bawah ini punya empat hal: apa yang diputuskan, alternatifnya, alasannya, dan bukti dari data. Angka-angka diambil dari hasil menjalankan `run_gdgoc.py`, bukan dari perkiraan. Kalau ada keputusan yang perlu diubah, cari kodenya lewat kolom "Kode" lalu lihat dampaknya di sanity check.

---

## D1. Cutoff memakai resolusi detik: event di detik yang sama dengan `prediction_time` dianggap tersedia

**Keputusan.** Sebuah event dianggap tersedia jika `waktu_event < cutoff + 1 detik`. Dengan begitu data triase yang dicatat di detik yang sama dengan prediksi tetap dipakai.

**Alternatif.** Aturan strict `waktu_event < cutoff`.

**Alasan.** `prediction_time` ternyata adalah timestamp triase yang dipotong ke detik penuh. Pengecekan `floor(timestamp_triase, detik) == prediction_time` berlaku untuk 100% baris di `triage_vitals`, `triage_symptoms`, dan `triage_symptom_severity`. Jadi triase terjadi pada saat prediksi, bukan sesudahnya. Aturan strict akan menganggap data itu "masa depan" hanya karena ada selisih sub-detik.

**Bukti.**
- Mode strict membuang seluruh data triase: vital 30.000 dari 30.000 baris, gejala 19.065 dari 19.065 baris, dan severity 16.594 dari 16.594 baris.
- Sanity check Macro F1 (5-fold CV pada train):
  - Dengan triase, mode same-second: **0,676**
  - Tanpa triase: **0,291**
  - Selisih ini menunjukkan triase adalah sumber sinyal utama. Membuangnya karena strict akan menurunkan skor secara drastis.

**Risiko.** Jika juri menganggap triase terjadi setelah prediksi, keputusan ini bisa dipertanyakan. Mitigasi: jelaskan di write-up bahwa `prediction_time` adalah timestamp triase yang dipotong detik, dan tunjukkan pengecekan `floor` di atas.

**Kode.** `relational_features.py` → `ONE_SECOND`, `apply_cutoff()`. Pemanggilan dengan `strict=True` hanya untuk diagnostik di `main()`.

---

## D2. Riwayat medis memakai aturan yang sama dan semua baris lolos

**Keputusan.** Tabel riwayat (`conditions`, `medications`, `procedures`, `immunizations`, `allergies`, `observations`, `previous_encounter`) memakai aturan D1.

**Alasan.** Pengecekan awal menunjukkan 0 baris dengan waktu di atas `prediction_time`. Jadi aturan manapun akan memberi hasil yang sama untuk tabel ini. Memakai satu aturan saja membuat pipeline lebih konsisten.

**Bukti.** Di log, semua tabel riwayat memiliki `kept == rows_in` dan `dropped = 0`. Dari 3.583 baris `conditions` yang terhubung ke pasien dalam indeks, semuanya berada sebelum cutoff.

**Kode.** `history()`, `previous_encounters()`, `observations()`.

---

## D3. Keluhan utama (`presenting_complaints`) diasumsikan tersedia saat triase

**Keputusan.** Tabel ini tidak punya kolom waktu, sehingga dipakai langsung dan diasumsikan dicatat saat triase.

**Alternatif.** Membuang tabel ini, atau menganggapnya tidak tersedia.

**Alasan.** Tabel ini memiliki satu baris per encounter (5.000 baris). Setiap teksnya berawalan `Reports:` dan berisi keluhan yang biasanya dicatat saat triase. Jadi kemungkinan besar tercatat di saat yang sama dengan triase.

**Bukti.** Semua 5.000 baris punya awalan `Reports:`, dan encounter-nya cocok 100% dengan tabel indeks.

**Risiko.** Ini asumsi, bukan hasil verifikasi. Tingkat keyakinan sedang. Cara memverifikasinya kalau ada data lain: periksa apakah teks keluhan sudah ada di tabel lain yang bertimestamp.

**Kode.** `triage()`, bagian keluhan utama.

---

## D4. Parsing pendapatan (`income_at_visit`): koma sebagai pemisah ribuan, `k` sebagai ribuan, `$` dan `USD` dibuang

**Keputusan.** Fungsi `parse_amount()` menghapus `USD` dan `$`, menghapus koma, dan mengalikan akhiran `k` dengan 1.000.

**Alternatif.** Koma sebagai desimal (`20,982` menjadi `20.982`). Itu tidak masuk akal untuk pendapatan.

**Alasan.** Format yang ditemukan: `13.1k`, `20,982`, `USD 122,462`, `$51,777`, dan angka biasa.

**Bukti.**
- Ada 455 nilai kosong dan 232 nilai bukan kosong yang gagal diparse. Semua 232 berformat `$51,777` dan sejenisnya.
- Setelah `$` ditambahkan ke aturan, tersisa 0 yang gagal.

**Risiko.** `13.1k` diasumsikan 13.100. Ini wajar, tetapi perlu disebutkan. Nilai yang berbentuk `20,982` diasumsikan 20.982 (dengan koma ribuan), bukan 20,982 desimal. Pola ini konsisten dengan pendapatan.

**Kode.** `rf.parse_amount()`, `demographics()`.

---

## D5. Koma desimal pada vital dan observasi diubah menjadi titik

**Keputusan.** `to_float_decimal_comma()` mengganti `,` dengan `.` sebelum diubah menjadi angka.

**Alternatif.** Membuang nilai yang bukan angka murni.

**Alasan.** Ditemukan nilai seperti `18,33` (BMI), `0,9` (SpO2 pecahan), dan `300,5` (berat badan). Di kolom ini, koma hanya berfungsi sebagai desimal. Ada 112 nilai vital dan 165 nilai observasi numerik yang berkoma. Kalau dibuang, data itu hilang begitu saja.

**Kode.** `rf.to_float_decimal_comma()`, `triage()`, `observations()`.

---

## D6. Konversi satuan vital

**Keputusan.**
- SpO2 dengan satuan `fraction` dikali 100 (0,96 menjadi 96). Ada 431 baris.
- Tekanan darah dengan satuan `kPa` dikali 7,50062 untuk menjadi mmHg. Ada 618 baris.

**Alasan.** Label satuan jelas dan nilainya sesuai dengan rentang masing-masing satuan. Misalnya SpO2 0,62 sampai 0,99 jelas merupakan pecahan.

**Risiko.** Kalau label satuannya salah, konversi ini justru memperburuk. Mitigasi: setelah konversi, rentang fisiologis (D7) akan menangkap nilai yang tidak masuk akal.

**Kode.** `convert_vital_units()`.

---

## D7. Nilai di luar rentang fisiologis dijadikan NaN, tidak diperbaiki

**Keputusan.** Nilai di luar batas klinis diganti NaN. Batas ada di `VITAL_RANGES` dan `OBS_RANGES` di `run_gdgoc.py`.

**Alternatif.**
- Menebak perbaikan, misalnya membagi 10 untuk nilai seperti BMI 304 atau sistolik 1390.
- Melakukan winsorize atau clipping ke batas.

**Alasan.** Nilai seperti BMI 304 atau detak jantung 990 kemungkinan besar salah ketik, tetapi kita tidak tahu pasti nilai yang benar. Menebak akan memasukkan informasi palsu. NaN lebih jujur, dan LightGBM bisa menangani NaN.

**Bukti.**
- Vital: 127 nilai dijadikan NaN (Heart rate 9, Respiratory rate 16, Systolic BP 23, Diastolic BP 24, SpO2 42, BMI 13).
- Observasi: 60 nilai dijadikan NaN.

**Risiko.** Batas dipilih berdasarkan pengetahuan klinis umum, bukan dari dokumen panitia. Perlu diverifikasi jika ada referensi resmi.

**Kode.** `apply_valid_range()`, `VITAL_RANGES`, `OBS_RANGES`.

---

## D8. `Body Height` tidak diberi rentang

**Keputusan.** Tidak ada batas untuk tinggi badan.

**Alasan.** Nilainya campur: sebagian kira-kira 0,16 dan sebagian 1.860. Satuannya kemungkinan meter dan sentimeter, tetapi belum diverifikasi. Membuat batas sekarang bisa salah.

**Tindak lanjut.** Cek kolom `unit` untuk `Body Height`, lalu tambahkan batas. Ini belum dikerjakan.

**Kode.** `OBS_RANGES` (sengaja tidak ada).

---

## D9. Kode yang jarang (muncul kurang dari 20 kali) dibuang dari fitur

**Keputusan.** Hanya kode dengan frekuensi minimal `MIN_COUNT = 20` yang menjadi kolom.

**Alternatif.** Semua kode dipakai, atau batas yang berbeda.

**Alasan.** Kode yang sangat jarang membuat kolom yang hampir selalu 0, sehingga mudah overfit dan menambah noise.

**Vocab dibangun dari siapa?** Dari seluruh baris fitur, termasuk train dan test. Ini tidak memakai label, jadi tidak ada kebocoran label. Alternatif yang lebih ketat adalah membangun vocab dari train saja, tetapi belum dipakai.

**Kode.** `rf.vocab_from()`, konstanta `MIN_COUNT`.

---

## D10. `city` dan `county` dibuang

**Keputusan.** Kedua kolom geografi ini tidak dipakai.

**Alasan.** Kardinalitasnya tinggi dan sebagian banyak yang kosong. Kolom seperti ini mudah membuat model menghafal lokasi, bukan pola klinis.

**Alternatif.** Target encoding dengan CV. Lebih kompleks dan belum dicoba.

**Kode.** `demographics()`.

---

## D11. Fitur recency dan count memakai `_days_before` dan `min`

**Keputusan.** Untuk tiap tabel dibuat `_n` (jumlah event) dan `_days_since_last` (jarak hari ke event terbaru).

**Alasan.** Jumlah dan keterbaruan adalah dua sinyal dasar untuk riwayat medis. Kapan sebuah kondisi terakhir tercatat sering lebih penting daripada hanya ada atau tidak.

**Catatan.** Untuk triase, `_days_since_last` hampir selalu sekitar 0 karena semuanya terjadi di waktu yang sama. Fitur ini tidak merusak model, tetapi juga tidak berguna.

**Kode.** `rf.agg_recency()`.

---

## D12. Kosong diisi 0 untuk bag dan count, NaN tetap NaN untuk nilai numerik

**Keputusan.** Kolom biner (bag) dan jumlah (count) yang kosong diisi 0. Nilai numerik seperti tekanan darah dibiarkan NaN.

**Alasan.** Bag dan count kosong berarti "tidak tercatat", dan itu wajar disimpulkan sebagai 0. Nilai numerik yang kosong berarti "tidak diketahui", dan 0 akan menyesatkan karena 0 adalah nilai nyata. LightGBM bisa menangani NaN.

**Kode.** `assemble()`.

---

## D13. Statistik per kode: `last` untuk vital dan observasi, plus `mean` untuk observasi

**Keputusan.**
- Vital memakai `last`, karena hanya ada satu pengukuran per encounter.
- Observasi memakai `last` dan `mean`. `last` adalah nilai terakhir sebelum cutoff. `mean` merangkum riwayat.

**Alasan.** `last` paling relevan untuk kondisi saat prediksi. `mean` menangkap kondisi kronis.

**Kode.** `rf.agg_numeric_by_code()`, `triage()`, `observations()`.

---

## D14. Nama kolom dibersihkan, tabrakan diberi akhiran

**Keputusan.** Karakter selain huruf, angka, dan `_` diganti `_`. Jika dua nama bertabrakan, yang kedua mendapat `_2`.

**Alasan.** LightGBM menolak karakter seperti spasi, `[`, `:`, dan `,` pada nama kolom. Tabrakan terjadi pada satu kasus: `urine protein test = +` dan `= ++` sama-sama menjadi `..._finding` setelah simbol `+` dibuang.

**Kode.** `rf.clean_col_names()`.

---

## D15. Sanity check memakai 5-fold CV, class_weight balanced, dan tanpa tuning

**Keputusan.** Model `LGBMClassifier` 300 pohon, learning rate 0,05, `class_weight="balanced"`, dan `StratifiedKFold` 5 fold. Tidak ada tuning hyperparameter.

**Alasan.** Tujuannya memeriksa apakah fitur bermakna, bukan mencari model terbaik. Tuning dilakukan setelah fitur stabil. Kelas `strep_pharyngitis` hanya 5,4%, sehingga `class_weight` dipakai.

**Catatan.** Skor 0,676 adalah out-of-fold pada train. Ini bukan skor leaderboard. Jangan mengoptimalkan ke public leaderboard karena hanya 30% data uji.

**Kode.** `sanity_cv()`.

---

## D16. Pengecekan `patient_id` saat menggabungkan train dan test

**Keputusan.** Sebelum digabung, `patient_id` dari train dan dari tabel fitur harus cocok. Jika tidak, pipeline berhenti.

**Alasan.** Ini mencegah kesalahan join yang diam-diam menggeser fitur ke pasien yang salah.

**Kode.** `main()`, bagian train dan test.

---

## Bukti verifikasi independen

Untuk memastikan kode benar, aku menghitung ulang beberapa fitur tanpa memakai template:

| Pemeriksaan | Hasil |
|---|---|
| `cond_n` untuk 3 pasien dengan kondisi terbanyak, dibandingkan hitung ulang langsung dari `conditions.csv` | 3 dari 3 cocok (7, 6, 6) |
| `vital__Heart_rate` untuk 5 encounter acak, dibandingkan dengan nilai mentah | 5 dari 5 cocok |
| Baris train dan test yang tidak punya satu pun fitur | 0 |
| Pengecekan baris yang melewati cutoff (`check_no_future`) | 0 |

---

## Hal yang belum dikerjakan (kandidat perbaikan)

1. Verifikasi satuan `Body Height` lalu tambahkan batasnya (D8).
2. Normalisasi sinonim gejala. Contohnya `throat pain`, `painful swallowing`, dan `throat discomfort` mungkin punya makna sama, tapi sekarang jadi kolom terpisah.
3. Tuning hyperparameter dan ensemble, setelah fitur stabil.
4. Target encoding untuk `city` dan `county` (D10).
5. Membangun vocab dari train saja, sebagai pilihan yang lebih ketat (D9).

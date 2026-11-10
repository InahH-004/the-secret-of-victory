# Panduan Template: Fitur dari Data Relasional (untuk GEMASTIK dan kasus lain)

Panduan ini menjelaskan cara memakai `relational_features.py` untuk dataset lain. Tujuannya satu: kamu tidak perlu mengulang semua keputusan dari nol, cukup mengikuti checklist dan mengubah bagian yang spesifik dataset.

---

## 1. Kapan template ini cocok

Template ini cocok jika datamu memenuhi tiga kondisi:

1. **Ada satu entitas yang diprediksi**, misalnya pasien, pengguna, atau transaksi, dan setiap entitas punya satu baris di tabel indeks dengan waktu prediksi.
2. **Ada banyak tabel relasional** yang terhubung ke entitas itu lewat kunci, dan sebagian punya timestamp.
3. **Ada risiko kebocoran waktu**, yaitu data dari masa depan bisa ikut terpakai.

Kalau tidak ada timestamp sama sekali, sebagian besar aturan cutoff tidak berlaku, tapi fungsi agregasi tetap bisa dipakai.

---

## 2. Yang disediakan template (dan tidak perlu diubah)

| Fungsi | Gunanya |
|---|---|
| `load_tables` | Membaca semua CSV secara rekursif |
| `parse_ts` | Parse timestamp campuran menjadi UTC |
| `to_float_decimal_comma` | Angka dengan koma desimal |
| `parse_amount` | Nilai uang dengan `$`, `USD`, `k`, dan pemisah ribuan |
| `clean_col_names` | Nama kolom aman untuk LightGBM, dengan penanganan tabrakan |
| `attach_cutoff` | Pasang cutoff ke tiap baris lewat kunci |
| `apply_cutoff` | Filter point-in-time, dengan log otomatis |
| `check_no_future` | Pengaman kebocoran |
| `apply_valid_range` | Nilai di luar rentang jadi NaN |
| `vocab_from` | Kode yang cukup sering |
| `agg_recency` | Jumlah dan jarak waktu ke event terbaru |
| `agg_code_bag` | Kolom biner per kode |
| `agg_numeric_by_code` | Statistik numerik per kode |
| `explode_tokens` | Teks bebas dipecah menjadi token |
| `to_entity_level` | Menyamakan level fitur ke level entitas |

---

## 3. Yang harus kamu tulis sendiri (per dataset)

Ada tiga bagian yang selalu berbeda antar dataset:

1. **Konfigurasi**: nama tabel, nama kolom kunci, rentang nilai, daftar tabel riwayat, dan ambang kemunculan.
2. **Fungsi per sumber data**: satu fungsi untuk tiap jenis tabel. Di GDGoC ada `triage()`, `history()`, `previous_encounters()`, `observations()`, dan `demographics()`.
3. **Fungsi rakit dan pemanggilan**: `main()` yang menyusun urutan dan memasangkan fitur ke train dan test.

Kalau kamu menemukan jenis tabel baru di GEMASTIK, cukup tulis satu fungsi baru dengan pola yang sama. Lihat contoh di bagian 6.

---

## 4. Checklist adaptasi (urutan kerja)

Ikuti urutan ini. Setiap langkah punya keputusan yang harus dicatat di log.

### Langkah 1. Inventaris data
- [ ] Daftar semua tabel, kolom kunci, dan kolom waktu.
- [ ] Baca data dictionary dan catat `model_role` tiap kolom.
- [ ] Hitung jumlah baris per tabel dan jumlah entitas unik. Ini menunjukkan grain tiap tabel.

### Langkah 2. Tentukan cutoff dan resolusinya
- [ ] Cari kolom waktu di tabel indeks (`prediction_time` di GDGoC).
- [ ] **Uji resolusi.** Cek apakah cutoff adalah timestamp event yang dipotong. Contoh uji:
  ```python
  ts = parse_ts(df_event[col])
  print((ts.dt.floor("s") == cutoff).mean())   # kalau mendekati 1.0, cutoff = event dipotong
  ```
- [ ] Putuskan aturan: strict (`<`) atau same-second (`< cutoff + resolusi`). Catat hasil uji ini.

### Langkah 3. Klasifikasi tiap tabel
Setiap tabel masuk salah satu kategori berikut. Kategorinya menentukan fungsi dan aturan yang dipakai.

| Kategori | Ciri | Contoh di GDGoC | Aturan |
|---|---|---|---|
| Statis | Tidak ada waktu | `patients` | Langsung dipakai |
| Riwayat | Waktu sebelum cutoff, terkait entitas | `conditions`, `observations` | `attach_cutoff` + `apply_cutoff` lewat `patient_id` |
| Momen prediksi | Waktu sama dengan cutoff, terkait encounter | `triage_vitals` | `attach_cutoff` lewat `encounter_id` + aturan same-second |
| Tanpa waktu, dianggap momen prediksi | Tidak ada kolom waktu | `presenting_complaints` | Langsung dipakai, asumsi dicatat |

### Langkah 4. Periksa format dan satuan
- [ ] Angka: cek apakah ada koma desimal, `$`, `k`, atau pemisah ribuan. Pakai `value.astype("string").str.contains(",")` untuk mengecek.
- [ ] Satuan: cek kombinasi nama dan satuan dengan `pd.crosstab(df.name, df.unit)`. Ini mengungkap satuan yang campur.
- [ ] Tipe: jangan percaya `dtype`. Pandas bisa menganggap kolom sebagai `string` padahal isinya angka.

### Langkah 5. Tetapkan rentang nilai
- [ ] Untuk setiap konsep numerik, tulis batas `(min, max)` berdasarkan domain.
- [ ] Catat batas dari mana. Di GDGoC batasnya dari pengetahuan klinis umum, bukan dokumen panitia.
- [ ] Nilai di luar batas jadi NaN, jangan ditebak. Catat jumlahnya.

### Langkah 6. Pilih jenis agregasi
| Jenis data | Agregasi |
|---|---|
| Kejadian (ada/tidak) | `agg_code_bag` |
| Jumlah dan keterbaruan | `agg_recency` |
| Nilai numerik per kode | `agg_numeric_by_code` dengan `last`, `mean`, `max`, atau `min` |
| Teks bebas | `explode_tokens` lalu `agg_code_bag` |
| Ringkasan per entitas | `groupby` langsung, seperti `dur_days` |

### Langkah 7. Atur ambang kemunculan
- [ ] Tentukan `MIN_COUNT`. Mulai dari 20, lalu lihat jumlah kolom yang dihasilkan.
- [ ] Vocab dibangun dari fitur (tanpa label), jadi boleh dari train+test.

### Langkah 8. Rakit dan isi kosong
- [ ] `bag` dan `count` kosong diisi 0, karena artinya "tidak tercatat".
- [ ] Numerik tetap NaN, karena artinya "tidak diketahui". LightGBM bisa menangani NaN.

### Langkah 9. Validasi (wajib)
- [ ] Pengecekan kebocoran: `check_no_future` harus mengembalikan 0.
- [ ] Hitung ulang 3 sampai 5 fitur dengan cara berbeda, tanpa template. Pilih pasien yang punya banyak data, jangan pasien kosong, karena pasien kosong selalu 0 dan tidak membuktikan apa-apa.
- [ ] Cek `missing_feature_rows` untuk train dan test.
- [ ] Pastikan `patient_id` cocok saat join.

### Langkah 10. Ablasi dan sanity check
- [ ] Bandingkan skor dengan dan tanpa satu kelompok fitur, misalnya triase. Ini membuktikan kelompok itu berguna dan membantu menjelaskan keputusan di laporan.
- [ ] Sanity check hanya untuk melihat apakah fitur bermakna. Jangan dipakai untuk memilih model akhir atau mengoptimalkan ke public leaderboard.

---

## 5. Pitfall yang sering terjadi

1. **Filter strict membuang data yang sebenarnya valid.** Ini terjadi di GDGoC: mode strict membuang 100% triase karena selisihnya sub-detik. Selalu uji resolusi cutoff di langkah 2.
2. **Koma desimal dibaca sebagai teks.** `18,33` tidak bisa diubah menjadi angka tanpa penanganan khusus.
3. **Koma pada uang bisa berarti pemisah ribuan.** `20,982` adalah 20.982, bukan 20,982 desimal. Aturannya beda dengan angka vital.
4. **Satuan campur dengan label yang salah.** Cek `crosstab` nama dan satuan. Cek juga rentangnya setelah konversi.
5. **Index duplikat merusak agregasi.** `explode` menghasilkan index berulang. Selalu `reset_index(drop=True)` setelah explode.
6. **Nama kolom bertabrakan setelah dibersihkan.** Simbol seperti `+` dan `=` bisa menghapus pembeda. Template sudah menambahkan akhiran otomatis.
7. **Kolom kategori dianggap angka oleh LightGBM.** Pandas versi baru memakai tipe `string`. Cek dengan `is_numeric_dtype`, bukan `dtype == object`.
8. **Kolom kardinalitas tinggi membuat overfit.** `city` dan `county` dibuang di GDGoC. Untuk kolom seperti ini, pertimbangkan target encoding dengan CV.
9. **Join satu-ke-banyak tanpa validasi.** Gunakan `validate=` di `merge`. Template sudah memakainya di `attach_cutoff` dan `main()`.
10. **Pasien kosong terlihat seperti sinyal.** Jika semua fitur nol untuk pasien tertentu, bisa jadi memang tidak ada data, bukan sinyal medis. Catat jumlah baris kosong.

---

## 6. Contoh: menambah jenis tabel baru

Misalkan di GEMASTIK ada tabel `lab_tests.csv` dengan kolom `patient_id`, `test_date`, `test_name`, dan `result`, dengan hasil numerik. Tulis fungsi baru dengan pola yang sama seperti `observations()`:

```python
def lab_tests(t, u, log):
    """Hasil lab per pasien: nilai terakhir dan rata-rata per jenis tes, sebelum cutoff."""
    lab = rf.attach_cutoff(t["lab_tests"], u, "patient_id")               # pasang cutoff lewat pasien
    lab = rf.apply_cutoff(lab, "test_date", log, "lab_tests")             # simpan yang tersedia pada cutoff
    lab["result"] = rf.to_float_decimal_comma(lab["result"])              # ubah format angka
    lab = rf.apply_valid_range(lab, "test_name", "result", LAB_RANGES, log, "lab_tests")  # buang nilai tidak masuk akal
    ok = lab[lab["result"].notna()]                                       # hanya nilai yang valid
    vocab = rf.vocab_from(ok, "test_name", MIN_COUNT)                     # tes yang cukup sering
    out = [(rf.agg_recency(lab, "patient_id", "lab"), "patient", "count")]  # jumlah dan keterbaruan
    out.append((rf.agg_numeric_by_code(ok, "patient_id", "test_name", "result", "lab",
                                       vocab, stats=("last", "mean")), "patient", "num"))  # nilai per tes
    return out
```

Lalu tambahkan `parts += lab_tests(t, u, log)` di `main()`, dan tambahkan `LAB_RANGES` di konfigurasi. Hanya dua tempat itu yang berubah. Semua fungsi lain tetap sama.

---

## 7. Contoh: memakai template di dataset baru dari nol

```python
import relational_features as rf

log = []                                                    # wadah log keputusan
t = rf.load_tables("folder_data")                           # 1. muat semua tabel

idx = t["tabel_indeks"].copy()                              # 2. tabel indeks
idx["cutoff"] = rf.parse_ts(idx["waktu_prediksi"]).dt.floor("s")  # cutoff dalam detik
u = idx[["id_entitas", "id_sampel", "cutoff"]]              # universe: satu baris per sampel

ev = rf.attach_cutoff(t["tabel_event"], u, "id_entitas")    # 3. pasang cutoff ke event
ev = rf.apply_cutoff(ev, "waktu_event", log, "tabel_event") # simpan yang tersedia saja
vocab = rf.vocab_from(ev, "kode", 20)                       # 4. kode yang cukup sering
fitur = rf.agg_code_bag(ev, "id_entitas", "kode", "ev", vocab)  # 5. kolom biner

print(log)                                                  # 6. tampilkan keputusan
```

Untuk jenis tabel yang lebih kompleks, lihat `run_gdgoc.py` sebagai contoh lengkap.

---

## 8. Template untuk laporan final (GEMASTIK)

Gunakan urutan berikut di write-up supaya setiap keputusan terlihat jelas bagi juri:

1. **Pemahaman data.** Grain tiap tabel, jumlah entitas, dan kategori tiap tabel (langkah 1 dan 3).
2. **Desain cutoff.** Hasil uji resolusi, aturan yang dipilih, dan bukti angkanya (langkah 2). Ini bagian yang paling dinilai karena menunjukkan kamu memahami risiko kebocoran.
3. **Pembersihan.** Format angka, satuan, dan rentang. Sertakan jumlah nilai yang diubah dan dibuang (langkah 4 dan 5).
4. **Fitur.** Daftar kelompok fitur dengan alasan tiap kelompok (langkah 6 dan 7).
5. **Validasi.** Hasil hitung ulang mandiri dan pemeriksaan kebocoran (langkah 9).
6. **Ablasi.** Dampak tiap kelompok fitur terhadap skor (langkah 10). Bagian ini yang membuktikan keputusanmu bukan sekadar pilihan acak.
7. **Keterbatasan.** Asumsi yang belum terverifikasi, misalnya keluhan utama dan batas rentang.

Log keputusan dari `log_keputusan.json` bisa langsung dijadikan tabel di bagian 1 sampai 5.

# Penjelasan Baris per Baris: Pipeline Fitur GDGoC

Dokumen ini menjelaskan cara kerja `relational_features.py` (template) dan `run_gdgoc.py` (konfigurasi dataset) statement demi statement. Setiap penjelasan mengutip kode aslinya. Untuk alasan di balik setiap keputusan, lihat `LOG_KEPUTUSAN.md`.

---

## Bagian 0. Gambaran besar

Pipeline ini mengubah banyak tabel relasional menjadi **satu baris fitur per encounter**, lalu melatih model di atasnya. Urutan kerjanya:

1. **Muat tabel.** Semua CSV dibaca ke dalam satu dictionary.
2. **Bangun universe.** Satu baris per encounter yang diprediksi, lengkap dengan `cutoff`.
3. **Hitung fitur per bagian.** Demografi, triase, riwayat, kunjungan sebelumnya, dan observasi, masing-masing diringkas menjadi kolom.
4. **Rakit.** Semua bagian digabung ke level encounter. Nilai kosong diisi sesuai jenisnya.
5. **Bersihkan dan simpan.** Nama kolom dibersihkan, lalu fitur dipasang ke train dan test.
6. **Sanity check.** Macro F1 dihitung dengan dan tanpa triase untuk menguji dampaknya.

Konsep kunci: setiap fungsi yang menghasilkan fitur **hanya melihat data yang sudah tersedia pada cutoff**. Pemeriksaan ini dilakukan di `apply_cutoff()`, dan dijaga oleh `check_no_future()`.

---

## Bagian 1. `relational_features.py` (template)

### 1.1 Konstanta

```python
ONE_SECOND = pd.Timedelta(seconds=1)
```
Satu detik sebagai objek `Timedelta`. Konstanta ini mewakili resolusi `prediction_time`. Event yang terjadi di detik yang sama dengan cutoff dianggap tersedia.

### 1.2 `load_tables(root, exclude)`

1. `tables = {}`: wadah hasil, berupa dictionary dari nama tabel ke DataFrame.
2. `glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)`: mencari semua file `.csv` di `root` dan subfolder-nya. Pola `**` dengan `recursive=True` penting karena data ada di `Riwayat Medis/Riwayat Medis/` dan `Triase/Triase/`.
3. `name = os.path.splitext(os.path.basename(path))[0]`: mengambil nama file tanpa folder dan tanpa `.csv`. Nama ini dipakai sebagai kunci dictionary.
4. `if name in exclude: continue`: melewati `DATA_DICTIONARY` karena itu metadata, bukan data pasien.
5. `tables[name] = pd.read_csv(path)`: membaca file ke DataFrame.

### 1.3 `parse_ts(series)`

Satu baris: `pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")`.
- `format="ISO8601"` membaca format campuran, dengan atau tanpa `Z`, dan dengan atau tanpa milidetik.
- `utc=True` menyamakan semua waktu ke zona UTC. Ini penting agar perbandingan antar tabel benar.
- `errors="coerce"` mengubah teks yang gagal diparse menjadi `NaT`, bukan menghentikan program. Jumlah `NaT` dicatat di log sebagai `ts_unparsed`.

### 1.4 `to_float_decimal_comma(series)`

1. `s = series.astype("string").str.strip()`: ubah ke tipe string dan buang spasi di tepi.
2. `s = s.str.replace(",", ".", regex=False)`: ganti koma dengan titik. Ini mengubah `18,33` menjadi `18.33`. `regex=False` membuat penggantian literal, jadi lebih cepat dan aman.
3. `return pd.to_numeric(s, errors="coerce").astype(float)`: ubah ke angka. Teks yang bukan angka menjadi `NaN`, lalu hasilnya dipaksa ke `float`.

Fungsi ini hanya boleh dipakai untuk kolom yang koma-nya berarti desimal. Pendapatan memakai aturan yang berbeda, lihat `parse_amount`.

### 1.5 `parse_amount(x)`

1. `if pd.isna(x): return np.nan`: nilai kosong langsung dikembalikan sebagai `NaN`.
2. `s = str(x).upper().replace("USD", "").replace("$", "").replace(",", "").strip()`: rantai operasi string.
   - `.upper()` menyamakan huruf, sehingga `k` dan `K` sama.
   - Menghapus `USD` dan `$`.
   - Menghapus koma. Di kolom ini koma adalah pemisah ribuan, bukan desimal.
   - `.strip()` membuang spasi.
3. `mult = 1.0`: pengali default.
4. `if s.endswith("K"): mult, s = 1_000.0, s[:-1]`: jika akhirannya `K`, pengali menjadi 1000 dan huruf `K` dibuang. Contoh: `13.1K` menjadi `13.1` lalu dikali 1000.
5. `try: return float(s) * mult`: ubah ke angka lalu kalikan pengali.
6. `except ValueError: return np.nan`: jika teks tidak bisa diubah menjadi angka, kembalikan `NaN`.

### 1.6 `clean_col_names(df)`

1. `df = df.copy()`: kerja pada salinan agar DataFrame asli tidak berubah.
2. `seen, names = {}, []`: `seen` menghitung berapa kali sebuah nama dasar sudah dipakai. `names` menampung nama baru.
3. `for c in df.columns:`: proses setiap nama kolom.
4. `base = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")`: ganti semua karakter selain huruf, angka, dan `_` dengan `_`, lalu buang `_` di tepi. Contoh: `Heart rate` menjadi `Heart_rate`.
5. `n = seen.get(base, 0)`: ambil berapa kali nama dasar ini sudah muncul sebelumnya.
6. `seen[base] = n + 1`: catat satu kemunculan lagi.
7. `names.append(base if n == 0 else f"{base}_{n + 1}")`: kemunculan pertama memakai nama dasar. Kemunculan berikutnya diberi akhiran `_2`, `_3`, dan seterusnya.
8. `df.columns = names`: pasang nama baru.
9. `assert df.columns.is_unique`: pengaman terakhir untuk memastikan tidak ada nama kembar.

### 1.7 `attach_cutoff(df, index_df, key)`

1. `lookup = index_df[[key, "cutoff"]]`: ambil hanya kolom kunci dan cutoff dari tabel indeks.
2. `return df.merge(lookup, on=key, how="inner", validate="many_to_one")`: gabungkan.
   - `how="inner"` membuang baris yang tidak punya pasangan di indeks. Pasien di luar universe ikut terbuang.
   - `validate="many_to_one"` memastikan setiap `key` di tabel indeks hanya muncul sekali. Kalau tidak, cutoff bisa terduplikasi tanpa disadari.

Fungsi ini menambah kolom `cutoff` ke setiap baris, sehingga `apply_cutoff` bisa membandingkannya.

### 1.8 `apply_cutoff(df, time_col, log, name, strict)`

Ini inti dari aturan point-in-time.

1. `ts = parse_ts(df[time_col])`: ubah kolom waktu event ke datetime UTC.
2. `limit = df["cutoff"] if strict else df["cutoff"] + ONE_SECOND`: pilih batas.
   - Mode default: `cutoff + 1 detik`. Event di detik yang sama tetap lolos.
   - Mode strict: `cutoff`. Event di detik yang sama terbuang.
3. `keep = ts < limit`: boolean per baris. `NaT < apa pun` menghasilkan `False`, jadi waktu yang gagal diparse otomatis dibuang.
4. `log.append({...})`: catat jumlah baris masuk, lolos, dan dibuang, beserta mode yang dipakai. Catatan ini menjadi bukti di laporan.
5. `out = df.loc[keep].copy()`: ambil baris yang lolos. `.copy()` mencegah peringatan pandas tentang slice.
6. `out["_ts"] = ts[keep]`: simpan waktu event sebagai kolom datetime. Kolom ini dipakai oleh `agg_numeric_by_code` untuk mengurutkan.
7. `out["_days_before"] = ((out["cutoff"] - out["_ts"]).dt.total_seconds() / 86400).clip(lower=0)`: hitung jarak dalam hari dari event ke cutoff. `clip(lower=0)` memastikan nilai tidak negatif. Dalam mode default, event di detik yang sama bisa punya selisih sedikit negatif, dan ini dibulatkan ke 0.
8. `return out`: kembalikan tabel yang sudah dipangkas.

### 1.9 `check_no_future(df, log, name)`

1. `bad = int((ts >= df["cutoff"] + ONE_SECOND).sum())`: hitung baris yang waktunya masih di luar batas.
2. `log.append(...)`: catat hasil pemeriksaan.
3. `assert bad == 0, ...`: hentikan program kalau ada kebocoran. Fungsi ini aman dipanggil kapan saja karena `apply_cutoff` sudah memfilter lebih dulu.

Catatan: fungsi ini didefinisikan di template, tetapi di `run_gdgoc.py` belum dipanggil secara eksplisit. Pemeriksaan kebocoran di pipeline dilakukan oleh `apply_cutoff`. Memanggil `check_no_future` setelah setiap tabel adalah pengaman tambahan yang bisa ditambahkan.

### 1.10 `apply_valid_range(df, name_col, value_col, ranges, log, name)`

1. `df = df.copy()`: kerja pada salinan.
2. `for label, (lo, hi) in ranges.items():`: ulangi untuk setiap konsep yang punya batas. `ranges` adalah dictionary dari nama ke pasangan `(min, max)`.
3. `m = (df[name_col] == label) & df[value_col].notna()`: pilih baris dengan nama ini dan nilai yang tidak kosong.
4. `out = m & ((df[value_col] < lo) | (df[value_col] > hi))`: dari baris tersebut, pilih yang berada di luar batas.
5. `log.append(...)`: catat jumlahnya beserta batas yang dipakai.
6. `df.loc[out, value_col] = np.nan`: ganti nilai di luar batas dengan `NaN`. Nilainya tidak diubah ke nilai lain.
7. `return df`: kembalikan tabel.

### 1.11 `vocab_from(df, code_col, min_count)`

1. `vc = df[code_col].value_counts()`: hitung berapa kali setiap kode muncul.
2. `return set(vc[vc >= min_count].index)`: simpan hanya kode yang muncul minimal `min_count` kali, sebagai `set` agar pengecekan `in` cepat.

### 1.12 `agg_recency(df, key, prefix)`

1. `g = df.groupby(key)`: kelompokkan baris per `key`, yaitu per pasien atau per encounter.
2. `pd.DataFrame({...})`: buat tabel dengan dua kolom.
   - `f"{prefix}_n": g.size()`: jumlah baris event per key.
   - `f"{prefix}_days_since_last": g["_days_before"].min()`: jarak terkecil ke cutoff, yaitu event paling baru. Nilai terkecil berarti event paling dekat dengan cutoff.

### 1.13 `agg_code_bag(df, key, code_col, prefix, vocab)`

Membuat satu kolom biner per kode.

1. `sub = df[df[code_col].isin(vocab)]`: ambil hanya baris dengan kode di vocab.
2. `if sub.empty: return pd.DataFrame(index=pd.Index([], name=key))`: jika tidak ada yang lolos, kembalikan tabel kosong dengan index yang benar. Ini mencegah error saat digabung.
3. `ct = pd.crosstab(sub[key], sub[code_col])`: tabel silang. Baris adalah key, kolom adalah kode, dan isinya jumlah kemunculan.
4. `ct = (ct > 0).astype(np.int8)`: ubah jumlah menjadi 0 atau 1. Informasi yang dipakai adalah ada atau tidaknya kode, bukan frekuensinya. Frekuensi sudah ditangkap oleh `_n`.
5. `ct.columns = [f"{prefix}__{c}" for c in ct.columns]`: beri awalan supaya kolom mudah dilacak ke tabel asalnya. Contoh (sebelum nama dibersihkan): `cond__E11.9`.
6. `return ct`: kembalikan matriks biner.

### 1.14 `agg_numeric_by_code(df, key, code_col, value_col, prefix, vocab, stats)`

Menghasilkan statistik numerik per kode, misalnya nilai terakhir dan rata-rata untuk setiap jenis observasi.

1. `sub = df[df[code_col].isin(vocab) & df[value_col].notna()]`: pilih kode di vocab dan nilai yang tidak kosong.
2. `sub = sub.sort_values("_ts")`: urutkan berdasarkan waktu. Ini wajib untuk `last`, karena `last` mengambil nilai paling akhir dalam urutan waktu.
3. `parts = []`: wadah hasil untuk setiap statistik.
4. `for st in stats:`: ulangi untuk setiap statistik, misalnya `last` lalu `mean`.
5. `grouped = sub.groupby([key, code_col])[value_col]`: kelompokkan per pasangan (key, kode).
6. `w = getattr(grouped, st)().unstack(code_col)`: hitung statistik lalu ubah kode menjadi kolom. `getattr(grouped, st)` memanggil `last`, `mean`, `max`, atau `min` secara dinamis.
7. `w.columns = [f"{prefix}__{c}__{st}" for c in w.columns]`: beri nama `prefix__kode__statistik`. Contoh: `vital__Heart_rate__last`.
8. `parts.append(w)`: simpan hasil statistik ini.
9. `return pd.concat(parts, axis=1) if parts else pd.DataFrame()`: gabungkan semua statistik secara kolom. Kalau `stats` kosong, kembalikan tabel kosong.

### 1.15 `explode_tokens(df, key, col, sep, strip_prefix)`

Mengubah teks bebas menjadi satu baris per token.

1. `s = df[col].astype("string")`: pastikan kolom berupa string.
2. `if strip_prefix: s = s.str.replace(strip_prefix, "", regex=False)`: buang awalan tetap. Contoh: `Reports:` dibuang dari `Reports: Fever, Throat symptoms`.
3. `tmp = df[[key]].copy()`: ambil kolom key sebagai dasar.
4. `tmp["token"] = s.str.split(sep)`: pecah teks menjadi list berdasarkan `sep`, misalnya koma. Setiap baris sekarang berisi list.
5. `tmp = tmp.explode("token").dropna(subset=["token"]).reset_index(drop=True)`: ubah list menjadi satu baris per elemen. `dropna` membuang elemen kosong. `reset_index(drop=True)` memberi index baru yang unik. Tanpa ini, `pd.crosstab` gagal karena index yang duplikat. Inilah perbaikan yang kulakukan saat menjalankan pipeline.
6. `tmp["token"] = tmp["token"].astype("string").str.strip().str.lower()`: rapikan spasi dan ubah ke huruf kecil supaya `Fever` dan `fever` dianggap sama.
7. `return tmp[tmp["token"].fillna("") != ""]`: buang token yang kosong setelah dirapikan.

### 1.16 `to_entity_level(part, universe, level)`

Menyamakan level semua fitur menjadi level encounter.

1. `if level == "encounter": return part`: fitur yang sudah per encounter tidak perlu diubah.
2. `aligned = part.reindex(universe["patient_id"].values)`: fitur per pasien diurutkan mengikuti urutan `patient_id` di universe. Pasien yang tidak punya fitur menjadi `NaN`.
3. `aligned.index = universe["encounter_id"].values`: ganti index menjadi `encounter_id`. Karena satu pasien hanya punya satu encounter di universe, pemetaan ini satu-ke-satu.
4. `return aligned`: kembalikan fitur level encounter.

---

## Bagian 2. `run_gdgoc.py` (konfigurasi dan pipeline dataset)

### 2.1 Konfigurasi

- `MIN_COUNT = 20`: ambang minimum kemunculan kode. Keputusan ini dijelaskan di D9.
- `VITAL_RANGES` dan `OBS_RANGES`: dictionary dari nama konsep ke pasangan `(min, max)`. Nama harus persis sama dengan nilai di data. Untuk observasi, nama dicocokkan dengan `observation_name`.
- `HIST`: dictionary dari nama tabel riwayat ke pasangan `(kolom kode, awalan)`. Dipakai di `history()` supaya tidak perlu menulis kode berulang.
- `CATEGORICAL`: daftar kolom teks yang nanti dikonversi menjadi kategori untuk LightGBM.
- `TRIAGE_PREFIXES`: awalan nama kolom yang berasal dari triase. Dipakai untuk uji ablasi, yaitu membandingkan skor dengan dan tanpa triase.

### 2.2 `build_universe(t)`

1. `idx = t["index_encounters"][["patient_id", "encounter_id", "prediction_time"]].copy()`: ambil kolom yang dibutuhkan dari tabel indeks.
2. `idx["cutoff"] = rf.parse_ts(idx["prediction_time"]).dt.floor("s")`: ubah `prediction_time` menjadi datetime lalu potong ke detik penuh. Langkah `floor("s")` ini menyatakan secara eksplisit bahwa cutoff punya resolusi detik.
3. `assert idx["cutoff"].notna().all()`: pastikan tidak ada waktu yang gagal diparse.
4. `assert idx["encounter_id"].is_unique and idx["patient_id"].is_unique`: pastikan satu encounter per pasien. Ini asumsi dasar untuk `to_entity_level`.
5. `return idx`: kembalikan universe.

### 2.3 `demographics(t, u, log)`

1. `p = t["patients"].copy()`: salinan tabel pasien.
2. `p["income"] = p["income_at_visit"].map(rf.parse_amount)`: parse pendapatan baris per baris dengan `parse_amount`.
3. `p["log_income"] = np.log1p(p["income"])`: logaritma dari `1 + pendapatan`. Skala pendapatan sangat lebar, dan log membuatnya lebih mudah dipelajari model. `log1p` aman untuk nol.
4. `log.append({... "income_unparsed" ...})`: catat jumlah nilai yang tidak kosong tetapi gagal diparse. Rumusnya: nilai yang `income` kosong dikurangi nilai yang `income_at_visit`-nya kosong. Hasilnya 0.
5. `log.append({... "decision" ...})`: catat bahwa `city` dan `county` dibuang (D10).
6. `keep = ["patient_id", "age_at_visit", "log_income"] + CATEGORICAL`: daftar kolom yang dipakai.
7. `p = p[keep].merge(u[["patient_id"]], on="patient_id")`: buang kolom lain dan batasi ke pasien di universe.
8. `return p.set_index("patient_id"), "patient", "other"`: kembalikan tiga hal: tabel, level pasien, dan jenis fitur `other`. Jenis `other` artinya kosong tetap `NaN`, bukan 0.

### 2.4 `convert_vital_units(v, log)`

1. `m = (v["vital_name"] == "SpO2") & (v["unit"] == "fraction")`: pilih SpO2 yang satuannya pecahan.
2. `log.append(...)`: catat jumlah baris yang dikonversi.
3. `v.loc[m, "value"] = v.loc[m, "value"] * 100`: kalikan 100. Contoh: `0,96` menjadi `96`.
4. `m = v["vital_name"].isin([...]) & (v["unit"] == "kPa")`: pilih tekanan darah yang satuannya kPa.
5. `v.loc[m, "value"] = v.loc[m, "value"] * 7.50062`: konversi kPa ke mmHg. 1 kPa sama dengan 7,50062 mmHg.

### 2.5 `triage(t, u, log)`

Bagian ini menghasilkan fitur dari empat sumber triase. Semua hasilnya level encounter.

**Vital signs:**
1. `v = rf.attach_cutoff(t["triage_vitals"], u, "encounter_id")`: pasang cutoff lewat encounter. Triase dihubungkan lewat `encounter_id`, bukan `patient_id`, karena triase milik encounter tertentu.
2. `v = rf.apply_cutoff(v, "measured_at", log, "triage_vitals")`: simpan yang tersedia pada cutoff.
3. `v["value"] = rf.to_float_decimal_comma(v["value"])`: ubah koma desimal menjadi titik.
4. `v = convert_vital_units(v, log)`: samakan satuan.
5. `v = rf.apply_valid_range(v, "vital_name", "value", VITAL_RANGES, log, "triage_vitals")`: nilai di luar rentang menjadi NaN.
6. `out.append((rf.agg_numeric_by_code(..., stats=("last",)), "encounter", "num"))`: ambil nilai terakhir tiap jenis vital. Setiap encounter hanya punya satu pengukuran, jadi `last` sama dengan nilai tunggal.

**Gejala:**
7. `s = rf.attach_cutoff(...)` dan `s = rf.apply_cutoff(...)`: pasang cutoff lalu filter.
8. `vocab = rf.vocab_from(s, "symptom_concept", MIN_COUNT)`: gejala yang cukup sering.
9. `out.append((rf.agg_code_bag(...), "encounter", "bag"))`: satu kolom biner per gejala.
10. `out.append((rf.agg_recency(s, "encounter_id", "sym_all"), "encounter", "count"))`: jumlah gejala dan jarak waktunya. Jarak hampir selalu 0 untuk triase.

**Severity:**
11. `sv["severity_score"] = pd.to_numeric(sv["severity_score"], errors="coerce")`: pastikan kolom skor berupa angka. `errors="coerce"` menjadikan teks yang tidak valid sebagai NaN.
12. `vocab = rf.vocab_from(sv[sv["severity_score"].notna()], ...)`: hitung vocab hanya dari baris yang punya skor valid.
13. `out.append((rf.agg_numeric_by_code(..., stats=("max",)), "encounter", "num"))`: skor tertinggi per gejala.
14. `g = sv.groupby("encounter_id")["severity_score"]`: kelompokkan skor per encounter untuk ringkasan keseluruhan.
15. `out.append((pd.DataFrame({"sev_all_max": g.max(), "sev_all_mean": g.mean()}), "encounter", "num"))`: skor maksimum dan rata-rata untuk seluruh gejala pada encounter tersebut.

**Keluhan utama:**
16. `pc = t["presenting_complaints"]`: tabel ini tidak punya kolom waktu.
17. `pc = pc[pc["encounter_id"].isin(u["encounter_id"])]`: pastikan encounter-nya ada di universe.
18. `log.append({... "decision" ...})`: catat asumsi bahwa keluhan dicatat saat triase (D3).
19. `tok = rf.explode_tokens(pc, "encounter_id", "presenting_complaints_text", sep=",", strip_prefix="Reports:")`: pecah teks keluhan menjadi token.
20. `vocab = rf.vocab_from(tok, "token", MIN_COUNT)`: keluhan yang cukup sering.
21. `out.append((rf.agg_code_bag(tok, "encounter_id", "token", "cc", vocab), "encounter", "bag"))`: satu kolom biner per keluhan. Awalan `cc__` berarti *chief complaint*.
22. `return out`.

### 2.6 `history(t, u, log)`

Mengulang langkah yang sama untuk lima tabel riwayat, dengan `for name, (code, pre) in HIST.items():`.

1. `h = rf.attach_cutoff(t[name], u, "patient_id")`: riwayat dihubungkan lewat `patient_id`, karena riwayat milik pasien, bukan encounter tertentu.
2. `h = rf.apply_cutoff(h, "recorded_at", log, name)`: buang yang setelah cutoff.
3. `vocab = rf.vocab_from(h, code, MIN_COUNT)`: kode yang cukup sering.
4. `out.append((rf.agg_recency(h, "patient_id", pre), "patient", "count"))`: jumlah dan jarak waktu.
5. `out.append((rf.agg_code_bag(h, "patient_id", code, pre, vocab), "patient", "bag"))`: satu kolom biner per kode.

### 2.7 `previous_encounters(t, u, log)`

1. `pe = rf.attach_cutoff(t["previous_encounter"], u, "patient_id")`: pasang cutoff.
2. `pe = rf.apply_cutoff(pe, "history_start", log, "previous_encounter")`: kunci waktu untuk kunjungan adalah waktu mulai.
3. `stop = rf.parse_ts(pe["history_stop"])`: waktu selesai kunjungan.
4. `log.append({... "history_stop melewati cutoff" ...})`: pengaman kedua. Kalau waktu selesai melewati cutoff, angkanya tercatat. Hasilnya 0.
5. `pe["dur_days"] = (stop[pe.index] - pe["_ts"]).dt.total_seconds() / 86400`: durasi kunjungan dalam hari. `stop[pe.index]` mengambil waktu selesai untuk baris yang lolos filter.
6. `out = [(rf.agg_recency(pe, "patient_id", "prev"), "patient", "count")]`: jumlah dan jarak waktu kunjungan.
7. `out.append((pe.groupby("patient_id")["dur_days"].mean().rename("prev_dur_days_mean").to_frame(), "patient", "num"))`: rata-rata durasi kunjungan per pasien.
8. `vocab = rf.vocab_from(pe, "history_type_code", MIN_COUNT)` dan `out.append((rf.agg_code_bag(...), "patient", "bag"))`: jenis kunjungan sebagai kolom biner.

### 2.8 `observations(t, u, log)`

Observasi punya dua jenis nilai, sehingga dipisah.

1. `ob = rf.attach_cutoff(...)` dan `ob = rf.apply_cutoff(ob, "observed_at", ...)`: pasang cutoff lalu filter.
2. `out.append((rf.agg_recency(ob, "patient_id", "obs"), "patient", "count"))`: dihitung dari semua observasi, termasuk yang teks.

**Bagian numerik:**
3. `num = ob[ob["value_type"] == "numeric"].copy()`: ambil baris bertipe numerik.
4. `num["value"] = rf.to_float_decimal_comma(num["value"])`: ubah koma desimal.
5. `num = rf.apply_valid_range(num, "observation_name", "value", OBS_RANGES, log, "observations")`: nilai tidak masuk akal menjadi NaN.
6. `numok = num[num["value"].notna()]`: hanya baris dengan nilai valid.
7. `vocab = rf.vocab_from(numok, "observation_code", MIN_COUNT)`: kode yang cukup sering dan punya nilai.
8. `out.append((rf.agg_numeric_by_code(..., stats=("last", "mean")), "patient", "num"))`: nilai terakhir dan rata-rata per kode.

**Bagian teks:**
9. `txt = ob[(ob["value_type"] == "text") & ob["value"].notna()].copy()`: ambil observasi teks yang punya nilai.
10. `txt["token"] = txt["observation_code"].astype(str) + "=" + txt["value"].astype(str).str.strip().str.lower()`: gabungkan kode dan nilai menjadi satu token. Contoh: `72166-2=never smoked tobacco (finding)`. Dengan begitu nilai yang berbeda untuk kode yang sama menjadi kolom yang berbeda.
11. `vocab = rf.vocab_from(txt, "token", MIN_COUNT)` dan `out.append((rf.agg_code_bag(txt, ..., "obstxt", vocab), "patient", "bag"))`: satu kolom biner per token.

### 2.9 `assemble(u, parts)`

Menggabungkan semua bagian menjadi satu tabel.

1. `frames, bag_or_count = [], []`: `frames` menampung tabel yang sudah level encounter. `bag_or_count` menampung nama kolom yang nanti diisi 0.
2. `for df, level, kind in parts:`: ulangi untuk setiap bagian.
3. `aligned = rf.to_entity_level(df, u, level)`: samakan level.
4. `if kind in ("bag", "count"): bag_or_count.extend(aligned.columns)`: catat kolom yang kosong berarti nol.
5. `feat = pd.concat(frames, axis=1)`: gabungkan kolom berdasarkan index `encounter_id`.
6. `feat = feat.loc[:, ~feat.columns.duplicated()]`: buang kolom kembar. Pengecekan ini ada sebagai pengaman.
7. `feat = feat.reindex(u["encounter_id"].values)`: urutkan sesuai universe dan pastikan semua encounter ada. Encounter tanpa data dari bagian mana pun menjadi baris `NaN`.
8. `feat[...] = feat[...].fillna(0)`: isi 0 hanya untuk kolom bag dan count. Kolom numerik tetap `NaN`.
9. `feat.index.name = "encounter_id"`: beri nama index.
10. `return feat.reset_index().copy()`: jadikan `encounter_id` kolom biasa. `.copy()` merapikan struktur internal DataFrame.

### 2.10 `sanity_cv(train_feat, drop_prefixes)`

1. `cols = [...]`: pilih kolom fitur. Kolom identitas dan label dibuang. Kolom dengan awalan di `drop_prefixes` juga dibuang, dipakai untuk uji ablasi.
2. `X = train_feat[cols].copy()`: matriks fitur.
3. `for c in X.columns: if not pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].astype("category")`: kolom teks menjadi kategori. LightGBM membutuhkan tipe ini. Pengecekan `is_numeric_dtype` dipakai karena pandas memakai tipe `string`, bukan `object`.
4. `y = train_feat["target_class"].astype("category")` dan `yc = y.cat.codes`: label dikonversi ke angka 0 sampai 3.
5. `oof = np.zeros((len(X), len(y.cat.categories)))`: wadah prediksi out-of-fold, berukuran jumlah baris dikali jumlah kelas.
6. `skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`: lima fold. `Stratified` menjaga proporsi kelas di tiap fold. `random_state` membuat hasilnya bisa diulang.
7. `for tr, va in skf.split(X, yc):`: untuk tiap fold, `tr` adalah indeks latih dan `va` adalah indeks validasi.
8. `m = lgb.LGBMClassifier(..., class_weight="balanced", ...)`: model boosting pohon. `class_weight="balanced"` memberi bobot lebih besar pada kelas yang jarang.
9. `m.fit(X.iloc[tr], yc.iloc[tr])`: latih pada data latih fold ini.
10. `oof[va] = m.predict_proba(X.iloc[va])`: prediksi probabilitas untuk data validasi fold ini. Setiap baris hanya diprediksi oleh model yang tidak pernah melihatnya.
11. `return f1_score(yc, oof.argmax(1), average="macro"), len(cols)`: ambil kelas dengan probabilitas tertinggi, lalu hitung Macro F1. Kembalikan juga jumlah fitur.

### 2.11 `main(root, out_dir)`

1. `os.makedirs(out_dir, exist_ok=True)`: buat folder output jika belum ada. `exist_ok=True` mencegah error kalau folder sudah ada.
2. `log = []`: wadah semua keputusan dan angka.
3. `t = rf.load_tables(root)` dan `u = build_universe(t)`: muat data dan bangun universe.
4. `parts = [demographics(t, u, log)]` lalu `parts += triage(...)`, `history(...)`, `previous_encounters(...)`, `observations(...)`: kumpulkan semua bagian. Setiap bagian adalah tuple `(df, level, kind)`.
5. Blok `for name, col in [...]`: diagnostik mode strict untuk triase. Hasilnya hanya dicatat ke log dan **tidak dipakai** untuk fitur. Tujuannya membuktikan bahwa mode strict membuang semua triase (D1).
6. `feat = assemble(u, parts)`: rakit semua bagian.
7. `feat = rf.clean_col_names(feat.copy())`: bersihkan nama kolom.
8. `feat = feat.assign(patient_id=u.set_index("encounter_id").loc[feat["encounter_id"], "patient_id"].values)`: tambahkan `patient_id` untuk verifikasi join. `.assign` menambah kolom tanpa mengubah tabel asal.
9. `for name in ("train", "test"):`: proses train dan test dengan cara yang sama.
10. `df = t[name].merge(feat, on="encounter_id", how="left", validate="one_to_one")`: gabungkan fitur ke tabel sampel. `validate="one_to_one"` memastikan tidak ada duplikasi.
11. `assert (df["patient_id_x"] == df["patient_id_y"]).all()`: pengecekan join (D16). Kolom `_x` dari train dan `_y` dari fitur harus sama di setiap baris.
12. `df = df.drop(columns=["patient_id_y"]).rename(columns={"patient_id_x": "patient_id"})`: buang duplikat dan rapikan nama.
13. `df.to_csv(...)`: simpan hasil train atau test.
14. `if name == "train": train_feat = df`: simpan train untuk sanity check.
15. `log.append({... "missing_feature_rows" ...})`: hitung baris yang tidak punya satu pun fitur. Hasilnya 0.
16. `feat.to_csv(...)`: simpan tabel fitur lengkap.
17. `f_full, n_full = sanity_cv(train_feat)` dan `f_notri, n_notri = sanity_cv(train_feat, drop_prefixes=TRIAGE_PREFIXES)`: dua kali sanity check, dengan dan tanpa triase.
18. `json.dump(log, ...)`: simpan semua log ke `log_keputusan.json`.
19. `print(...)`: tampilkan log dan ringkasan di layar.

### 2.12 Blok `if __name__ == "__main__":`

`argparse` membaca `--root` dan `--out` dari command line, lalu memanggil `main`. Blok ini hanya berjalan kalau file dijalankan langsung, bukan saat di-import. Karena itu `relational_features` bisa dipakai dari file lain tanpa menjalankan pipeline.

---

## Bagian 3. Cara membaca tabel hasil

- `features_all.csv`: satu baris per encounter (5.000 baris), berisi 515 fitur ditambah kolom kunci `encounter_id` dan `patient_id`.
- `train_features.csv` dan `test_features.csv`: sampel train dan test dengan fitur sudah terpasang. Kolom `target_class` hanya ada di train.
- Awalan nama kolom menunjukkan asalnya:
  - `age_at_visit`, `log_income`, dan kolom kategori berasal dari demografi.
  - `vital__` berasal dari vital triase.
  - `sym__`, `sym_all_` berasal dari gejala triase.
  - `sev__`, `sev_all_` berasal dari skor keparahan.
  - `cc__` berasal dari keluhan utama.
  - `cond__`, `med__`, `proc__`, `imm__`, `alg__` berasal dari riwayat medis.
  - `prev__` dan `prev_dur_days_mean` berasal dari kunjungan sebelumnya.
  - `obsnum__` berasal dari observasi numerik.
  - `obstxt__` berasal dari observasi teks.
- Akhiran `__last`, `__mean`, `__max` menunjukkan statistik yang dipakai.
- Akhiran `_n` adalah jumlah event, dan `_days_since_last` adalah jarak hari ke event terbaru.

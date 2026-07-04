# Data Dictionary — Dataset Kelulusan Mahasiswa

**Sumber:** UCI Machine Learning Repository  
**Judul:** Predict Students' Dropout and Academic Success  
**URL:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success  
**Lisensi:** CC BY 4.0  
**Jumlah Baris:** 4.424 | **Jumlah Kolom:** 37 (36 fitur + 1 target)  
**Separator:** semicolon (`;`)

---

## Target Variable

| Kolom  | Tipe   | Nilai               | Keterangan                                             |
|--------|--------|---------------------|--------------------------------------------------------|
| Target | string | Graduate / Dropout / Enrolled | Status akhir mahasiswa. Untuk project ini hanya digunakan Graduate dan Dropout (klasifikasi biner). |

Encoding target untuk model: `Graduate = 1`, `Dropout = 0`

---

## Fitur Demografis & Pendaftaran

| No | Kolom                        | Tipe        | Deskripsi                                                             |
|----|------------------------------|-------------|-----------------------------------------------------------------------|
| 1  | Marital status               | Categorical | Status pernikahan (1=Single, 2=Married, 3=Widower, 4=Divorced, 5=Facto union, 6=Legally separated) |
| 2  | Application mode             | Categorical | Mode/jalur pendaftaran (kode numerik)                                 |
| 3  | Application order            | Ordinal     | Urutan pilihan program studi saat mendaftar (1=pilihan utama)         |
| 4  | Course                       | Categorical | Kode program studi yang diambil                                       |
| 5  | Daytime/evening attendance   | Binary      | Waktu kuliah: 1=Siang/Reguler, 0=Malam                               |
| 6  | Previous qualification       | Categorical | Jenis kualifikasi pendidikan sebelumnya (kode numerik)                |
| 7  | Previous qualification (grade)| Numeric    | Nilai kualifikasi pendidikan sebelumnya (0–200)                       |
| 8  | Nacionality                  | Categorical | Kewarganegaraan mahasiswa (kode numerik)                              |
| 9  | Mother's qualification       | Categorical | Tingkat pendidikan ibu (kode numerik)                                 |
| 10 | Father's qualification       | Categorical | Tingkat pendidikan ayah (kode numerik)                                |
| 11 | Mother's occupation          | Categorical | Pekerjaan ibu (kode numerik)                                          |
| 12 | Father's occupation          | Categorical | Pekerjaan ayah (kode numerik)                                         |
| 13 | Admission grade              | Numeric     | Nilai masuk/penerimaan (0–200)                                        |
| 14 | Displaced                    | Binary      | Mahasiswa dari luar daerah: 1=Ya, 0=Tidak                             |
| 15 | Educational special needs    | Binary      | Kebutuhan pendidikan khusus: 1=Ya, 0=Tidak                            |
| 16 | Debtor                       | Binary      | Memiliki tunggakan keuangan: 1=Ya, 0=Tidak                            |
| 17 | Tuition fees up to date      | Binary      | SPP terbayar lunas: 1=Ya, 0=Tidak                                     |
| 18 | Gender                       | Binary      | Jenis kelamin: 1=Laki-laki, 0=Perempuan                               |
| 19 | Scholarship holder           | Binary      | Penerima beasiswa: 1=Ya, 0=Tidak                                      |
| 20 | Age at enrollment            | Numeric     | Usia saat mendaftar (tahun)                                           |
| 21 | International                | Binary      | Mahasiswa internasional: 1=Ya, 0=Tidak                                |

---

## Fitur Akademik — Semester 1

| No | Kolom                                          | Tipe    | Deskripsi                                        |
|----|------------------------------------------------|---------|--------------------------------------------------|
| 22 | Curricular units 1st sem (credited)            | Numeric | SKS yang diakui/dikreditkan dari sem sebelumnya  |
| 23 | Curricular units 1st sem (enrolled)            | Numeric | Jumlah SKS yang diambil di semester 1            |
| 24 | Curricular units 1st sem (evaluations)         | Numeric | Jumlah evaluasi/ujian yang diikuti di semester 1 |
| 25 | Curricular units 1st sem (approved)            | Numeric | Jumlah SKS yang dinyatakan lulus di semester 1   |
| 26 | Curricular units 1st sem (grade)               | Numeric | Rata-rata nilai semester 1 (0–20)                |
| 27 | Curricular units 1st sem (without evaluations) | Numeric | Mata kuliah yang tidak mengikuti evaluasi sem 1  |

---

## Fitur Akademik — Semester 2

| No | Kolom                                          | Tipe    | Deskripsi                                        |
|----|------------------------------------------------|---------|--------------------------------------------------|
| 28 | Curricular units 2nd sem (credited)            | Numeric | SKS yang diakui/dikreditkan dari sem sebelumnya  |
| 29 | Curricular units 2nd sem (enrolled)            | Numeric | Jumlah SKS yang diambil di semester 2            |
| 30 | Curricular units 2nd sem (evaluations)         | Numeric | Jumlah evaluasi/ujian yang diikuti di semester 2 |
| 31 | Curricular units 2nd sem (approved)            | Numeric | Jumlah SKS yang dinyatakan lulus di semester 2   |
| 32 | Curricular units 2nd sem (grade)               | Numeric | Rata-rata nilai semester 2 (0–20)                |
| 33 | Curricular units 2nd sem (without evaluations) | Numeric | Mata kuliah yang tidak mengikuti evaluasi sem 2  |

---

## Fitur Makroekonomi

| No | Kolom             | Tipe    | Deskripsi                                    |
|----|-------------------|---------|----------------------------------------------|
| 34 | Unemployment rate | Numeric | Tingkat pengangguran nasional (%)            |
| 35 | Inflation rate    | Numeric | Tingkat inflasi nasional (%)                 |
| 36 | GDP               | Numeric | Pertumbuhan GDP nasional (%)                 |

---

## Catatan Etika dan Privasi

- Dataset ini telah **dianonimkan** — tidak mengandung nama, NIM, atau identitas pribadi mahasiswa.
- Fitur yang berpotensi sensitif (status pernikahan, gender, kewarganegaraan) digunakan hanya sebagai
  prediktor statistik, bukan untuk diskriminasi.
- Model **tidak boleh** digunakan sebagai satu-satunya dasar keputusan akademik.
- Penggunaan model hanya sebagai **Decision Support System** untuk deteksi dini dan intervensi.
- Hasil prediksi wajib diinterpretasikan oleh dosen wali atau pejabat akademik yang berwenang.

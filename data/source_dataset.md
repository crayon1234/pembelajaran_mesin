# Sumber Dataset

## Informasi Dataset

| Atribut       | Detail                                                                      |
|---------------|-----------------------------------------------------------------------------|
| **Nama**      | Predict Students' Dropout and Academic Success                              |
| **Sumber**    | UCI Machine Learning Repository                                             |
| **URL**       | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| **DOI**       | 10.24432/C5MC89                                                             |
| **Lisensi**   | Creative Commons Attribution 4.0 International (CC BY 4.0)                  |
| **Tahun**     | 2022 (publikasi dataset)                                                    |
| **Penulis**   | M. V. Martins, D. Tolledo, J. Machado, L. M. T. Baptista, V. Realinho      |

## Deskripsi

Dataset ini dikumpulkan dari sebuah institusi pendidikan tinggi di Portugal dan berisi informasi 
tentang mahasiswa yang terdaftar dalam berbagai program sarjana. Dataset mencakup informasi yang 
diketahui pada saat pendaftaran mahasiswa (jalur akademis, demografi, dan faktor sosio-ekonomi) 
dan kinerja akademis mahasiswa pada akhir semester pertama dan kedua.

## Distribusi Target (Original)

| Kelas    | Jumlah | Persentase |
|----------|--------|------------|
| Graduate | 2.209  | 49.93%     |
| Dropout  | 1.421  | 32.12%     |
| Enrolled | 794    | 17.95%     |

**Catatan:** Untuk project ini, kelas "Enrolled" dihapus karena masih dalam masa studi dan belum 
memiliki status kelulusan yang pasti. Hanya kelas **Graduate** dan **Dropout** yang digunakan 
(klasifikasi biner).

## Cara Penggunaan

```python
import pandas as pd
df = pd.read_csv("data/data.csv", sep=';')
df = df[df['Target'].isin(['Graduate', 'Dropout'])]
```

## Etika dan Privasi

- Data telah dianonimkan oleh penyedia dataset asli.
- Tidak mengandung identitas pribadi (nama, NIM, alamat).
- Penggunaan sesuai lisensi CC BY 4.0 dengan atribusi.
- Penggunaan model dibatasi sebagai decision support, bukan keputusan final.

## Referensi

Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). 
*Predicting Student Dropout and Academic Success*. 
Data, 7(11), 146. https://doi.org/10.3390/data7110146

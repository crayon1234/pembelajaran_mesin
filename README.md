# UAS Pembelajaran Mesin — Klasifikasi Kelulusan Mahasiswa

**Mata Kuliah:** Pembelajaran Mesin  
**Dosen:** Junta Zeniarja, M.Kom  
**Universitas:** Universitas Dian Nuswantoro  
**Semester:** Genap 2025/2026  
**Repository:** https://github.com/crayon1234/pembelajaran_mesin

---

## Deskripsi Project

Sistem prediksi status kelulusan mahasiswa menggunakan tiga algoritma klasifikasi:
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes (Gaussian NB)**
- **Support Vector Machine (SVM)**

Dataset: [UCI ML Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)  
Target: **Graduate** (lulus) vs **Dropout** (tidak lulus)

---

## Struktur Project

```
UAS_ML_Kelulusan/
├── data/
│   ├── data.csv                      # Dataset utama (semicolon-separated)
│   ├── data_dictionary.md            # Kamus data / penjelasan fitur
│   └── source_dataset.md             # Informasi sumber & lisensi dataset
├── notebook/
│   └── uas_ml_graduation_knn_nb_svm_optimization.ipynb   # Notebook eksperimen lengkap
├── src/
│   ├── ml_core.py                    # Fungsi inti ML (load, preprocess, train, eval)
│   ├── data_generator.py             # Audit dataset & data dictionary generator
│   ├── train.py                      # Script training pipeline lengkap
│   └── predict.py                    # Script inference (single & batch)
├── models/
│   └── best_student_graduation_model.joblib   # Model terbaik (SVM) + scaler
├── reports/
│   ├── audit_dataset.json            # Hasil audit dataset
│   ├── all_experiment_results.csv    # Semua hasil eksperimen
│   ├── classification_reports.json   # Classification report per model
│   ├── cm_baseline_all.png           # Confusion matrix baseline
│   ├── cm_optimized_all.png          # Confusion matrix optimized
│   ├── metric_comparison.png         # Grafik perbandingan metrik
│   ├── feature_importance.png        # Feature importance plot
│   ├── feature_distributions.png     # Distribusi fitur
│   └── target_distribution.png       # Distribusi target
├── app_streamlit.py                  # Aplikasi web Streamlit
├── app_gradio.py                     # Aplikasi web Gradio
├── requirements.txt                  # Dependensi Python
└── README.md                         # Dokumentasi ini
```

---

## Cara Menjalankan

### 1. Install Dependencies

```
python -m pip install -r requirements.txt
```

### 2. Training Pipeline (hasilkan model + laporan)

> Model sudah tersedia di folder `models/`. Langkah ini hanya perlu dijalankan ulang jika model dihapus atau ingin melatih ulang dari awal.

```
python src/train.py
```

Output: model tersimpan di `models/`, laporan di `reports/`

### 3. Aplikasi Web Streamlit

```
python -m streamlit run app_streamlit.py
```

Buka browser: http://localhost:8501

### 4. Aplikasi Web Gradio

```
python app_gradio.py
```

Buka browser: http://localhost:7860

### 5. Jalankan Notebook (Eksperimen Lengkap)

```
jupyter notebook notebook/uas_ml_graduation_knn_nb_svm_optimization.ipynb
```

atau pakai JupyterLab:

```
jupyter lab
```

Lalu buka file `notebook/uas_ml_graduation_knn_nb_svm_optimization.ipynb` dari browser.

### 6. Prediksi Batch

```
python src/predict.py --batch data/data.csv --output reports/batch_predictions.csv
```

---

## Hasil Eksperimen

| Model | Accuracy | F1-Macro | Balanced Accuracy | ROC-AUC |
|-------|----------|----------|-------------------|---------|
| KNN (Baseline) | 0.854 | 0.842 | 0.833 | ~0.91 |
| Naive Bayes (Baseline) | 0.849 | 0.838 | 0.833 | ~0.92 |
| **SVM (Baseline)** | **0.909** | **0.902** | **0.893** | **~0.96** |
| KNN (Optimized) | 0.873 | 0.861 | 0.849 | ~0.93 |
| Naive Bayes (Optimized) | 0.849 | 0.838 | 0.833 | ~0.92 |
| **SVM (Optimized)** | **0.909** | **0.902** | **0.893** | **~0.96** |
| SVM+FeatureSelection | 0.909 | 0.902 | 0.894 | ~0.96 |

**Model Terbaik: SVM (kernel=rbf, C=1, gamma=scale)**

---

## Teknik Optimasi yang Diterapkan

1. **GridSearchCV** — pencarian hyperparameter terbaik (KNN, NB, SVM)
2. **Stratified K-Fold CV** (k=5) — validasi silang untuk estimasi bias rendah
3. **SMOTE** — oversampling untuk menangani class imbalance
4. **Feature Selection** (Mutual Information, top-15) — reduksi fitur tidak relevan
5. **StandardScaler** — normalisasi fitur untuk KNN & SVM
6. **Macro-F1 & Balanced Accuracy** — metrik utama (tidak bias mayoritas)
7. **Error Analysis** — analisis pola salah prediksi

---

## Random Seed

Semua eksperimen menggunakan `RANDOM_SEED = 42` untuk reproducibility.

---

## Link Repository

https://github.com/crayon1234/pembelajaran_mesin

---

## Disclaimer Etika

> Aplikasi ini adalah **Decision Support System** — bukan alat keputusan final kelulusan.
> Hasil prediksi hanya digunakan untuk deteksi dini dan intervensi akademik.
> Data mahasiswa telah dianonimkan. Identitas pribadi tidak tersimpan dalam sistem.

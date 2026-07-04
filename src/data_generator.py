"""
data_generator.py - Dataset audit, EDA, and pipeline documentation generator
UAS Pembelajaran Mesin - Klasifikasi Kelulusan Mahasiswa
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from ml_core import load_data, RANDOM_SEED


def audit_dataset(df, target_col='Target'):
    """
    Perform full dataset audit and return audit dictionary.
    """
    audit = {}

    # Basic info
    audit['n_rows'] = int(df.shape[0])
    audit['n_cols'] = int(df.shape[1])
    audit['columns'] = df.columns.tolist()

    # Data types
    audit['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Target distribution
    audit['target_distribution'] = df[target_col].value_counts().to_dict()
    audit['target_distribution_pct'] = (
        df[target_col].value_counts(normalize=True).round(4) * 100
    ).to_dict()

    # Missing values
    missing = df.isnull().sum()
    audit['missing_values'] = missing[missing > 0].to_dict()
    audit['total_missing'] = int(missing.sum())
    audit['missing_pct'] = round(missing.sum() / (df.shape[0] * df.shape[1]) * 100, 4)

    # Duplicates
    audit['duplicate_rows'] = int(df.duplicated().sum())

    # Numeric stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    audit['numeric_columns'] = numeric_cols

    # Outlier count via IQR
    outlier_counts = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
        if n_out > 0:
            outlier_counts[col] = n_out
    audit['outlier_counts_iqr'] = outlier_counts

    # Class imbalance ratio
    vc = df[target_col].value_counts()
    if len(vc) >= 2:
        majority = int(vc.iloc[0])
        minority = int(vc.iloc[-1])
        audit['imbalance_ratio'] = round(majority / minority, 4)
    else:
        audit['imbalance_ratio'] = None

    # Potential leakage flag (manual note)
    audit['leakage_notes'] = (
        "Fitur curricular units (approved/grade) berpotensi leakage jika target "
        "diukur pada saat yang sama. Namun karena ini dataset retrospektif (historis), "
        "fitur ini valid sebagai prediktor kelulusan. Tidak ada kolom ID mahasiswa "
        "atau NIM yang dapat membocorkan identitas langsung."
    )

    return audit


def generate_data_dictionary():
    """Return data dictionary as list of dicts."""
    dictionary = [
        {"feature": "Marital status", "type": "categorical", "description": "Status pernikahan mahasiswa (1=Single, 2=Married, ...)"},
        {"feature": "Application mode", "type": "categorical", "description": "Mode aplikasi pendaftaran (kode numerik)"},
        {"feature": "Application order", "type": "ordinal", "description": "Urutan pilihan program studi saat mendaftar"},
        {"feature": "Course", "type": "categorical", "description": "Kode program studi yang diambil"},
        {"feature": "Daytime/evening attendance", "type": "binary", "description": "Kelas pagi/siang (1) atau malam (0)"},
        {"feature": "Previous qualification", "type": "categorical", "description": "Jenis kualifikasi pendidikan sebelumnya"},
        {"feature": "Previous qualification (grade)", "type": "numeric", "description": "Nilai kualifikasi sebelumnya (0-200)"},
        {"feature": "Nacionality", "type": "categorical", "description": "Kewarganegaraan mahasiswa (kode numerik)"},
        {"feature": "Mother's qualification", "type": "categorical", "description": "Tingkat pendidikan ibu"},
        {"feature": "Father's qualification", "type": "categorical", "description": "Tingkat pendidikan ayah"},
        {"feature": "Mother's occupation", "type": "categorical", "description": "Pekerjaan ibu (kode numerik)"},
        {"feature": "Father's occupation", "type": "categorical", "description": "Pekerjaan ayah (kode numerik)"},
        {"feature": "Admission grade", "type": "numeric", "description": "Nilai penerimaan mahasiswa (0-200)"},
        {"feature": "Displaced", "type": "binary", "description": "Apakah mahasiswa mengungsi/berpindah (1=Ya, 0=Tidak)"},
        {"feature": "Educational special needs", "type": "binary", "description": "Kebutuhan pendidikan khusus (1=Ya, 0=Tidak)"},
        {"feature": "Debtor", "type": "binary", "description": "Memiliki utang/tunggakan (1=Ya, 0=Tidak)"},
        {"feature": "Tuition fees up to date", "type": "binary", "description": "SPP terbayar lunas (1=Ya, 0=Tidak)"},
        {"feature": "Gender", "type": "binary", "description": "Jenis kelamin (1=Laki-laki, 0=Perempuan)"},
        {"feature": "Scholarship holder", "type": "binary", "description": "Penerima beasiswa (1=Ya, 0=Tidak)"},
        {"feature": "Age at enrollment", "type": "numeric", "description": "Usia saat mendaftar (tahun)"},
        {"feature": "International", "type": "binary", "description": "Mahasiswa internasional (1=Ya, 0=Tidak)"},
        {"feature": "Curricular units 1st sem (credited)", "type": "numeric", "description": "SKS yang dikreditkan semester 1"},
        {"feature": "Curricular units 1st sem (enrolled)", "type": "numeric", "description": "SKS yang diambil semester 1"},
        {"feature": "Curricular units 1st sem (evaluations)", "type": "numeric", "description": "Jumlah evaluasi/ujian semester 1"},
        {"feature": "Curricular units 1st sem (approved)", "type": "numeric", "description": "SKS lulus semester 1"},
        {"feature": "Curricular units 1st sem (grade)", "type": "numeric", "description": "Rata-rata nilai semester 1"},
        {"feature": "Curricular units 1st sem (without evaluations)", "type": "numeric", "description": "Mata kuliah tanpa evaluasi semester 1"},
        {"feature": "Curricular units 2nd sem (credited)", "type": "numeric", "description": "SKS yang dikreditkan semester 2"},
        {"feature": "Curricular units 2nd sem (enrolled)", "type": "numeric", "description": "SKS yang diambil semester 2"},
        {"feature": "Curricular units 2nd sem (evaluations)", "type": "numeric", "description": "Jumlah evaluasi/ujian semester 2"},
        {"feature": "Curricular units 2nd sem (approved)", "type": "numeric", "description": "SKS lulus semester 2"},
        {"feature": "Curricular units 2nd sem (grade)", "type": "numeric", "description": "Rata-rata nilai semester 2"},
        {"feature": "Curricular units 2nd sem (without evaluations)", "type": "numeric", "description": "Mata kuliah tanpa evaluasi semester 2"},
        {"feature": "Unemployment rate", "type": "numeric", "description": "Tingkat pengangguran nasional (%)"},
        {"feature": "Inflation rate", "type": "numeric", "description": "Tingkat inflasi nasional (%)"},
        {"feature": "GDP", "type": "numeric", "description": "Pertumbuhan GDP nasional (%)"},
        {"feature": "Target", "type": "target", "description": "Kelas target: Graduate (lulus), Dropout (tidak lulus/keluar)"},
    ]
    return dictionary


if __name__ == '__main__':
    import os, sys
    # Run from project root
    df = load_data("data/data.csv")
    print(f"Dataset loaded: {df.shape}")

    audit = audit_dataset(df)
    print(json.dumps({k: v for k, v in audit.items() if k != 'columns'}, indent=2))

    os.makedirs("reports", exist_ok=True)
    with open("reports/audit_dataset.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print("audit_dataset.json saved.")

    dd = generate_data_dictionary()
    dd_df = pd.DataFrame(dd)
    print(dd_df.to_string(index=False))

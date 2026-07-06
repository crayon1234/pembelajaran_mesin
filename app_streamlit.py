"""
app_streamlit.py - Aplikasi Web UI/UX Streamlit
Prediksi Status Kelulusan Mahasiswa
UAS Pembelajaran Mesin - Klasifikasi Kelulusan

Cara menjalankan (dari root project):
    streamlit run app_streamlit.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from ml_core import load_model, load_data

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/best_student_graduation_model.joblib"
DATA_PATH  = "data/data.csv"
AUDIT_PATH = "reports/audit_dataset.json"
RESULTS_PATH = "reports/all_experiment_results.csv"

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1f4e79;
        text-align: center; padding: 10px 0 5px 0;
    }
    .sub-header {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f4ff; border-radius: 10px; padding: 15px;
        text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.07);
    }
    .result-graduate {
        background: #d4edda; border-left: 5px solid #28a745;
        padding: 15px; border-radius: 8px; font-size: 1.1rem;
        color: #1a1a1a;
    }
    .result-dropout {
        background: #f8d7da; border-left: 5px solid #dc3545;
        padding: 15px; border-radius: 8px; font-size: 1.1rem;
        color: #1a1a1a;
    }
    .disclaimer {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 10px 15px; border-radius: 5px; font-size: 0.85rem; color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ─── CACHE LOADERS ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

@st.cache_data(show_spinner=False)
def get_data():
    if os.path.exists(DATA_PATH):
        return load_data(DATA_PATH)
    return None

@st.cache_data(show_spinner=False)
def get_audit():
    if os.path.exists(AUDIT_PATH):
        with open(AUDIT_PATH, encoding='utf-8') as f:
            return json.load(f)
    return None

@st.cache_data(show_spinner=False)
def get_results():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return None

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Graduation_hat.svg/120px-Graduation_hat.svg.png",
             width=80)
    st.markdown("## 🎓 Navigasi")
    page = st.radio(
        "Pilih Halaman:",
        ["🏠 Dashboard Data", "🔮 Prediksi Kelulusan",
         "📊 Evaluasi Model", "ℹ️ Tentang Aplikasi"],
        index=0
    )
    st.markdown("---")
    bundle = get_model()
    if bundle:
        st.success(f"✅ Model: **{bundle.get('model_name', 'Loaded')}**")
    else:
        st.warning("⚠️ Model belum tersedia.\nJalankan `python src/train.py` terlebih dahulu.")

# ─── HEADER ──────────────────────────────────────────────────
st.markdown('<div class="main-header">🎓 Sistem Prediksi Kelulusan Mahasiswa</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning · KNN · Naive Bayes · SVM · Universitas Dian Nuswantoro</div>', unsafe_allow_html=True)
st.markdown("---")

# ════════════════════════════════════════════════════════════
# PAGE 1 - DASHBOARD DATA
# ════════════════════════════════════════════════════════════
if page == "🏠 Dashboard Data":
    st.subheader("📁 Dashboard Dataset & Audit")

    df = get_data()
    audit = get_audit()

    if df is None:
        st.error("Dataset tidak ditemukan di `data/data.csv`.")
    else:
        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Baris", f"{df.shape[0]:,}")
        c2.metric("Total Fitur", str(df.shape[1] - 1))
        c3.metric("Kelas Target", "2 (Graduate / Dropout)")
        c4.metric("Missing Values", str(df.isnull().sum().sum()))

        st.markdown("#### Distribusi Target")
        binary_df = df[df['Target'].isin(['Graduate', 'Dropout'])]
        vc = binary_df['Target'].value_counts()

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(vc.reset_index().rename(columns={'count': 'Jumlah', 'Target': 'Status'}))
        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['#28a745', '#dc3545']
            vc.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
            ax.set_title('Distribusi Kelas Target', fontweight='bold')
            ax.set_ylabel('Jumlah Mahasiswa')
            ax.tick_params(axis='x', rotation=0)
            for bar in ax.patches:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("#### Cuplikan Dataset (5 baris pertama)")
        st.dataframe(df.head())

        st.markdown("#### Statistik Deskriptif (Fitur Numerik)")
        numeric_df = df.select_dtypes(include='number')
        st.dataframe(numeric_df.describe().round(3))

        if audit:
            st.markdown("#### Hasil Audit Data")
            col1, col2, col3 = st.columns(3)
            col1.info(f"**Duplikat:** {audit.get('duplicate_rows', 0)} baris")
            col2.info(f"**Missing:** {audit.get('total_missing', 0)} ({audit.get('missing_pct', 0)}%)")
            col3.info(f"**Imbalance Ratio:** {audit.get('imbalance_ratio', 'N/A')}")
            st.info(f"**Catatan Leakage:** {audit.get('leakage_notes', '-')}")

        # Correlation heatmap
        st.markdown("#### Heatmap Korelasi (Top 12 Fitur Numerik)")
        top_corr = numeric_df.corr().abs().mean().nlargest(12).index
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        sns.heatmap(numeric_df[top_corr].corr(), annot=True, fmt='.2f',
                    cmap='coolwarm', linewidths=0.5, ax=ax2, annot_kws={'size': 8})
        ax2.set_title('Heatmap Korelasi Fitur', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

# ════════════════════════════════════════════════════════════
# PAGE 2 - PREDIKSI
# ════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi Kelulusan":
    st.subheader("🔮 Form Prediksi Kelulusan Mahasiswa")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> Aplikasi ini hanya berfungsi sebagai <em>Decision Support System</em>.
    Hasil prediksi bukan keputusan final kelulusan. Gunakan sebagai alat bantu deteksi dini dan
    intervensi akademik oleh dosen wali atau pihak program studi.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    if bundle is None:
        st.error("Model belum dimuat. Jalankan `python src/train.py` terlebih dahulu.")
    else:
        feature_names = bundle['feature_names']

        st.markdown("#### Isi Data Akademik Mahasiswa")
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            # Academic performance - semester 1
            with c1:
                st.markdown("**Semester 1**")
                cu1_enrolled  = st.number_input("SKS Diambil (Sem.1)", 0, 20, 6)
                cu1_approved  = st.number_input("SKS Lulus (Sem.1)", 0, 20, 5)
                cu1_grade     = st.number_input("Rata-rata Nilai (Sem.1)", 0.0, 20.0, 12.0, 0.1)
                cu1_eval      = st.number_input("Jumlah Evaluasi (Sem.1)", 0, 30, 6)

            # Academic performance - semester 2
            with c2:
                st.markdown("**Semester 2**")
                cu2_enrolled  = st.number_input("SKS Diambil (Sem.2)", 0, 20, 6)
                cu2_approved  = st.number_input("SKS Lulus (Sem.2)", 0, 20, 5)
                cu2_grade     = st.number_input("Rata-rata Nilai (Sem.2)", 0.0, 20.0, 12.0, 0.1)
                cu2_eval      = st.number_input("Jumlah Evaluasi (Sem.2)", 0, 30, 6)

            # Personal & financial
            with c3:
                st.markdown("**Data Pribadi & Keuangan**")
                age           = st.number_input("Usia Saat Daftar", 17, 70, 20)
                admission_gr  = st.number_input("Nilai Penerimaan", 0.0, 200.0, 130.0, 0.1)
                tuition_ok    = st.selectbox("SPP Terbayar?", [1, 0], format_func=lambda x: "Ya" if x else "Tidak")
                scholarship   = st.selectbox("Penerima Beasiswa?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
                debtor        = st.selectbox("Memiliki Tunggakan?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
                gender        = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x else "Perempuan")

            submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

        if submitted:
            # Load median values dari data asli sebagai default realistis
            import pandas as _pd
            try:
                df_ref = load_data(DATA_PATH)
                df_ref = df_ref[df_ref['Target'].isin(['Graduate','Dropout'])]
                df_ref.columns = [c.strip().replace('\t','') for c in df_ref.columns]
                feature_names_clean = [f.strip() for f in feature_names]
                medians = df_ref[feature_names_clean].median().to_dict()
            except Exception:
                medians = {}
                feature_names_clean = [f.strip() for f in feature_names]

            # Mulai dari median, lalu override dengan input user
            input_dict = {f: medians.get(f, 0.0) for f in feature_names_clean}
            input_dict.update({
                'Curricular units 1st sem (enrolled)': cu1_enrolled,
                'Curricular units 1st sem (approved)': cu1_approved,
                'Curricular units 1st sem (grade)': cu1_grade,
                'Curricular units 1st sem (evaluations)': cu1_eval,
                'Curricular units 2nd sem (enrolled)': cu2_enrolled,
                'Curricular units 2nd sem (approved)': cu2_approved,
                'Curricular units 2nd sem (grade)': cu2_grade,
                'Curricular units 2nd sem (evaluations)': cu2_eval,
                'Age at enrollment': age,
                'Admission grade': admission_gr,
                'Tuition fees up to date': tuition_ok,
                'Scholarship holder': scholarship,
                'Debtor': debtor,
                'Gender': gender,
            })

            try:
                pipeline = bundle.get('pipeline')
                model    = bundle.get('model')
                scaler   = bundle.get('scaler')
                x_df = _pd.DataFrame([input_dict], columns=feature_names_clean)

                if pipeline is not None:
                    # Bundle baru: pakai Pipeline langsung
                    pred = pipeline.predict(x_df)[0]
                    prob = pipeline.predict_proba(x_df)[0]
                else:
                    # Bundle lama: pakai scaler + model terpisah
                    x_sc = scaler.transform(x_df)
                    pred = model.predict(x_sc)[0]
                    prob = model.predict_proba(x_sc)[0]

                label  = 'Graduate' if pred == 1 else 'Dropout'
                p_grad = float(prob[1])
                p_drop = float(prob[0])
            except Exception as e:
                st.error(f"❌ Error saat prediksi: {e}")
                st.stop()

            st.markdown("---")
            st.markdown("### Hasil Prediksi")
            col_res, col_chart = st.columns([1, 1])

            with col_res:
                if label == 'Graduate':
                    st.markdown(f"""
                    <div class="result-graduate">
                    ✅ <strong>Prediksi: LULUS TEPAT WAKTU</strong><br>
                    Probabilitas Graduate: <strong>{p_grad:.1%}</strong><br>
                    Probabilitas Dropout: <strong>{p_drop:.1%}</strong>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-dropout">
                    ⚠️ <strong>Prediksi: RISIKO TIDAK LULUS / DROPOUT</strong><br>
                    Probabilitas Dropout: <strong>{p_drop:.1%}</strong><br>
                    Probabilitas Graduate: <strong>{p_grad:.1%}</strong>
                    </div>""", unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer" style="margin-top:10px">
                Hasil ini hanya untuk keperluan <strong>deteksi dini</strong> dan 
                intervensi akademik. Bukan keputusan final kelulusan.
                </div>""", unsafe_allow_html=True)

            with col_chart:
                fig, ax = plt.subplots(figsize=(4, 3))
                bars = ax.barh(['Graduate', 'Dropout'], [p_grad, p_drop],
                               color=['#28a745', '#dc3545'], edgecolor='white')
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probabilitas')
                ax.set_title('Distribusi Probabilitas', fontweight='bold')
                for bar, val in zip(bars, [p_grad, p_drop]):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:.1%}', va='center', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

# ════════════════════════════════════════════════════════════
# PAGE 3 - EVALUASI MODEL
# ════════════════════════════════════════════════════════════
elif page == "📊 Evaluasi Model":
    st.subheader("📊 Evaluasi & Perbandingan Model")

    results_df = get_results()
    if results_df is None:
        st.warning("File hasil eksperimen tidak ditemukan. Jalankan `python src/train.py` terlebih dahulu.")
    else:
        # Display table
        st.markdown("#### Tabel Komparasi Semua Eksperimen")
        display_cols = ['model', 'accuracy', 'f1_macro', 'balanced_accuracy',
                        'precision_macro', 'recall_macro', 'roc_auc', 'cv_f1_mean']
        st.dataframe(
            results_df[display_cols].style
            .highlight_max(subset=['accuracy', 'f1_macro', 'balanced_accuracy'],
                           color='#d4edda', axis=0)
            .format({c: '{:.4f}' for c in display_cols if c not in ['model', 'roc_auc']}),
            use_container_width=True
        )

        # Bar chart comparison
        st.markdown("#### Grafik Perbandingan Metrik")
        metrics_to_plot = ['accuracy', 'f1_macro', 'balanced_accuracy']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        palette = sns.color_palette("husl", len(results_df))
        for i, m in enumerate(metrics_to_plot):
            ax = axes[i]
            bars = ax.bar(results_df['model'], results_df[m], color=palette, edgecolor='white')
            ax.set_title(m.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylabel('Score')
            for bar, val in zip(bars, results_df[m]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        plt.suptitle('Perbandingan Model: Baseline vs Optimized', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Confusion matrix images
        st.markdown("#### Confusion Matrix (Model Teroptimasi)")
        cm_files = [f for f in os.listdir('reports') if f.startswith('cm_') and f.endswith('.png')] \
                   if os.path.exists('reports') else []
        if cm_files:
            cols = st.columns(min(len(cm_files), 3))
            for i, fname in enumerate(cm_files):
                with cols[i % 3]:
                    st.image(f"reports/{fname}", caption=fname.replace('cm_', '').replace('.png', '').replace('_', ' '))
        else:
            st.info("Confusion matrix plots akan tersedia setelah training selesai.")

        # Metric comparison image
        if os.path.exists('reports/metric_comparison.png'):
            st.markdown("#### Grafik Perbandingan Komprehensif")
            st.image('reports/metric_comparison.png', use_column_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 - TENTANG
# ════════════════════════════════════════════════════════════
elif page == "ℹ️ Tentang Aplikasi":
    st.subheader("ℹ️ Tentang Aplikasi")

    st.markdown("""
    ### Sistem Prediksi Kelulusan Mahasiswa
    
    Aplikasi ini dikembangkan sebagai bagian dari **Ujian Akhir Semester** mata kuliah 
    **Pembelajaran Mesin** — Universitas Dian Nuswantoro.

    ---
    
    #### Tujuan
    Membangun sistem klasifikasi untuk memprediksi status kelulusan mahasiswa: 
    **Graduate** (Lulus Tepat Waktu) atau **Dropout** (Tidak Lulus / Keluar).
    
    #### Dataset
    - **Sumber:** UCI Machine Learning Repository — *Predict Students' Dropout and Academic Success*
    - **Jumlah Data:** ±4.424 baris, 36 fitur prediktor
    - **Target:** Graduate / Dropout (binary classification)
    
    #### Algoritma yang Digunakan
    | Algoritma | Deskripsi |
    |---|---|
    | **KNN** | K-Nearest Neighbors — klasifikasi berdasarkan kedekatan tetangga |
    | **Naive Bayes** | Gaussian Naive Bayes — berdasarkan teorema Bayes |
    | **SVM** | Support Vector Machine — mencari hyperplane optimal |
    
    #### Teknik Optimasi
    - GridSearchCV dengan Stratified K-Fold CV (k=5)
    - SMOTE untuk class imbalance handling
    - Feature Selection (Mutual Information)
    - Evaluasi: Macro-F1, Balanced Accuracy, ROC-AUC
    
    #### Disclaimer Etika
    > Aplikasi ini **BUKAN** alat keputusan final. Hasil prediksi hanya digunakan sebagai 
    > **Decision Support** untuk membantu dosen wali atau program studi dalam mendeteksi 
    > mahasiswa berisiko secara dini dan melakukan intervensi akademik yang tepat.
    > Data mahasiswa telah dianonimkan dan tidak mengandung identitas pribadi.
    
    ---
    **Dikembangkan untuk UAS Pembelajaran Mesin 2025/2026**  
    Dosen: Junta Zeniarja, M.Kom | Universitas Dian Nuswantoro
    """)

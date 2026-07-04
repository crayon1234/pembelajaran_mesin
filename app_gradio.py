"""
app_gradio.py - Aplikasi Web UI/UX Gradio
Prediksi Status Kelulusan Mahasiswa
UAS Pembelajaran Mesin - Klasifikasi Kelulusan

Cara menjalankan (dari root project):
    python app_gradio.py
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

MODEL_PATH = "models/best_student_graduation_model.joblib"


def load_bundle():
    try:
        from ml_core import load_model
        return load_model(MODEL_PATH)
    except Exception as e:
        return None


def predict_graduation(
    cu1_enrolled, cu1_approved, cu1_grade, cu1_eval,
    cu2_enrolled, cu2_approved, cu2_grade, cu2_eval,
    age, admission_grade, tuition_ok, scholarship, debtor, gender
):
    bundle = load_bundle()
    if bundle is None:
        return "❌ Model tidak ditemukan. Jalankan python src/train.py terlebih dahulu.", ""

    feature_names = bundle['feature_names']
    model  = bundle['model']
    scaler = bundle['scaler']

    input_dict = {f: 0.0 for f in feature_names}
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
        'Admission grade': admission_grade,
        'Tuition fees up to date': 1 if tuition_ok == "Ya" else 0,
        'Scholarship holder': 1 if scholarship == "Ya" else 0,
        'Debtor': 1 if debtor == "Ya" else 0,
        'Gender': 1 if gender == "Laki-laki" else 0,
    })

    x = np.array([input_dict.get(f, 0.0) for f in feature_names]).reshape(1, -1)
    x_sc = scaler.transform(x)
    pred  = model.predict(x_sc)[0]
    prob  = model.predict_proba(x_sc)[0]
    label = 'Graduate' if pred == 1 else 'Dropout'
    p_grad = float(prob[1])
    p_drop = float(prob[0])

    if label == 'Graduate':
        result = f"✅ **LULUS TEPAT WAKTU (Graduate)**\nProbabilitas: {p_grad:.1%}"
    else:
        result = f"⚠️ **RISIKO DROPOUT / TIDAK LULUS**\nProbabilitas Dropout: {p_drop:.1%}"

    detail = (
        f"Probabilitas Graduate : {p_grad:.4f} ({p_grad:.1%})\n"
        f"Probabilitas Dropout  : {p_drop:.4f} ({p_drop:.1%})\n\n"
        f"⚠️ Disclaimer: Hasil ini hanya untuk deteksi dini dan intervensi akademik.\n"
        f"Bukan keputusan final kelulusan mahasiswa."
    )
    return result, detail


def build_interface():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio tidak terinstall. Install dengan: python -m pip install gradio")
        return None

    with gr.Blocks(title="Prediksi Kelulusan Mahasiswa", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎓 Sistem Prediksi Kelulusan Mahasiswa
        **Machine Learning · KNN · Naive Bayes · SVM · Universitas Dian Nuswantoro**
        
        > ⚠️ Aplikasi ini adalah *Decision Support System* — bukan alat keputusan final kelulusan.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📚 Data Akademik Semester 1")
                cu1_enrolled = gr.Slider(0, 20, value=6, step=1, label="SKS Diambil (Sem.1)")
                cu1_approved = gr.Slider(0, 20, value=5, step=1, label="SKS Lulus (Sem.1)")
                cu1_grade    = gr.Slider(0.0, 20.0, value=12.0, step=0.1, label="Rata-rata Nilai (Sem.1)")
                cu1_eval     = gr.Slider(0, 30, value=6, step=1, label="Jumlah Evaluasi (Sem.1)")

            with gr.Column():
                gr.Markdown("### 📚 Data Akademik Semester 2")
                cu2_enrolled = gr.Slider(0, 20, value=6, step=1, label="SKS Diambil (Sem.2)")
                cu2_approved = gr.Slider(0, 20, value=5, step=1, label="SKS Lulus (Sem.2)")
                cu2_grade    = gr.Slider(0.0, 20.0, value=12.0, step=0.1, label="Rata-rata Nilai (Sem.2)")
                cu2_eval     = gr.Slider(0, 30, value=6, step=1, label="Jumlah Evaluasi (Sem.2)")

            with gr.Column():
                gr.Markdown("### 👤 Data Pribadi & Keuangan")
                age           = gr.Slider(17, 70, value=20, step=1, label="Usia Saat Daftar")
                admission_gr  = gr.Slider(0.0, 200.0, value=130.0, step=0.1, label="Nilai Penerimaan")
                tuition_ok    = gr.Radio(["Ya", "Tidak"], value="Ya", label="SPP Terbayar?")
                scholarship   = gr.Radio(["Ya", "Tidak"], value="Tidak", label="Penerima Beasiswa?")
                debtor        = gr.Radio(["Ya", "Tidak"], value="Tidak", label="Memiliki Tunggakan?")
                gender        = gr.Radio(["Laki-laki", "Perempuan"], value="Laki-laki", label="Jenis Kelamin")

        predict_btn = gr.Button("🔮 Prediksi Kelulusan", variant="primary", size="lg")

        with gr.Row():
            result_out = gr.Markdown(label="Hasil Prediksi")
            detail_out = gr.Textbox(label="Detail Probabilitas", lines=5)

        predict_btn.click(
            fn=predict_graduation,
            inputs=[
                cu1_enrolled, cu1_approved, cu1_grade, cu1_eval,
                cu2_enrolled, cu2_approved, cu2_grade, cu2_eval,
                age, admission_gr, tuition_ok, scholarship, debtor, gender
            ],
            outputs=[result_out, detail_out]
        )

        gr.Markdown("""
        ---
        **Dataset:** UCI ML Repository — Predict Students' Dropout and Academic Success  
        **Algoritma:** KNN, Naive Bayes, SVM (dengan GridSearchCV + SMOTE + Feature Selection)  
        **UAS Pembelajaran Mesin 2025/2026 | Dosen: Junta Zeniarja, M.Kom | UDINUS**
        """)

    return demo


if __name__ == '__main__':
    demo = build_interface()
    if demo:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    else:
        print("Gagal memuat antarmuka Gradio.")

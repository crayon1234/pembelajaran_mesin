"""
train.py - Full training pipeline: baseline + optimization + save model
UAS Pembelajaran Mesin - Klasifikasi Kelulusan Mahasiswa

Usage (from project root):
    python src/train.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Allow import from src/ when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ml_core import (
    load_data, preprocess, build_baseline_models, build_param_grids,
    evaluate_model, run_cross_validation, optimize_model, select_features,
    save_model, RANDOM_SEED
)
from data_generator import audit_dataset, generate_data_dictionary

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm),
                                   display_labels=['Dropout', 'Graduate'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metric_comparison(results_df, path):
    metrics = ['accuracy', 'f1_macro', 'balanced_accuracy', 'precision_macro', 'recall_macro']
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for i, m in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(results_df['model'], results_df[m], color=colors[:len(results_df)])
        ax.set_title(m.replace('_', ' ').title(), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, results_df[m]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.suptitle('Baseline vs Optimized Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    print("=" * 60)
    print("  UAS ML - Student Graduation Classification")
    print("  Training Pipeline")
    print("=" * 60)

    # ── LOAD DATA ──────────────────────────────────────────────
    print("\n[1] Loading and auditing dataset...")
    df = load_data("data/data.csv")
    audit = audit_dataset(df)
    with open("reports/audit_dataset.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"    Rows: {audit['n_rows']} | Cols: {audit['n_cols']}")
    print(f"    Target distribution: {audit['target_distribution']}")
    print(f"    Duplicates: {audit['duplicate_rows']} | Missing: {audit['total_missing']}")
    print(f"    Imbalance ratio: {audit['imbalance_ratio']}")
    print("    audit_dataset.json saved.")

    # ── PREPROCESS ─────────────────────────────────────────────
    print("\n[2] Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_names, X_train_raw, X_test_raw = \
        preprocess(df, random_state=RANDOM_SEED)
    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Train class dist: {np.bincount(y_train)} (after SMOTE)")

    # ── BASELINE MODELS ────────────────────────────────────────
    print("\n[3] Training baseline models...")
    baselines = build_baseline_models()
    baseline_results = []
    cm_plots = {}

    for name, model in baselines.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name=f'{name} (Baseline)')
        cv_mean, cv_std = run_cross_validation(model, X_train, y_train, cv=5)
        metrics['cv_f1_mean'] = cv_mean
        metrics['cv_f1_std'] = cv_std
        baseline_results.append(metrics)
        print(f"    {name}: Acc={metrics['accuracy']} | F1={metrics['f1_macro']} "
              f"| BalAcc={metrics['balanced_accuracy']} | CV-F1={cv_mean}±{cv_std}")

    # Save baseline classification reports
    classification_reports = {'baseline': {r['model']: r['classification_report'] for r in baseline_results}}

    # ── OPTIMIZATION ───────────────────────────────────────────
    print("\n[4] Optimizing models (GridSearchCV)...")
    param_grids = build_param_grids()
    clf_map = {
        'KNN': baselines['KNN'].__class__(),
        'Naive Bayes': baselines['Naive Bayes'].__class__(),
        'SVM': baselines['SVM'].__class__(probability=True, random_state=RANDOM_SEED)
    }

    optimized_results = []
    best_models = {}

    for name, model in clf_map.items():
        print(f"    Optimizing {name}...", end=' ')
        best_model, best_params = optimize_model(
            model, param_grids[name], X_train, y_train, method='grid', cv=5
        )
        metrics = evaluate_model(best_model, X_test, y_test, model_name=f'{name} (Optimized)')
        cv_mean, cv_std = run_cross_validation(best_model, X_train, y_train, cv=5)
        metrics['cv_f1_mean'] = cv_mean
        metrics['cv_f1_std'] = cv_std
        metrics['best_params'] = best_params
        optimized_results.append(metrics)
        best_models[name] = best_model
        print(f"F1={metrics['f1_macro']} | Params: {best_params}")

    classification_reports['optimized'] = {r['model']: r['classification_report'] for r in optimized_results}

    # ── FEATURE SELECTION EXPERIMENT ──────────────────────────
    print("\n[5] Feature selection experiment (top-15 mutual info)...")
    X_train_sel, X_test_sel, selector = select_features(X_train, y_train, X_test, k=15)
    selected_idx = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_idx]
    print(f"    Selected features: {selected_features}")

    # Retrain best model (SVM) with selected features
    svm_sel = clf_map['SVM'].__class__(probability=True, random_state=RANDOM_SEED)
    best_svm_sel, _ = optimize_model(svm_sel, param_grids['SVM'], X_train_sel, y_train, method='grid', cv=5)
    metrics_sel = evaluate_model(best_svm_sel, X_test_sel, y_test, model_name='SVM (FeatureSelection)')
    optimized_results.append(metrics_sel)
    print(f"    SVM+FeatureSel: Acc={metrics_sel['accuracy']} | F1={metrics_sel['f1_macro']}")

    # ── COMPILE ALL RESULTS ────────────────────────────────────
    all_results = baseline_results + optimized_results
    results_summary = []
    for r in all_results:
        results_summary.append({
            'model': r['model'],
            'accuracy': r['accuracy'],
            'precision_macro': r['precision_macro'],
            'recall_macro': r['recall_macro'],
            'f1_macro': r['f1_macro'],
            'balanced_accuracy': r['balanced_accuracy'],
            'roc_auc': r.get('roc_auc', 'N/A'),
            'cv_f1_mean': r.get('cv_f1_mean', 'N/A'),
            'cv_f1_std': r.get('cv_f1_std', 'N/A'),
        })

    results_df = pd.DataFrame(results_summary)
    results_df.to_csv("reports/all_experiment_results.csv", index=False)
    print("\n    all_experiment_results.csv saved.")
    print(results_df[['model', 'accuracy', 'f1_macro', 'balanced_accuracy']].to_string(index=False))

    # Save classification reports
    with open("reports/classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(classification_reports, f, ensure_ascii=False, indent=2)
    print("    classification_reports.json saved.")

    # ── PLOTS ──────────────────────────────────────────────────
    print("\n[6] Generating plots...")
    # Confusion matrices for optimized models
    for r in optimized_results:
        if 'FeatureSelection' not in r['model']:
            safe_name = r['model'].replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')
            plot_confusion_matrix(r['confusion_matrix'], r['model'],
                                  f"reports/cm_{safe_name}.png")

    # Metric comparison chart
    comp_df = pd.DataFrame([
        {k: v for k, v in r.items() if k in
         ['model', 'accuracy', 'f1_macro', 'balanced_accuracy', 'precision_macro', 'recall_macro']}
        for r in all_results if 'FeatureSelection' not in r['model']
    ])
    plot_metric_comparison(comp_df, "reports/metric_comparison.png")
    print("    Plots saved to reports/")

    # ── DETERMINE BEST MODEL & SAVE ───────────────────────────
    print("\n[7] Selecting and saving best model...")
    # Best = highest f1_macro among optimized models
    optimized_df = pd.DataFrame([
        {'name': k, 'f1': evaluate_model(v, X_test, y_test)['f1_macro']}
        for k, v in best_models.items()
    ])
    best_name = optimized_df.loc[optimized_df['f1'].idxmax(), 'name']
    best_final_model = best_models[best_name]
    print(f"    Best model: {best_name}")

    # Bundle: model + scaler + feature_names
    bundle = {
        'model': best_final_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_name': best_name,
        'selected_features': selected_features
    }
    save_model(bundle, "models/best_student_graduation_model.joblib")

    print("\n" + "=" * 60)
    print("  Training pipeline complete!")
    print(f"  Best model: {best_name}")
    print("=" * 60)


if __name__ == '__main__':
    main()

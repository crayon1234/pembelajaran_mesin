"""
ml_core.py - Core Machine Learning utilities
UAS Pembelajaran Mesin - Klasifikasi Kelulusan Mahasiswa
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, balanced_accuracy_score,
                              confusion_matrix, classification_report, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

RANDOM_SEED = 42
DATA_PATH = "data/data.csv"
MODEL_PATH = "models/best_student_graduation_model.joblib"

# ── 1. DATA LOADING ─────────────────────────────────────────────────────────

def load_data(path=DATA_PATH):
    """Load dataset with semicolon separator."""
    df = pd.read_csv(path, sep=';')
    # Clean column names
    df.columns = [c.strip().replace('\t', '') for c in df.columns]
    return df


# ── 2. PREPROCESSING ────────────────────────────────────────────────────────

def preprocess(df, target_col='Target', test_size=0.2, random_state=RANDOM_SEED):
    """
    Full preprocessing pipeline:
    - Filter only Graduate/Dropout (binary classification)
    - Drop duplicates
    - Encode target: Graduate=1, Dropout=0
    - Feature/target split
    - Train-test stratified split
    - SMOTE on training set only
    - StandardScaler
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Keep only Graduate and Dropout (binary problem)
    df = df[df[target_col].isin(['Graduate', 'Dropout'])].copy()
    df = df.drop_duplicates()

    # Encode target
    df[target_col] = df[target_col].map({'Graduate': 1, 'Dropout': 0})

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # SMOTE on training only
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_res, y_test, scaler, feature_names, X_train, X_test


# ── 3. MODEL BUILDING ────────────────────────────────────────────────────────

def build_baseline_models():
    """Return dict of baseline classifiers with default params."""
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(var_smoothing=1e-9),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RANDOM_SEED)
    }


def build_param_grids():
    """Hyperparameter grids for GridSearchCV."""
    return {
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'Naive Bayes': {
            'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto']
        }
    }


# ── 4. EVALUATION ────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """Compute full evaluation metrics."""
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    metrics = {
        'model': model_name,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision_macro': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'recall_macro': round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'f1_macro': round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'balanced_accuracy': round(balanced_accuracy_score(y_test, y_pred), 4),
        'roc_auc': round(auc, 4) if auc is not None else 'N/A',
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred,
                                                        target_names=['Dropout', 'Graduate'],
                                                        zero_division=0)
    }
    return metrics


def run_cross_validation(model, X, y, cv=5):
    """Run stratified k-fold CV and return mean/std macro-F1."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
    return round(scores.mean(), 4), round(scores.std(), 4)


# ── 5. OPTIMIZATION ──────────────────────────────────────────────────────────

def optimize_model(model, param_grid, X_train, y_train, method='grid', cv=5):
    """
    Run GridSearchCV or RandomizedSearchCV.
    Returns best estimator and best params.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    if method == 'grid':
        searcher = GridSearchCV(model, param_grid, cv=skf,
                                scoring='f1_macro', n_jobs=-1, verbose=0)
    else:
        searcher = RandomizedSearchCV(model, param_grid, cv=skf,
                                      scoring='f1_macro', n_iter=20,
                                      n_jobs=-1, random_state=RANDOM_SEED, verbose=0)
    searcher.fit(X_train, y_train)
    return searcher.best_estimator_, searcher.best_params_


# ── 6. FEATURE SELECTION ─────────────────────────────────────────────────────

def select_features(X_train, y_train, X_test, k=15):
    """Select top-k features using mutual information."""
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)
    return X_train_sel, X_test_sel, selector


# ── 7. SAVE / LOAD MODEL ────────────────────────────────────────────────────

def save_model(obj, path=MODEL_PATH):
    """Save model/pipeline with joblib."""
    joblib.dump(obj, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH):
    """Load model/pipeline from joblib file."""
    return joblib.load(path)


# ── 8. INFERENCE ─────────────────────────────────────────────────────────────

def predict_single(model, scaler, feature_values, feature_names):
    """
    Predict a single sample.
    feature_values: list of raw feature values (unscaled)
    Returns: label string and probability
    """
    x = np.array(feature_values).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    prob = model.predict_proba(x_scaled)[0]
    label = 'Graduate' if pred == 1 else 'Dropout'
    return label, prob.tolist()

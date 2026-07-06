"""
retrain_fix.py - Retrain with proper sklearn Pipeline to fix prediction bug
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                              roc_auc_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE

SEED = 42

# 1. Load
df = pd.read_csv('data/data.csv', sep=';')
df.columns = [c.strip().replace('\t','') for c in df.columns]
df = df[df['Target'].isin(['Graduate','Dropout'])].copy().drop_duplicates()
df['Target'] = df['Target'].map({'Graduate':1,'Dropout':0})

X = df.drop(columns=['Target'])
y = df['Target']
feature_names = X.columns.tolist()
print(f"Dataset: {X.shape} | Classes: {y.value_counts().to_dict()}")

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

# 3. SMOTE on train only
smote = SMOTE(random_state=SEED)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_res.shape} | {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

# 4. Build sklearn Pipeline (scaler + SVM in one object)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1, gamma='auto', probability=True, random_state=SEED))
])

# 5. GridSearch
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
param_grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 'auto'], 'svm__kernel': ['rbf']}
gs = GridSearchCV(pipe, param_grid, cv=skf, scoring='f1_macro', n_jobs=-1)
gs.fit(X_train_res, y_train_res)
best_pipe = gs.best_estimator_
print(f"Best params: {gs.best_params_}")

# 6. Evaluate
y_pred = best_pipe.predict(X_test)
y_prob = best_pipe.predict_proba(X_test)[:,1]
print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Macro       : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Balanced Acc   : {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC        : {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Dropout','Graduate']))

# 7. Verify predictions manually
print("\n=== MANUAL VERIFICATION ===")
# Dropout case
test_dropout = pd.DataFrame([{f: 0.0 for f in feature_names}])
test_dropout['Curricular units 1st sem (enrolled)'] = 6
test_dropout['Curricular units 1st sem (approved)'] = 1
test_dropout['Curricular units 1st sem (grade)']    = 5.0
test_dropout['Curricular units 1st sem (evaluations)'] = 2
test_dropout['Curricular units 2nd sem (enrolled)'] = 6
test_dropout['Curricular units 2nd sem (approved)'] = 0
test_dropout['Curricular units 2nd sem (grade)']    = 0.0
test_dropout['Curricular units 2nd sem (evaluations)'] = 1
test_dropout['Age at enrollment']      = 28
test_dropout['Admission grade']        = 105.0
test_dropout['Tuition fees up to date'] = 0
test_dropout['Scholarship holder']     = 0
test_dropout['Debtor']                 = 1

p_do = best_pipe.predict(test_dropout)[0]
pr_do = best_pipe.predict_proba(test_dropout)[0]
label_do = 'Graduate' if p_do == 1 else 'Dropout'
print(f"Dropout case  -> {label_do} | P(Dropout)={pr_do[0]:.3f} P(Graduate)={pr_do[1]:.3f}")

# Graduate case
test_grad = pd.DataFrame([{f: 0.0 for f in feature_names}])
test_grad['Curricular units 1st sem (enrolled)'] = 7
test_grad['Curricular units 1st sem (approved)'] = 7
test_grad['Curricular units 1st sem (grade)']    = 15.0
test_grad['Curricular units 1st sem (evaluations)'] = 9
test_grad['Curricular units 2nd sem (enrolled)'] = 7
test_grad['Curricular units 2nd sem (approved)'] = 7
test_grad['Curricular units 2nd sem (grade)']    = 14.5
test_grad['Curricular units 2nd sem (evaluations)'] = 9
test_grad['Age at enrollment']      = 18
test_grad['Admission grade']        = 155.0
test_grad['Tuition fees up to date'] = 1
test_grad['Scholarship holder']     = 1
test_grad['Debtor']                 = 0

p_gr = best_pipe.predict(test_grad)[0]
pr_gr = best_pipe.predict_proba(test_grad)[0]
label_gr = 'Graduate' if p_gr == 1 else 'Dropout'
print(f"Graduate case -> {label_gr} | P(Dropout)={pr_gr[0]:.3f} P(Graduate)={pr_gr[1]:.3f}")

# 8. Save bundle with medians
os.makedirs('models', exist_ok=True)
feature_medians = df[feature_names].median().to_dict()
bundle = {
    'pipeline': best_pipe,
    'feature_names': feature_names,
    'feature_medians': feature_medians,
    'model_name': 'SVM',
    'classes': [0, 1],
}
joblib.dump(bundle, 'models/best_student_graduation_model.joblib')
print("\nModel bundle saved: models/best_student_graduation_model.joblib")

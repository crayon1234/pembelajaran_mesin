"""
predict.py - Inference script for single-sample and batch prediction
UAS Pembelajaran Mesin - Klasifikasi Kelulusan Mahasiswa

Usage (from project root):
    # Single prediction (interactive):
    python src/predict.py

    # Batch prediction:
    python src/predict.py --batch data/data.csv --output reports/batch_predictions.csv
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ml_core import load_model, load_data

MODEL_PATH = "models/best_student_graduation_model.joblib"


def predict_single_from_dict(bundle, input_dict):
    """
    Predict from a dict of feature_name -> value.
    Returns label and probability dict.
    """
    model = bundle['model']
    scaler = bundle['scaler']
    feature_names = bundle['feature_names']

    values = [float(input_dict.get(f, 0)) for f in feature_names]
    x = np.array(values).reshape(1, -1)
    x_scaled = scaler.transform(x)

    pred = model.predict(x_scaled)[0]
    prob = model.predict_proba(x_scaled)[0]

    label = 'Graduate' if pred == 1 else 'Dropout'
    return label, {'Dropout': round(float(prob[0]), 4), 'Graduate': round(float(prob[1]), 4)}


def predict_batch(bundle, df):
    """
    Batch predict for a DataFrame.
    Returns DataFrame with prediction columns appended.
    """
    model = bundle['model']
    scaler = bundle['scaler']
    feature_names = bundle['feature_names']

    # Keep only relevant features
    X = df[feature_names] if all(f in df.columns for f in feature_names) else df.iloc[:, :len(feature_names)]
    X_scaled = scaler.transform(X.values)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    df = df.copy()
    df['Predicted_Label'] = ['Graduate' if p == 1 else 'Dropout' for p in preds]
    df['Prob_Dropout'] = probs[:, 0].round(4)
    df['Prob_Graduate'] = probs[:, 1].round(4)
    return df


def interactive_predict(bundle):
    """Simple CLI interactive prediction."""
    feature_names = bundle['feature_names']
    print("\n=== Student Graduation Predictor ===")
    print("Enter feature values (press Enter for default=0):\n")

    input_dict = {}
    key_features = [
        'Curricular units 2nd sem (approved)',
        'Curricular units 1st sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 1st sem (grade)',
        'Tuition fees up to date',
        'Age at enrollment',
        'Admission grade',
        'Scholarship holder',
        'Debtor',
        'Gender'
    ]

    for feat in key_features:
        if feat in feature_names:
            val = input(f"  {feat}: ").strip()
            input_dict[feat] = float(val) if val else 0.0

    label, probs = predict_single_from_dict(bundle, input_dict)
    print(f"\n  Prediction : {label}")
    print(f"  Prob Dropout  : {probs['Dropout']:.2%}")
    print(f"  Prob Graduate : {probs['Graduate']:.2%}")
    return label, probs


def main():
    parser = argparse.ArgumentParser(description='Predict student graduation')
    parser.add_argument('--batch', type=str, default=None,
                        help='Path to CSV file for batch prediction')
    parser.add_argument('--output', type=str, default='reports/batch_predictions.csv',
                        help='Output path for batch predictions')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='Path to model bundle (.joblib)')
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}. Run train.py first.")
        sys.exit(1)

    bundle = load_model(args.model)
    print(f"Model loaded: {bundle.get('model_name', 'Unknown')}")

    if args.batch:
        print(f"Running batch prediction on {args.batch}...")
        df = load_data(args.batch)
        # Remove target column if present
        if 'Target' in df.columns:
            df = df.drop(columns=['Target'])
        result_df = predict_batch(bundle, df)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        result_df.to_csv(args.output, index=False)
        print(f"Batch predictions saved to {args.output}")
        print(result_df['Predicted_Label'].value_counts())
    else:
        interactive_predict(bundle)


if __name__ == '__main__':
    main()

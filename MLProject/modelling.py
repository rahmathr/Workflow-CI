"""
modelling.py - untuk MLflow Project (Kriteria 3)
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
import mlflow
import mlflow.sklearn
import dagshub

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators",      type=int, default=100)
parser.add_argument("--max_depth",         type=str, default="None")
parser.add_argument("--min_samples_split", type=int, default=5)
parser.add_argument("--min_samples_leaf",  type=int, default=2)
args = parser.parse_args()

max_depth = None if args.max_depth == "None" else int(args.max_depth)

dagshub.init(repo_owner='rahmathr', repo_name='Eksperimen_SML_Rahmat', mlflow=True)

print("Memuat dataset...")
train_df = pd.read_csv('heart_train.csv')
test_df  = pd.read_csv('heart_test.csv')

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test  = test_df.drop('target', axis=1)
y_test  = test_df['target']

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

def plot_confusion_matrix(y_true, y_pred, path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Sakit', 'Sakit'],
                yticklabels=['Tidak Sakit', 'Sakit'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('Aktual'); plt.xlabel('Prediksi')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    return path

def plot_feature_importance(model, feature_names, path='feature_importance.png', top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 5))
    plt.bar(range(top_n), importances[indices], color='#5C6BC0', edgecolor='white')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right', fontsize=8)
    plt.title(f'Top {top_n} Feature Importance', fontweight='bold')
    plt.ylabel('Importance Score')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    return path

def plot_roc_curve(y_true, y_proba, path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='#5C6BC0', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    return path

with mlflow.start_run(run_name="RandomForest_CI") as run:
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAccuracy : {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         str(max_depth))
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("min_samples_leaf",  args.min_samples_leaf)
    mlflow.log_param("random_state",      42)

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   roc_auc)

    mlflow.sklearn.log_model(model, "random_forest_ci")

    mlflow.log_artifact(plot_confusion_matrix(y_test, y_pred),                "plots")
    mlflow.log_artifact(plot_feature_importance(model, list(X_train.columns)),"plots")
    mlflow.log_artifact(plot_roc_curve(y_test, y_pred_proba),                 "plots")

    report_path = "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact(report_path, "reports")

    run_id = run.info.run_id
    print(f"\nRun ID: {run_id}")

    # Simpan run_id ke RUN_ID_PATH (env var dari workflow) atau fallback ke workspace
    run_id_path = os.environ.get('RUN_ID_PATH', os.path.join(os.environ.get('GITHUB_WORKSPACE', '.'), 'run_id.txt'))
    with open(run_id_path, 'w') as f:
        f.write(run_id)
    print(f"Run ID disimpan ke: {run_id_path}")

print("\nSelesai!")

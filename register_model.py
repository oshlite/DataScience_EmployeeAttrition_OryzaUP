"""
Register Model to MLflow Model Registry
"""

import mlflow
from mlflow.client import MlflowClient

# Initialize MLflow
mlflow.set_experiment("Employee Attrition")
client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name("Employee Attrition")
experiment_id = experiment.experiment_id

print(f"Experiment ID: {experiment_id}")
print(f"Experiment Name: {experiment.name}")

# Get all runs dari experiment
runs = client.search_runs(experiment_ids=[experiment_id])

print(f"\nTotal runs: {len(runs)}")
print("\n" + "="*70)

# Filter dan tampilkan Logistic Regression runs
for i, run in enumerate(runs):
    run_id = run.info.run_id
    run_name = run.info.run_name

    # Get metrics dari run
    metrics = run.data.metrics

    print(f"\nRun {i+1}:")
    print(f"  ID: {run_id}")
    print(f"  Name: {run_name}")

    if 'roc_auc' in metrics:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    if 'test_accuracy' in metrics:
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")

print("\n" + "="*70)

# Cari Logistic Regression run (punya test_accuracy ~0.74 dan roc_auc ~0.81)
logistic_run = None
for run in runs:
    if run.info.run_name == "LogisticRegression":
        metrics = run.data.metrics
        if 'roc_auc' in metrics and abs(metrics['roc_auc'] - 0.81) < 0.05:
            logistic_run = run
            break

if logistic_run:
    run_id = logistic_run.info.run_id
    print(f"\nFound Logistic Regression run: {run_id}")
    print(f"   ROC-AUC: {logistic_run.data.metrics['roc_auc']:.4f}")

    try:
        # Register model
        model_uri = f"runs:/{run_id}/model"
        print(f"\nRegistering model from: {model_uri}")

        result = mlflow.register_model(
            model_uri=model_uri,
            name="Employee_Attrition_Model"
        )

        print(f"\nModel registered successfully!")
        print(f"   Model Name: {result.name}")
        print(f"   Version: {result.version}")
        print(f"   Run ID: {result.run_id}")

    except Exception as e:
        print(f"\nError: {e}")
        print(f"   (Model mungkin sudah terdaftar sebelumnya)")
else:
    print("\nLogistic Regression run not found!")

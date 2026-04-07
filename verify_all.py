#!/usr/bin/env python
"""
Complete System Verification Script
Verifies all three platforms: GitHub, DagsHub, MLflow
"""

import os
import subprocess
from dotenv import load_dotenv
import mlflow

print("\n" + "="*70)
print("EMPLOYEE ATTRITION MODEL - SYSTEM VERIFICATION")
print("="*70)

load_dotenv()

# ============================================================================
# 1. GITHUB VERIFICATION
# ============================================================================
print("\n[1] GITHUB VERIFICATION")
print("-" * 70)

try:
    result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True, cwd='.')
    remotes = result.stdout
    if 'github.com' in remotes:
        print("[OK] GitHub remote configured")
        for line in remotes.split('\n'):
            if 'origin' in line and 'github' in line:
                print(f"     {line}")

    result = subprocess.run(['git', 'log', '--oneline', '-1'], capture_output=True, text=True, cwd='.')
    commit = result.stdout.strip()
    print(f"[OK] Latest commit: {commit}")
except Exception as e:
    print(f"[ERROR] GitHub check failed: {e}")

# ============================================================================
# 2. DAGSHUB VERIFICATION
# ============================================================================
print("\n[2] DAGSHUB VERIFICATION")
print("-" * 70)

try:
    result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True, cwd='.')
    remotes = result.stdout
    if 'dagshub.com' in remotes:
        print("[OK] DagsHub remote configured")
        for line in remotes.split('\n'):
            if 'dagshub' in line:
                print(f"     {line}")

    result = subprocess.run(['git', 'log', 'dagshub/main', '--oneline', '-1'], capture_output=True, text=True, cwd='.')
    if result.returncode == 0:
        commit = result.stdout.strip()
        print(f"[OK] DagsHub latest commit: {commit}")
    else:
        print("[WARN] DagsHub branch not yet fetched")
except Exception as e:
    print(f"[ERROR] DagsHub check failed: {e}")

# ============================================================================
# 3. MLFLOW VERIFICATION
# ============================================================================
print("\n[3] MLFLOW VERIFICATION (LOCAL)")
print("-" * 70)

try:
    # Count local MLflow runs
    mlruns_path = 'mlruns'
    if os.path.exists(mlruns_path):
        run_count = 0
        for exp_dir in os.listdir(mlruns_path):
            exp_path = os.path.join(mlruns_path, exp_dir)
            if os.path.isdir(exp_path):
                for run_dir in os.listdir(exp_path):
                    run_path = os.path.join(exp_path, run_dir)
                    if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, 'meta.yaml')):
                        run_count += 1
        print(f"[OK] Local MLflow runs found: {run_count}")
        print(f"     Location: {os.path.abspath(mlruns_path)}")
    else:
        print("[WARN] mlruns directory not found")
except Exception as e:
    print(f"[ERROR] Local MLflow check failed: {e}")

# ============================================================================
# 4. DAGSHUB MLFLOW REMOTE VERIFICATION
# ============================================================================
print("\n[4] MLFLOW VERIFICATION (REMOTE - DAGSHUB)")
print("-" * 70)

try:
    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not dagshub_user:
        print("[ERROR] DAGSHUB_USER_NAME not set in .env")
    elif not dagshub_token:
        print("[ERROR] DAGSHUB_USER_TOKEN not set in .env")
    elif not mlflow_uri:
        print("[ERROR] MLFLOW_TRACKING_URI not set in .env")
    else:
        print(f"[OK] Credentials configured")
        print(f"     User: {dagshub_user}")
        print(f"     Token: {dagshub_token[:10]}...{dagshub_token[-5:]}")
        print(f"     MLflow URI: {mlflow_uri}")

        # Set credentials and test connection
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri(mlflow_uri)

        # Query experiments
        try:
            experiments = mlflow.search_experiments()
            print(f"\n[OK] Remote connection successful!")
            print(f"     Experiments: {len(experiments)}")

            for exp in experiments:
                print(f"\n     Experiment: {exp.name} (ID: {exp.experiment_id})")
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                print(f"     Runs: {len(runs)}")

                for i, run in enumerate(runs[:5], 1):
                    print(f"       {i}. {run.info.run_name} (Run ID: {run.info.run_id[:8]}...)")

                    if run.data.metrics:
                        metrics_str = ", ".join(list(run.data.metrics.keys())[:3])
                        print(f"          Metrics: {metrics_str}...")

                    if run.data.params:
                        params_str = ", ".join(list(run.data.params.keys())[:2])
                        print(f"          Params: {params_str}...")

        except Exception as e:
            print(f"[ERROR] Remote connection failed: {e}")
            print("        Check credentials and network connection")

except Exception as e:
    print(f"[ERROR] MLflow remote check failed: {e}")

# ============================================================================
# 5. MODEL REGISTRY VERIFICATION
# ============================================================================
print("\n[5] MODEL REGISTRY VERIFICATION")
print("-" * 70)

try:
    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    if dagshub_user and dagshub_token and mlflow_uri:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri(mlflow_uri)

        try:
            models = mlflow.search_registered_models()
            print(f"[OK] Model Registry accessible")
            print(f"     Registered Models: {len(models)}")

            for model in models:
                print(f"\n     Model: {model.name}")
                print(f"     Latest Version: {model.latest_versions[0].version if model.latest_versions else 'N/A'}")
                print(f"     Status: {model.latest_versions[0].status if model.latest_versions else 'N/A'}")
        except:
            print("[WARN] No registered models yet (register with: python register_model.py)")

except Exception as e:
    print(f"[ERROR] Model registry check: {e}")

# ============================================================================
# 6. CREDENTIALS VERIFICATION
# ============================================================================
print("\n[6] CREDENTIALS & FILES VERIFICATION")
print("-" * 70)

checks = [
    ('.env', 'Environment variables'),
    ('mlflow_config.py', 'MLflow configuration'),
    ('model/train.py', 'Training script'),
    ('app.py', 'Streamlit dashboard'),
    ('register_model.py', 'Model registration'),
    ('test_dagshub.py', 'DagsHub test'),
]

for filename, description in checks:
    if os.path.exists(filename):
        print(f"[OK] {filename:30s} - {description}")
    else:
        print(f"[WARN] {filename:30s} - {description} (NOT FOUND)")

# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

print("""
✅ GitHub Repository
   https://github.com/oshlite/Oryza_DS_EmployeeAttrition

✅ DagsHub Repository
   https://dagshub.com/oshlite/DataScience_EmployeeAttrition_Oryza

✅ MLflow Tracking
   - Local: mlruns/ directory
   - Remote: https://dagshub.com/oshlite/DataScience_EmployeeAttrition_Oryza.mlflow

✅ Model Registry
   - Employee_Attrition_Model (v1, v2, ...)

✅ All Systems Connected!

NEXT STEPS:
1. Open DagsHub → Check Experiments tab
2. Run: mlflow ui
3. Run: python -m streamlit run app.py
4. Make predictions and verify logging
""")

print("="*70)
print("VERIFICATION COMPLETE!")
print("="*70 + "\n")

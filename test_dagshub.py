"""
Test DagsHub Connection
"""

import os
from dotenv import load_dotenv
import mlflow
import requests

print("Testing DagsHub Integration...")
print("="*60)

# Load environment variables
load_dotenv()

dagshub_user = os.getenv("DAGSHUB_USER_NAME")
dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

# Check if credentials exist
print("\nChecking credentials...")
if not dagshub_user:
    print("ERROR: DAGSHUB_USER_NAME not found in .env")
    print("       Run: cp .env.example .env")
    print("       Edit .env with your DagsHub credentials")
    exit(1)

if not dagshub_token:
    print("ERROR: DAGSHUB_USER_TOKEN not found in .env")
    exit(1)

if not mlflow_uri:
    print("ERROR: MLFLOW_TRACKING_URI not found in .env")
    exit(1)

print(f"User: {dagshub_user}")
print(f"Token: {dagshub_token[:20]}...")
print(f"MLflow URI: {mlflow_uri}")

# Configure DagsHub credentials for MLflow
print("\nConfiguring MLflow for DagsHub...")
try:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    print("[OK] Credentials set")
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)

# Set MLflow tracking URI
print("\nSetting MLflow tracking URI...")
try:
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"[OK] MLflow URI set to: {mlflow_uri}")
except Exception as e:
    print(f"ERROR setting MLflow URI: {e}")
    exit(1)

# Test connection by listing experiments
print("\nTesting connection to DagsHub...")
try:
    mlflow.set_experiment("Employee Attrition")
    experiments = mlflow.search_experiments()
    print(f"[OK] Connected! Found {len(experiments)} experiment(s)")

    for exp in experiments:
        print(f"  - {exp.name}")

except Exception as e:
    print(f"ERROR: {e}")
    print("\nPossible issues:")
    print("1. Token expired or invalid")
    print("2. Repository doesn't exist on DagsHub")
    print("3. Network connection issue")
    exit(1)

print("\n" + "="*60)
print("SUCCESS! DagsHub is properly configured!")
print("\nYou can now:")
print("1. Run: python model/train.py")
print("2. Check DagsHub: https://dagshub.com/{}/DataScience_EmployeeAttrition_Oryza".format(dagshub_user))
print("3. View MLflow UI: mlflow ui")
print("\nAll metrics will auto-sync to DagsHub!")

@echo off
REM Set DagsHub credentials
set DAGSHUB_USER_NAME=oshlite
set DAGSHUB_USER_TOKEN=9655485333449a6af53ed0cc6866996a11d85be5
set MLFLOW_TRACKING_URI=https://dagshub.com/oshlite/DataScience_EmployeeAttrition_Oryza.mlflow
set MLFLOW_TRACKING_USERNAME=oshlite
set MLFLOW_TRACKING_PASSWORD=9655485333449a6af53ed0cc6866996a11d85be5

echo [OK] DagsHub credentials set
echo [OK] Starting MLflow UI with DagsHub...
echo.
echo Open: http://127.0.0.1:5000
echo.

REM Start MLflow UI
python -m mlflow ui

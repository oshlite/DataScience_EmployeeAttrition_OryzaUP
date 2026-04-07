import mlflow
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_mlflow():
    """Setup MLflow with DagsHub tracking"""

    dagshub_user = os.getenv("DAGSHUB_USER_NAME")
    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    if dagshub_user and dagshub_token and mlflow_uri:
        # Configure DagsHub credentials for MLflow authentication
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Set MLflow tracking URI to DagsHub
        mlflow.set_tracking_uri(mlflow_uri)

        # Set experiment name
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Employee Attrition")
        mlflow.set_experiment(experiment_name)

        return True

    return False

def log_dashboard_metrics(metrics_dict):
    """Log dashboard metrics to MLflow"""
    if not mlflow.active_run():
        mlflow.start_run()

    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)

    mlflow.end_run()

def log_model_prediction(input_data, prediction, probability):
    """Log model prediction to MLflow"""
    if not mlflow.active_run():
        mlflow.start_run()

    # Log input parameters
    for key, value in input_data.items():
        if isinstance(value, (int, float)):
            mlflow.log_param(f"input_{key}", value)

    # Log prediction metrics
    mlflow.log_metric("prediction_result", prediction)
    mlflow.log_metric("prediction_probability", probability)


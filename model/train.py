"""
Model Training Script with MLflow Integration
Employee Attrition Prediction
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup MLflow with DagsHub (from .env)
print("[*] Setting up MLflow tracking...")
dagshub_user = os.getenv("DAGSHUB_USER_NAME")
dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

if dagshub_user and dagshub_token and mlflow_uri:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"[OK] MLflow configured for remote tracking (DagsHub)")
else:
    print(f"[OK] Using local MLflow tracking")

# Setup MLflow experiment
mlflow.set_experiment("Employee Attrition")


class AttritionModelTrainer:
    """Main training pipeline with MLflow tracking"""

    def __init__(self, data_path='attrition_final.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.models = {}

    def load_and_prepare_data(self):
        """Load and prepare data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)

        # Separate features and target
        X = self.df.drop('Attrition', axis=1)
        y = self.df['Attrition']

        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")

    def preprocess_features(self):
        """Preprocess features (encoding and scaling)"""
        print("Preprocessing features...")

        # Encode categorical variables
        categorical_cols = self.X_train.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            le = LabelEncoder()
            self.X_train[col] = le.fit_transform(self.X_train[col].astype(str))
            self.X_test[col] = le.transform(self.X_test[col].astype(str))
            self.label_encoders[col] = le

        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("Features preprocessed [OK]")

    def handle_imbalance(self):
        """Handle class imbalance using SMOTE"""
        print("Handling class imbalance with SMOTE...")
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print(f"Training set after SMOTE: {len(self.X_train)}")

    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*60)
        print("Training: Logistic Regression")
        print("="*60)

        with mlflow.start_run(run_name="LogisticRegression", nested=True):
            # Model initialization
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs',
                class_weight='balanced'
            )

            # Log hyperparameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("solver", "lbfgs")
            mlflow.log_param("class_weight", "balanced")

            # Train
            model.fit(self.X_train, self.y_train)

            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # Metrics
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            precision = precision_score(self.y_test, y_pred_test)
            recall = recall_score(self.y_test, y_pred_test)
            f1 = f1_score(self.y_test, y_pred_test)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Save model
            model_path = "model/logistic_regression_best.pkl"
            joblib.dump(model, model_path)

            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy:  {test_accuracy:.4f}")
            print(f"Precision:      {precision:.4f}")
            print(f"Recall:         {recall:.4f}")
            print(f"F1 Score:       {f1:.4f}")
            print(f"ROC-AUC:        {roc_auc:.4f}")

            self.models['Logistic Regression'] = model

    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("Training: Random Forest Classifier")
        print("="*60)

        with mlflow.start_run(run_name="RandomForest", nested=True):
            # GridSearch for hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            }

            model = RandomForestClassifier(random_state=42, n_jobs=-1)

            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )

            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_

            # Log best parameters
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(f"rf_{param}", value)

            mlflow.log_metric("best_cv_roc_auc", grid_search.best_score_)

            # Predictions
            y_pred_train = best_model.predict(self.X_train)
            y_pred_test = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]

            # Metrics
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            precision = precision_score(self.y_test, y_pred_test)
            recall = recall_score(self.y_test, y_pred_test)
            f1 = f1_score(self.y_test, y_pred_test)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log model
            mlflow.sklearn.log_model(best_model, "model")

            print(f"Best Parameters:  {grid_search.best_params_}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy:  {test_accuracy:.4f}")
            print(f"Precision:      {precision:.4f}")
            print(f"Recall:         {recall:.4f}")
            print(f"F1 Score:       {f1:.4f}")
            print(f"ROC-AUC:        {roc_auc:.4f}")

            self.models['Random Forest'] = best_model

    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n" + "="*60)
        print("Training: Gradient Boosting Classifier")
        print("="*60)

        with mlflow.start_run(run_name="GradientBoosting", nested=True):
            # Model initialization
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=0
            )

            # Log hyperparameters
            mlflow.log_param("model_type", "GradientBoosting")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 5)

            # Train
            model.fit(self.X_train, self.y_train)

            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]

            # Metrics
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            precision = precision_score(self.y_test, y_pred_test)
            recall = recall_score(self.y_test, y_pred_test)
            f1 = f1_score(self.y_test, y_pred_test)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy:  {test_accuracy:.4f}")
            print(f"Precision:      {precision:.4f}")
            print(f"Recall:         {recall:.4f}")
            print(f"F1 Score:       {f1:.4f}")
            print(f"ROC-AUC:        {roc_auc:.4f}")

            self.models['Gradient Boosting'] = model

    def save_preprocessing_artifacts(self):
        """Save scaler and label encoders"""
        joblib.dump(self.scaler, "model/scaler.pkl")
        joblib.dump(self.label_encoders, "model/label_encoders.pkl")
        print("Preprocessing artifacts saved [OK]")

    def run_full_pipeline(self):
        """Run complete training pipeline"""
        print("\n" + "="*60)
        print("EMPLOYEE ATTRITION MODEL TRAINING")
        print("="*60)

        # Data preparation (OUTSIDE of any run)
        self.load_and_prepare_data()
        self.preprocess_features()
        self.handle_imbalance()

        # Save preprocessing artifacts
        self.save_preprocessing_artifacts()

        # Train models (each in its own independent run)
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_gradient_boosting()

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print("\nAll models and artifacts saved!")
        print("View MLflow dashboard: mlflow ui")


if __name__ == "__main__":
    # Initialize trainer
    trainer = AttritionModelTrainer()

    # Run full pipeline
    try:
        trainer.run_full_pipeline()
    finally:
        # Ensure all runs are properly closed (fixes encoding issues)
        mlflow.end_run()
        print("\n[OK] All MLflow runs properly closed!")

"""
Fraud detection model training script.

Usage:
    python train_model.py [--n-samples N] [--model-dir DIR] [--run-name NAME]

Generates synthetic labeled transaction data, engineers features, trains an
XGBoost classifier, evaluates it, logs everything to MLflow, and saves the
pickled model to disk for the consumer service to load.
"""

import argparse
import os
import pickle
import random
import logging
from datetime import datetime, timedelta

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from feature_engineering import extract_features_from_df, FEATURE_COLUMNS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="/app/.env")

HIGH_RISK_MERCHANTS = ["QuickCash", "PayDay", "LoanShark", "ShadyMerchants"]
HIGH_RISK_COUNTRIES = [
    "Nigeria",
    "Somalia",
    "Yemen",
    "Venezuela",
    "Afghanistan",
    "Syria",
]
TRANSACTION_TYPES = [
    "purchase",
    "cash_withdrawal",
    "card_testing",
    "chargeback",
    "refund",
]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]
LOCATIONS = ["US", "GB", "DE", "FR", "JP", "CA", "AU", "CN", "RU", "BR"]


def _random_date(days_back: int = 365) -> str:
    base = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
    return base.isoformat()


def generate_synthetic_data(n_samples: int = 100_000) -> pd.DataFrame:
    """Generate a synthetic labeled transaction dataset."""
    logger.info("Generating %d synthetic transactions...", n_samples)
    compromised_users = set(random.sample(range(1000, 9999), 50))
    records = []

    for _ in range(n_samples):
        user_id = random.randint(1000, 9999)
        amount = round(random.uniform(0.01, 9999.0), 2)
        merchant = random.choice(
            HIGH_RISK_MERCHANTS + [f"Merchant_{random.randint(1, 500)}"] * 8
        )
        merchant_country = random.choice(
            HIGH_RISK_COUNTRIES + [f"Country_{random.randint(1, 100)}"] * 8
        )
        txn_type = "purchase"
        is_fraud = 0

        # Account takeover
        if user_id in compromised_users and amount > 500 and random.random() < 0.3:
            is_fraud = 1
            amount = round(random.uniform(500, 5000), 2)
            merchant = random.choice(HIGH_RISK_MERCHANTS)
            txn_type = "cash_withdrawal"

        # Card testing
        if (
            not is_fraud
            and amount < 2.0
            and user_id % 1000 == 0
            and random.random() < 0.25
        ):
            is_fraud = 1
            amount = round(random.uniform(0.01, 2.0), 2)
            txn_type = "card_testing"

        # Merchant collusion
        if (
            not is_fraud
            and merchant in HIGH_RISK_MERCHANTS
            and amount > 3000
            and random.random() < 0.15
        ):
            is_fraud = 1
            amount = round(random.uniform(300, 1500), 2)
            txn_type = "refund"

        # Merchant dispute
        if not is_fraud and merchant in HIGH_RISK_MERCHANTS and random.random() < 0.1:
            is_fraud = 1
            amount = round(random.uniform(100, 500), 2)
            txn_type = "chargeback"

        # Geo anomaly
        if (
            not is_fraud
            and merchant_country in HIGH_RISK_COUNTRIES
            and user_id % 500 == 0
            and random.random() < 0.01
        ):
            is_fraud = 1
            amount = round(random.uniform(100, 1000), 2)

        # Baseline random fraud
        if not is_fraud and random.random() < 0.002:
            is_fraud = 1
            amount = round(random.uniform(100, 2000), 2)

        # Clamp final fraud rate
        is_fraud = is_fraud if random.random() < 0.985 else 0

        records.append(
            {
                "transaction_id": f"txn_{random.randint(10_000_000, 99_999_999)}",
                "event_time": _random_date(),
                "user_id": user_id,
                "amount": amount,
                "currency": random.choice(CURRENCIES),
                "merchant": merchant,
                "merchant_country": merchant_country,
                "timestamp": _random_date(),
                "location": random.choice(LOCATIONS),
                "transaction_type": txn_type,
                "card_number": f"**** **** **** {random.randint(1000, 9999)}",
                "is_fraud": is_fraud,
            }
        )

    df = pd.DataFrame(records)
    fraud_rate = df["is_fraud"].mean()
    logger.info("Dataset generated. Fraud rate: %.2f%%", fraud_rate * 100)
    return df


def train(
    n_samples: int = 100_000,
    model_dir: str = "/app/models",
    run_name: str = "fraud-model-training",
):
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5500")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    df = generate_synthetic_data(n_samples)

    logger.info("Engineering features...")
    X = extract_features_from_df(df)
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Applying SMOTE to balance training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "Post-SMOTE train shape: %s | fraud samples: %d",
        X_train_res.shape,
        int(y_train_res.sum()),
    )

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("features", FEATURE_COLUMNS)

        logger.info("Training XGBoost classifier...")
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_res,
            y_train_res,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        logger.info("--- Evaluation Results ---")
        logger.info("Precision : %.4f", precision)
        logger.info("Recall    : %.4f", recall)
        logger.info("F1 Score  : %.4f", f1)
        logger.info("ROC AUC   : %.4f", auc)
        logger.info(
            "\n%s",
            classification_report(y_test, y_pred, target_names=["legit", "fraud"]),
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="fraud_model",
            registered_model_name="fraud-detection-model",
        )

        # Save pickled model for the consumer service
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "fraud_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", model_path)

        mlflow.log_artifact(model_path, artifact_path="model_artifact")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--model-dir", type=str, default="/app/models")
    parser.add_argument("--run-name", type=str, default="fraud-model-training")
    args = parser.parse_args()

    train(
        n_samples=args.n_samples,
        model_dir=args.model_dir,
        run_name=args.run_name,
    )

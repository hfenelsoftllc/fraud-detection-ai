"""
Fraud Detection Model Training DAG

Runs daily at 02:00 UTC.
1. generate_training_data  – builds a synthetic labeled dataset
2. feature_engineering     – validates feature extraction
3. train_and_evaluate      – trains XGBoost, logs metrics to MLflow
4. promote_model           – transitions the new model to Production in MLflow registry
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

default_args = {
    "owner": "fraud-detection",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

MODEL_DIR = "/app/models"
N_SAMPLES = int(os.getenv("TRAINING_N_SAMPLES", "100000"))
MIN_F1_THRESHOLD = float(os.getenv("MIN_F1_THRESHOLD", "0.70"))


# ── Task callables ─────────────────────────────────────────────────────────────


def generate_training_data(**context):
    sys.path.insert(0, "/app")
    from models.train_model import generate_synthetic_data
    import tempfile
    import pickle

    df = generate_synthetic_data(N_SAMPLES)
    tmp_path = os.path.join(MODEL_DIR, "training_data.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(tmp_path, "wb") as f:
        pickle.dump(df, f)
    context["ti"].xcom_push(key="data_path", value=tmp_path)
    return tmp_path


def feature_engineering(**context):
    import pickle

    sys.path.insert(0, "/app")
    from models.feature_engineering import extract_features_from_df

    data_path = context["ti"].xcom_pull(
        key="data_path", task_ids="generate_training_data"
    )
    with open(data_path, "rb") as f:
        df = pickle.load(f)

    X = extract_features_from_df(df)
    assert X.shape[0] == len(df), "Feature row count mismatch"
    assert X.isnull().sum().sum() == 0, "Features contain null values"
    context["ti"].xcom_push(key="feature_shape", value=str(X.shape))
    return X.shape


def train_and_evaluate(**context):
    sys.path.insert(0, "/app")
    from models.train_model import train

    run_name = f"fraud-training-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    train(
        n_samples=N_SAMPLES,
        model_dir=MODEL_DIR,
        run_name=run_name,
    )
    # Read back the saved model and evaluate on a small holdout to push metric to XCom
    import pickle
    from sklearn.metrics import f1_score
    import random

    model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    from models.train_model import generate_synthetic_data
    from models.feature_engineering import extract_features_from_df

    holdout = generate_synthetic_data(5000)
    X_h = extract_features_from_df(holdout)
    y_h = holdout["is_fraud"].values
    y_pred = model.predict(X_h)
    f1 = f1_score(y_h, y_pred, zero_division=0)
    context["ti"].xcom_push(key="holdout_f1", value=f1)
    return f1


def check_model_quality(**context):
    f1 = context["ti"].xcom_pull(key="holdout_f1", task_ids="train_and_evaluate")
    if f1 is None or f1 < MIN_F1_THRESHOLD:
        raise ValueError(
            f"Model F1 ({f1:.4f}) below minimum threshold ({MIN_F1_THRESHOLD}). "
            "Skipping promotion."
        )
    return True


def promote_model(**context):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5500")
    )
    client = MlflowClient()
    model_name = "fraud-detection-model"

    # Find the latest version
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")

    latest = sorted(versions, key=lambda v: int(v.version))[-1]

    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Model '{model_name}' version {latest.version} promoted to Production")


# ── DAG ────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="fraud_model_training",
    default_args=default_args,
    description="Daily fraud detection model training and promotion",
    schedule_interval="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["fraud-detection", "ml", "training"],
) as dag:

    t1 = PythonOperator(
        task_id="generate_training_data",
        python_callable=generate_training_data,
    )

    t2 = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    t3 = PythonOperator(
        task_id="train_and_evaluate",
        python_callable=train_and_evaluate,
    )

    t4 = ShortCircuitOperator(
        task_id="check_model_quality",
        python_callable=check_model_quality,
    )

    t5 = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    t1 >> t2 >> t3 >> t4 >> t5

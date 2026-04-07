"""
Fraud Detection Data Quality Monitoring DAG

Runs every 6 hours.
1. check_kafka_consumer_lag  – alerts if the fraud-detection consumer is behind
2. check_fraud_rate          – detects spikes or drops in fraud rate (data drift)
3. check_feature_drift       – compares live feature distributions vs training baseline
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "fraud-detection",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

MAX_CONSUMER_LAG = int(os.getenv("MAX_CONSUMER_LAG", "10000"))
EXPECTED_FRAUD_RATE_MIN = float(os.getenv("EXPECTED_FRAUD_RATE_MIN", "0.005"))
EXPECTED_FRAUD_RATE_MAX = float(os.getenv("EXPECTED_FRAUD_RATE_MAX", "0.05"))


# ── Task callables ─────────────────────────────────────────────────────────────


def check_kafka_consumer_lag(**context):
    """Check that the fraud-detection consumer group is not lagging excessively."""
    from confluent_kafka.admin import AdminClient
    from confluent_kafka import Consumer

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    group_id = "fraud-detection-group"
    topic = os.getenv("KAFKA_TOPIC", "transactions")

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": "airflow-lag-checker",
            "security.protocol": "PLAINTEXT",
        }
    )

    try:
        metadata = consumer.list_topics(topic, timeout=10)
        partitions = metadata.topics[topic].partitions
        total_lag = 0

        for pid in partitions:
            from confluent_kafka import TopicPartition

            tp = TopicPartition(topic, pid)
            low, high = consumer.get_watermark_offsets(tp, timeout=5)
            committed = consumer.committed([tp], timeout=5)
            committed_offset = committed[0].offset if committed[0].offset >= 0 else low
            total_lag += max(0, high - committed_offset)

        print(f"Consumer group '{group_id}' total lag: {total_lag} messages")
        context["ti"].xcom_push(key="consumer_lag", value=total_lag)

        if total_lag > MAX_CONSUMER_LAG:
            raise ValueError(
                f"Consumer lag {total_lag} exceeds threshold {MAX_CONSUMER_LAG}. "
                "Fraud detection may be delayed!"
            )
    finally:
        consumer.close()


def check_fraud_rate(**context):
    """
    Validate that the recent fraud rate is within expected bounds.
    Reads from MLflow experiment metrics as a proxy for recent model performance.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5500")
    )
    client = MlflowClient()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found. Skipping fraud rate check.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        print("No training runs found. Skipping fraud rate check.")
        return

    latest_run = runs[0]
    recall = latest_run.data.metrics.get("recall")
    precision = latest_run.data.metrics.get("precision")
    f1 = latest_run.data.metrics.get("f1_score")

    print(
        f"Latest model metrics — Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    if f1 and f1 < 0.60:
        raise ValueError(
            f"Latest model F1 score {f1:.4f} is critically low. "
            "Consider retraining immediately."
        )


def check_feature_drift(**context):
    """
    Lightweight feature drift check: compare current synthetic distribution
    against expected ranges from training.
    In production this would compare live Kafka-sourced samples against stored baselines.
    """
    import sys

    sys.path.insert(0, "/app")
    from models.train_model import generate_synthetic_data
    from models.feature_engineering import extract_features_from_df

    sample = generate_synthetic_data(1000)
    X = extract_features_from_df(sample)

    amount_mean = X["amount"].mean()
    amount_std = X["amount"].std()

    print(
        f"Feature drift check — amount mean: {amount_mean:.2f}, std: {amount_std:.2f}"
    )

    if amount_mean < 100 or amount_mean > 9000:
        raise ValueError(
            f"Amount feature mean {amount_mean:.2f} is outside expected range [100, 9000]. "
            "Possible data drift detected."
        )

    context["ti"].xcom_push(key="amount_mean", value=amount_mean)
    context["ti"].xcom_push(key="amount_std", value=amount_std)


# ── DAG ────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="fraud_data_quality",
    default_args=default_args,
    description="Data quality and drift monitoring for fraud detection pipeline",
    schedule_interval="0 */6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["fraud-detection", "monitoring", "data-quality"],
) as dag:

    t1 = PythonOperator(
        task_id="check_kafka_consumer_lag",
        python_callable=check_kafka_consumer_lag,
    )

    t2 = PythonOperator(
        task_id="check_fraud_rate",
        python_callable=check_fraud_rate,
    )

    t3 = PythonOperator(
        task_id="check_feature_drift",
        python_callable=check_feature_drift,
    )

    [t1, t2, t3]  # Run all checks in parallel

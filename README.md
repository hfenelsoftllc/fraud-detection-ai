# Fraud Detection System

A stream processing orchestration system designed to monitor financial transactions in real-time, enabling the immediate detection of fraudulent activities. The system provides rapid fraud identification, client notification, and facilitates proactive measures to block malicious actors — enhancing security and mitigating financial risk.

---

## Architecture

```
┌──────────────┐     ┌─────────────────────────────────────────────┐
│  Transaction │     │          Apache Kafka (KRaft mode)          │
│   Producer   │────▶│  topics: transactions  │  fraud-alerts       │
│  (synthetic) │     └──────────────┬──────────────────┬───────────┘
└──────────────┘                    │                  │
                                    ▼                  ▼
                         ┌─────────────────┐  ┌──────────────────┐
                         │ Fraud Detection │  │  Notification    │
                         │   Consumer      │  │  Service         │
                         │ (ML Inference)  │  │ (Email / Log)    │
                         └────────┬────────┘  └──────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │        MLflow           │
                     │  (Model Registry /      │
                     │   Experiment Tracking)  │
                     └────────────┬────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │       Apache Airflow    │
                     │  ┌──────────────────┐  │
                     │  │ Model Training   │  │ ← daily @ 02:00 UTC
                     │  │      DAG         │  │
                     │  └──────────────────┘  │
                     │  ┌──────────────────┐  │
                     │  │  Data Quality    │  │ ← every 6 hours
                     │  │  Monitoring DAG  │  │
                     │  └──────────────────┘  │
                     └─────────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │  MinIO (S3-compatible)  │
                     │  MLflow artifact store  │
                     └─────────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │       PostgreSQL        │
                     │  Airflow metadata DB    │
                     │  MLflow backend store   │
                     └─────────────────────────┘
```

> **Kafka runs in [KRaft mode](https://kafka.apache.org/documentation/#kraft)** (Kafka Raft metadata mode) — the embedded Raft-based controller replaces ZooKeeper entirely, simplifying the deployment to a single broker + controller container. KRaft has been production-ready since Confluent Platform 7.4 / Apache Kafka 3.3.

---

## Components

| Service | Image / Build | Port | Purpose |
|---|---|---|---|
| **producer** | `./producer` | — | Generates synthetic transactions → Kafka |
| **kafka** | `confluentinc/cp-kafka:latest` | 9092 / 29092 | Message broker (KRaft mode — no ZooKeeper) |
| **kafka-ui** | `provectuslabs/kafka-ui` | 8090 | Kafka topic browser |
| **consumer** | `./consumer` | — | Reads transactions, runs ML inference, publishes fraud alerts |
| **notification** | `./notification` | — | Reads fraud alerts, emits logs, optionally sends email |
| **airflow-webserver** | `./airflow` | 8080 | DAG UI |
| **airflow-scheduler** | `./airflow` | — | DAG scheduler |
| **airflow-worker** | `./airflow` | — | Celery workers (×2) |
| **mlflow-server** | `./mlflow` | 5500 | Experiment tracking + model registry |
| **minio** | `minio/minio` | 9000 / 9001 | S3-compatible artifact store |
| **postgres** | `postgres:18` | 5432 | Airflow + MLflow metadata |
| **redis** | `redis:7.4-bookworm` | 6379 | Celery broker |

---

## Fraud Detection Patterns

The producer simulates the following real-world fraud scenarios:

| Pattern | Description |
|---|---|
| Account Takeover | Compromised user IDs executing large cash withdrawals |
| Card Testing | Micro-transactions (<$2) to validate stolen card numbers |
| Merchant Collusion | High-value refunds routed through known risky merchants |
| Merchant Dispute | Chargebacks from high-risk merchants |
| Geography Anomaly | Transactions from high-risk countries for flagged users |
| Baseline Random Fraud | ~0.2% random fraud floor |

**Target fraud rate:** 1–2% of all transactions.

---

## ML Pipeline

### Features (14 engineered)

| Feature | Description |
|---|---|
| `amount` | Raw transaction amount |
| `log_amount` | `log(1 + amount)` for skew reduction |
| `is_high_amount` | 1 if amount > $1,000 |
| `txn_type_encoded` | Ordinal encoding of transaction type |
| `is_cash_withdrawal` / `is_card_testing` / `is_chargeback` / `is_refund` | One-hot transaction type flags |
| `is_high_risk_merchant` | Merchant in known risky list |
| `is_high_risk_country` | Merchant country in high-risk list |
| `user_segment` | `user_id % 100` — proxy for user cohort |
| `hour_of_day` | Hour extracted from `event_time` |
| `day_of_week` | Day of week (0=Mon … 6=Sun) |
| `is_weekend` | 1 if Saturday or Sunday |

### Model

- **Algorithm:** XGBoost Classifier
- **Class imbalance:** SMOTE oversampling on training set
- **Metrics tracked:** Precision, Recall, F1, ROC-AUC (logged to MLflow)
- **Promotion gate:** F1 ≥ 0.70 required before model is promoted to `Production`

---

## Airflow DAGs

| DAG | Schedule | Description |
|---|---|---|
| `fraud_model_training` | `0 2 * * *` (daily) | Generate data → feature engineering → train XGBoost → quality gate → promote to MLflow registry |
| `fraud_data_quality` | `0 */6 * * *` (6-hourly) | Check Kafka consumer lag, model metric health, feature drift |

---

## Quick Start

### Prerequisites

- Docker Desktop ≥ 4.x with **4 GB RAM** and **2 CPUs** allocated
- Docker Compose v2

### 1. Configure environment

```bash
cd src
cp .env.example .env
# Edit .env — set passwords; leave KAFKA_USERNAME/PASSWORD blank for local dev
```

> **KRaft cluster ID:** `docker-compose.yml` ships with a fixed `CLUSTER_ID` for convenience. For a production deployment generate a fresh one: `docker run --rm confluentinc/cp-kafka:latest kafka-storage random-uuid` and replace the `CLUSTER_ID` value in `docker-compose.yml`.

### 2. Start all services

```bash
docker compose up --build -d
```

### 3. Verify services

| UI | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5500 |
| Kafka UI | http://localhost:8090 |
| MinIO | http://localhost:9001 |

### 4. Trigger initial model training

In Airflow UI → enable and trigger `fraud_model_training` DAG, or run directly:

```bash
docker compose exec airflow-worker python /app/models/train_model.py \
  --n-samples 100000 \
  --model-dir /app/models
```

### 5. Watch fraud alerts

```bash
docker compose logs -f notification
```

---

## Project Structure

```
src/
├── .env                    # Local environment (gitignored)
├── .env.example            # Environment template
├── config.yml              # Application configuration
├── docker-compose.yml      # Full service stack
├── init-multiple-dbs.sh    # Creates MLflow DB in PostgreSQL
├── wait-for-it.sh          # TCP readiness helper
│
├── producer/               # Synthetic transaction generator
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── consumer/               # Kafka consumer + ML inference
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── notification/           # Fraud alert handler (log + email)
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── models/                 # Model artefacts (runtime-generated)
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── fraud_model.pkl     # Written by training pipeline
│
├── dags/                   # Airflow DAGs
│   ├── fraud_model_training_dag.py
│   └── fraud_data_quality_dag.py
│
├── airflow/                # Custom Airflow image
│   ├── Dockerfile
│   └── requirements.txt
│
└── mlflow/                 # MLflow tracking server image
    ├── Dockerfile
    └── requirements.txt
```

---

## Configuration Reference

All runtime configuration lives in `src/config.yml` and is overridable via environment variables in `.env`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `model.threshold` | `0.5` | Fraud score threshold for binary classification |
| `model.min_f1_threshold` | `0.70` | Minimum F1 required for model promotion |
| `model.n_training_samples` | `100000` | Synthetic samples per training run |
| `monitoring.max_consumer_lag` | `10000` | Alert threshold for Kafka consumer lag |
| `kafka.topics.transactions` | `transactions` | Input topic |
| `kafka.topics.fraud_alerts` | `fraud-alerts` | Output alert topic |

---

## Security Notes

- **Credentials:** Never commit `.env`. Use `.env.example` as the template.
- **Kafka:** Set `KAFKA_USERNAME`/`KAFKA_PASSWORD` for SASL_SSL (e.g., Confluent Cloud). Leave blank for local PLAINTEXT.
- **Email:** SMTP credentials are optional. Structured structured fraud alerts are always written to the container log stream.
- **Card numbers:** Synthetic only — generated by Faker, never real PAN data.


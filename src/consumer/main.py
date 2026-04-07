import os
import json
import signal
import logging
import pickle
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="/app/.env")

HIGH_RISK_MERCHANTS = {"QuickCash", "PayDay", "LoanShark", "ShadyMerchants"}
HIGH_RISK_COUNTRIES = {
    "Nigeria",
    "Somalia",
    "Yemen",
    "Venezuela",
    "Afghanistan",
    "Syria",
}
TRANSACTION_TYPES = [
    "purchase",
    "cash_withdrawal",
    "card_testing",
    "chargeback",
    "refund",
]


class FraudDetectionConsumer:
    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.input_topic = os.getenv("KAFKA_TOPIC", "transactions")
        self.fraud_topic = os.getenv("KAFKA_FRAUD_TOPIC", "fraud-alerts")
        self.group_id = os.getenv("KAFKA_GROUP_ID", "fraud-detection-group")
        self.model_path = os.getenv("MODEL_PATH", "/app/models/fraud_model.pkl")
        self.running = False
        self.model = None

        kafka_username = os.getenv("KAFKA_USERNAME")
        kafka_password = os.getenv("KAFKA_PASSWORD")

        base_config = {"security.protocol": "PLAINTEXT"}
        if kafka_username and kafka_password:
            base_config = {
                "security.protocol": "SASL_SSL",
                "sasl.mechanisms": "PLAIN",
                "sasl.username": kafka_username,
                "sasl.password": kafka_password,
            }

        consumer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            **base_config,
        }

        producer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": "fraud-alert-producer",
            "acks": "all",
            **base_config,
        }

        self.consumer = Consumer(consumer_config)
        self.alert_producer = Producer(producer_config)
        self.consumer.subscribe([self.input_topic])

        self._load_model()

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Loaded fraud detection model from %s", self.model_path)
            except Exception as e:
                logger.warning(
                    "Could not load model from %s: %s. Using label-passthrough mode.",
                    self.model_path,
                    str(e),
                )
        else:
            logger.warning(
                "No model found at %s. Using is_fraud label passthrough until model is trained.",
                self.model_path,
            )

    def _extract_features(self, transaction: dict) -> Optional[np.ndarray]:
        try:
            txn_type = transaction.get("transaction_type", "purchase")
            features = [
                float(transaction.get("amount", 0.0)),
                1.0 if txn_type == "cash_withdrawal" else 0.0,
                1.0 if txn_type == "card_testing" else 0.0,
                1.0 if txn_type == "chargeback" else 0.0,
                1.0 if txn_type == "refund" else 0.0,
                1.0 if transaction.get("merchant", "") in HIGH_RISK_MERCHANTS else 0.0,
                (
                    1.0
                    if transaction.get("merchant_country", "") in HIGH_RISK_COUNTRIES
                    else 0.0
                ),
                float(transaction.get("user_id", 0)) % 100,
            ]
            return np.array(features, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            logger.error("Feature extraction error: %s", str(e))
            return None

    def _predict_fraud(self, transaction: dict) -> bool:
        if self.model is None:
            return bool(transaction.get("is_fraud", 0))

        features = self._extract_features(transaction)
        if features is None:
            return bool(transaction.get("is_fraud", 0))

        try:
            prediction = self.model.predict(features)[0]
            return bool(prediction)
        except Exception as e:
            logger.error("Prediction error: %s. Falling back to label.", str(e))
            return bool(transaction.get("is_fraud", 0))

    def _publish_alert(self, transaction: dict):
        alert = {
            "transaction_id": transaction.get("transaction_id"),
            "user_id": transaction.get("user_id"),
            "amount": transaction.get("amount"),
            "currency": transaction.get("currency"),
            "merchant": transaction.get("merchant"),
            "merchant_country": transaction.get("merchant_country"),
            "location": transaction.get("location"),
            "transaction_type": transaction.get("transaction_type"),
            "event_time": transaction.get("event_time"),
            "alert_reason": "ML_FRAUD_DETECTED",
        }
        self.alert_producer.produce(
            self.fraud_topic,
            key=str(transaction.get("transaction_id", "")),
            value=json.dumps(alert),
        )
        self.alert_producer.poll(0)

    def run(self):
        self.running = True
        logger.info(
            "Fraud detection consumer started. Listening on topic: %s", self.input_topic
        )
        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())

                try:
                    transaction = json.loads(msg.value().decode("utf-8"))
                    is_fraud = self._predict_fraud(transaction)

                    if is_fraud:
                        logger.warning(
                            "FRAUD DETECTED | txn_id=%s | user=%s | amount=%.2f | type=%s",
                            transaction.get("transaction_id"),
                            transaction.get("user_id"),
                            float(transaction.get("amount", 0.0)),
                            transaction.get("transaction_type"),
                        )
                        self._publish_alert(transaction)

                    self.consumer.commit(asynchronous=False)

                except json.JSONDecodeError as e:
                    logger.error("Failed to decode message: %s", str(e))
                except Exception as e:
                    logger.error("Error processing message: %s", str(e))

        except KafkaException as e:
            logger.error("Kafka exception: %s", str(e))
        finally:
            self.consumer.close()
            self.alert_producer.flush(timeout=20)
            logger.info("Consumer shut down cleanly")

    def shutdown(self, signum, frame):
        logger.info("Shutting down fraud detection consumer...")
        self.running = False


if __name__ == "__main__":
    consumer = FraudDetectionConsumer()
    consumer.run()

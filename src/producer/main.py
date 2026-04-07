import os
import json
import random
import signal
import time
from typing import Optional
from dotenv import load_dotenv
from confluent_kafka import Producer

import logging

import faker
from jsonschema import FormatChecker, ValidationError, validate

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="/app/.env")

fake = faker.Faker()

TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_id": {"type": "string"},
        "event_time": {"type": "string", "format": "date-time"},
        "user_id": {"type": "integer", "minimum": 1000, "maximum": 9999},
        "amount": {"type": "number", "minimum": 0.01, "maximum": 10000},
        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "merchant": {"type": "string"},
        "merchant_country": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "location": {"type": "string"},
        "transaction_type": {"type": "string"},
        "card_number": {"type": "string"},
        "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1},
    },
    "required": [
        "transaction_id",
        "event_time",
        "user_id",
        "amount",
        "currency",
        "merchant",
        "merchant_country",
        "timestamp",
        "location",
        "transaction_type",
        "card_number",
        "is_fraud",
    ],
}


class TransactionProducer:
    def __init__(self):
        self.kafka_username = os.getenv("KAFKA_USERNAME")
        self.kafka_password = os.getenv("KAFKA_PASSWORD")
        self.kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.topic = os.getenv("KAFKA_TOPIC", "transactions")
        self.running = False

        # Initialize fraud simulation parameters before producer setup
        self.compromised_users = set(random.sample(range(1000, 9999), 50))
        self.high_risk_merchants = [
            "QuickCash",
            "PayDay",
            "LoanShark",
            "ShadyMerchants",
        ]
        self.high_risk_countries = [
            "Nigeria",
            "Somalia",
            "Yemen",
            "Venezuela",
            "Afghanistan",
            "Syria",
        ]
        self.fraud_pattern_weights = {
            "compromised_user": 0.30,
            "high_risk_merchant": 0.20,
            "high_risk_country": 0.10,
            "card_testing": 0.30,
            "merchant_dispute": 0.10,
            "merchant_collusion": 0.20,
            "rare": 0.05,
            "geo_anomaly": 0.05,
        }

        # Build confluent kafka producer configuration
        producer_config = {
            "bootstrap.servers": self.kafka_bootstrap,
            "client.id": "transaction-producer",
            "compression.type": "gzip",
            "linger.ms": 5,
            "batch.size": 16384,
            "acks": "all",
        }

        if self.kafka_username and self.kafka_password:
            producer_config.update(
                {
                    "security.protocol": "SASL_SSL",
                    "sasl.mechanisms": "PLAIN",
                    "sasl.username": self.kafka_username,
                    "sasl.password": self.kafka_password,
                }
            )
        else:
            producer_config["security.protocol"] = "PLAINTEXT"

        try:
            self.producer = Producer(producer_config)
            logger.info("Confluent Kafka Producer initialized successfully")
            self.running = True
        except Exception as e:
            logger.error(f"Failed to initialize confluent kafka producer: {str(e)}")
            raise

        # Configure graceful shutdown
        signal.signal(signal.SIGINT, self.stop_production)
        signal.signal(signal.SIGTERM, self.stop_production)

    # Run continuous production to Kafka
    def run_continuous_production(self, interval: float = 0.0):
        self.running = True
        logger.info("Starting transaction producer for topic: %s", self.topic)
        try:
            while self.running:
                if self.send_transaction():
                    time.sleep(interval)
        finally:
            self.stop_production(None, None)

    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    # Validate transaction against schema
    def validate_transaction(self, transaction: dict) -> bool:
        try:
            validate(
                instance=transaction,
                schema=TRANSACTION_SCHEMA,
                format_checker=FormatChecker(),
            )
            return True
        except ValidationError as e:
            logger.error(f"Invalid transaction: {str(e)}")
            return False

    # Generate transaction and apply fraud simulation patterns
    def generate_transaction(self) -> Optional[dict]:
        transaction = {
            "transaction_id": fake.uuid4(),
            "event_time": fake.date_time_between(
                start_date="-1y", end_date="now"
            ).isoformat(),
            "user_id": fake.random_int(min=1000, max=9999),
            "amount": round(random.uniform(0.01, 9999.0), 2),
            "currency": fake.currency_code(),
            "merchant": fake.company(),
            "merchant_country": fake.country(),
            "timestamp": fake.date_time_between(
                start_date="-1y", end_date="now"
            ).isoformat(),
            "location": fake.country_code(representation="alpha-2"),
            "transaction_type": "purchase",
            "card_number": fake.credit_card_number(),
            "is_fraud": 0,
        }

        is_fraud = 0
        amount = transaction["amount"]
        user_id = transaction["user_id"]
        merchant = transaction["merchant"]
        merchant_country = transaction["merchant_country"]

        is_fraud = 0
        amount = transaction["amount"]
        user_id = transaction["user_id"]
        merchant = transaction["merchant"]
        merchant_country = transaction["merchant_country"]

        # Account Takeover
        if user_id in self.compromised_users and amount > 500:
            if random.random() < 0.3:
                is_fraud = 1
                transaction["is_fraud"] = 1
                transaction["amount"] = round(random.uniform(500, 5000), 2)
                transaction["merchant"] = random.choice(self.high_risk_merchants)
                transaction["transaction_type"] = "cash_withdrawal"

        # Card testing
        if not is_fraud and amount < 2.0:
            if user_id % 1000 == 0 and random.random() < 0.25:
                is_fraud = 1
                transaction["is_fraud"] = 1
                transaction["amount"] = round(random.uniform(0.01, 2), 2)
                transaction["location"] = "US"
                transaction["transaction_type"] = "card_testing"

        # Merchant collusion
        if not is_fraud and merchant in self.high_risk_merchants:
            if amount > 3000 and random.random() < 0.15:
                is_fraud = 1
                transaction["is_fraud"] = 1
                transaction["amount"] = round(random.uniform(300, 1500), 2)
                transaction["transaction_type"] = "refund"

        # Merchant dispute
        if not is_fraud and merchant in self.high_risk_merchants:
            if random.random() < 0.1:
                is_fraud = 1
                transaction["is_fraud"] = 1
                transaction["amount"] = round(random.uniform(100, 500), 2)
                transaction["transaction_type"] = "chargeback"

        # Geography anomaly
        if not is_fraud and merchant_country in self.high_risk_countries:
            if user_id % 500 == 0 and random.random() < 0.01:
                is_fraud = 1
                transaction["is_fraud"] = 1
                transaction["location"] = random.choice(
                    ["CN", "RU", "GB", "DE", "FR", "JP"]
                )
                transaction["amount"] = round(random.uniform(100, 1000), 2)
                transaction["transaction_type"] = "purchase"

        # Baseline random fraud (~0.2%)
        if not is_fraud and random.random() < 0.002:
            is_fraud = 1
            transaction["is_fraud"] = 1
            transaction["amount"] = round(random.uniform(100, 2000), 2)

        # Ensure final fraud rate stays between 1-2%
        transaction["is_fraud"] = is_fraud if random.random() < 0.985 else 0

        if self.validate_transaction(transaction):
            return transaction
        return None

    # Send a generated transaction to Kafka
    def send_transaction(self) -> bool:
        try:
            transaction = self.generate_transaction()
            if not transaction:
                return False

            self.producer.produce(
                self.topic,
                key=transaction["transaction_id"],
                value=json.dumps(transaction),
                callback=self.delivery_report,
            )
            self.producer.poll(0)  # trigger delivery callbacks
            return True
        except Exception as e:
            logger.error(f"Failed to send transaction: {str(e)}")
            return False

    # Graceful shutdown
    def stop_production(self, signum, frame):
        if self.running:
            logger.info("Stopping transaction producer")
            self.running = False
            if self.producer:
                logger.info("Flushing confluent kafka producer")
                self.producer.flush(timeout=20)
                logger.info("Producer stopped cleanly")


if __name__ == "__main__":
    producer = TransactionProducer()
    producer.run_continuous_production()

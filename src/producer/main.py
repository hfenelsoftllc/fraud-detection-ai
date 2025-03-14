import os
import json
import random
import time
from typing import Optional
from dotenv import load_dotenv
from confluent_kafka import Producer

import logging

import faker
from jsonschema import FormatChecker, ValidationError, validate

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",level=logging.INFO)

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path='/app/.env')

fake =faker.Faker()

TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_id": {"type": "string"},
        "event_time": {"type": "string", "format": "date-time"},
        "user_id": {"type": "integer", "minimum": 1000, "maximum": 9999},
        "amount": {"type": "number", "minimum":0.01, "maximum": 10000},
        "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
        "merchant": {"type": "string"},
        "merchant_country": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "location": {"type": "string", "pattern": "^[A-Z]{2}$"},
        "transaction_type": {"type": "string"},
        "card_number": {"type": "string"},
        "is_fraud": {"type": "integer", "minimum": 0, "maximum": 1}
    },
    "required": ["transaction_id", "event_time", "user_id", "amount", "currency", "merchant", "merchant_country", "timestamp", "transaction_type", "card_number", "is_fraud"] 
}

class TransactionProducer():
    def __init__(self):
        self.producer = TransactionProducer('KAFKA_BOOTSTRAP_SERVERS', bootstrap_servers='localhost:9092')
        self.kafka_username = os.getenv('KAFKA_USERNAME')
        self.kafka_password = os.getenv('KAFKA_PASSWORD')
        self.topic = os.getenv('KAFKA_TOPIC', 'transaction')
        self.running = False

        # confluent kafka configuration
        self.producer_config = {
            'bootstrap.servers': 'localhost:9092',
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': self.kafka_username,
            'sasl.password': self.kafka_password,
            'client.id': 'transaction-producer',
            'compression.type': 'gzip',
            'linger.ms': 5,
            'batch.size': 16384,
            'acks': 'all'
        }

        if self.kafka_username and self.kafka_password:
            self.producer_config.update({
                'sasl.username': self.kafka_username,
                'sasl.password': self.kafka_password,
                'security.protocol': 'SASL_SSL',
                'sasl.mechanisms': 'PLAIN'
            })
        else:
            self.producer_config ['security.protocol'] = 'PLAINTEXT'

        try:
            self.producer = Producer(self.producer_config)
            logger.info("Confluent Kafka Producer initialized successfully")
            self.running = True
        except Exception as e:
            logger.error(f"Failed to initialize confluent kafka producer: {str(e)}")
            raise e
            self.running = False

            # Initialize the transaction generator
            self.compromised_users = set(random.sample(range(1000, 9999), 50)) # 0.5% of users
            self.high_risk_merchants = ['QuickCash', 'PayDay', 'LoanShark', 'ShadyMerchants']
            self.high_risk_countries = ['Nigeria', 'Somalia', 'Yemen', 'Venezuela', 'Afghanistan', 'Syria']
            self.fraud_pattern_weights ={
                'compromised_user': 0.3,
                'high_risk_merchant': 0.2,
                'high_risk_country': 0.1,
                'card_testing': 0.3,
                'merchant_dispute': 0.1,
                'merchant_collusion': 0.2,
                'rare': 0.05,
                'geo_anomaly': 0.05
            }

            # Configure graceful shutdown
            signal.signal(signal.SIGINT, self.stop_production)
            signal.signal(signal.SIGTERM, self.stop_production)

    # Run Continuous production to Kafka to getting transactions
    def run_continuos_production(self, interval: float=0.0):
        self.running = True
        logger.info("Starting transaction producer for topic: %s", self.topic)
        try:
            while self.running:
                if self.produce_transaction():
                    logger.info("Transaction produced successfully")
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
        except ValidationError as e:
            logger.error(f"Invalid transaction: {str(e)}")
            return False
    
    # Generate transaction and test for fraud
    def generate_transaction(self) -> Optional [dict[str, any]]:
        transaction ={
            'transaction_id': fake.uuid4(),
            'event_time': fake.date_time_between(start_date='-1y', end_date='now').isoformat(),
            'user_id': fake.random_int(min=1000, max=9999),
            'amount': round(fake.pyfloat(min=0.01, max=10000) + fake.random(), 2),
            'currency': fake.currency_code('USD'),
            'merchant': fake.company(),
            'merchant_country': fake.country(),
            'timestamp': fake.date_time_between(start_date='-1y', end_date='now').timestamp().isoformat(),
            'transaction_type': 'purchase',
            'card_number': fake.credit_card_number(),
            'is_fraud': 0
        }

        is_fraud = 0
        amount = transaction['amount']
        user_id = transaction['user_id']
        merchant = transaction['merchant']
        merchant_country = transaction['merchant_country']

        # Account Takeover
        if user_id in self.compromised_users and amount > 500:
            if random.random() < 0.3:
                is_fraud = 1
                transaction['is_fraud'] = 1
                transaction['amount'] = random.uniform(500, 5000)
                transaction['merchant'] = random.choice(self.high_risk_merchants)
                transaction['transaction_type'] = 'cash_withdrawal'

        # Card testing
        if not is_fraud and amount < 2.0:
            # simulate rapid transactions
            if user_id % 1000 == 0 and random.random() < 0.25:            
                is_fraud = 1
                transaction['is_fraud'] = 1
                transaction['amount'] = round(random.uniform(0.01, 2),2)
                transaction['location'] = 'US'
                transaction['transaction_type'] = 'card_testing'

        # Merchant collusion
        if not is_fraud and merchant in self.high_risk_merchants:
            if amount >3000 and random.random() < 0.15:
                is_fraud = 1
                transaction['is_fraud'] = 1
                transaction['amount'] = round(random.uniform(300, 1500), 2)
                transaction['transaction_type'] = 'refund'

        # Merchant dispute
        if not is_fraud and merchant in self.high_risk_merchants:
            if random.random() < 0.1:
                is_fraud = 1
                transaction['is_fraud'] = 1
                transaction['amount'] = round(random.uniform(100, 500), 2)
                transaction['transaction_type'] = 'chargeback'

        # geography anomaly
        if not is_fraud and merchant_country in self.high_risk_countries:
            if user_id % 500 == 0 and random.random() < 0.01:
                is_fraud = 1
                transaction['is_fraud'] = 1
                transaction['location'] = random.choice(['CN', 'RU', 'GB', 'DE', 'FR', 'JP'])
                transaction['amount'] = round(random.uniform(100, 1000), 2)
                transaction['transaction_type'] = 'purchase'

        # Baseline random fraud (0.1 - 0.3)
        if not is_fraud and random.random() < 0.002:
            is_fraud = 1
            transaction['is_fraud'] = 1
            transaction['amount'] = round(random.uniform(100, 2000), 2)
            # transaction['transaction_type'] = 'purchase'

        # ensure that final fraud rate is between 1 - 2%
        transaction['is_fraud'] = is_fraud if random.random() < 0.985 else 0

        # validate modified transaction
        if self.validate_transaction(transaction):
            return transaction

    # Send transaction to Kafka
    def send_transaction(self, transaction: dict) -> bool:
        try:
            transaction = self.generate_transaction()
            if not transaction:
                return False

            self.producer.produce(
                self.topic,
                key=transaction['transaction_id'],
                value=json.dumps(transaction),
                callback=self.delivery_report
                )
            self.producer.poll(0) # polling to trigger the callback            
            return True
        except Exception as e:
            logger.error(f"Failed to send transaction: {str(e)}")
            return False     

    # stop production transaction
    def stop_production(self, signum, frame):
        if self.running:
            logger.info("Stopping transaction producer")
            self.running = False

            if self.producer:
                self.producer.flush(timeout=20)
                logger.info("Flushing confluent kafka producer")
                self.producer.close()
                logger.info("Closing confluent kafka producer")


if __name__ == "__main__":
    producer = TransactionProducer()
    producer.run_continuos_production()

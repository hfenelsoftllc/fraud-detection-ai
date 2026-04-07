import os
import json
import signal
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from confluent_kafka import Consumer, KafkaError, KafkaException

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="/app/.env")


class FraudNotificationService:
    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.fraud_topic = os.getenv("KAFKA_FRAUD_TOPIC", "fraud-alerts")
        self.group_id = os.getenv("KAFKA_GROUP_ID", "notification-group")
        self.running = False

        # SMTP config (optional — leave blank to disable email)
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.alert_from_email = os.getenv(
            "ALERT_FROM_EMAIL", "fraud-alerts@resend.dev"
        )
        self.alert_to_email = os.getenv("ALERT_TO_EMAIL")

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
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            **base_config,
        }

        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([self.fraud_topic])

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def _send_email(self, alert: dict):
        if not all(
            [self.smtp_host, self.smtp_user, self.smtp_password, self.alert_to_email]
        ):
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self.alert_from_email
            msg["To"] = self.alert_to_email
            msg["Subject"] = f"[FRAUD ALERT] Transaction {alert.get('transaction_id')}"

            body = (
                f"Fraudulent transaction detected.\n\n"
                f"Transaction ID : {alert.get('transaction_id')}\n"
                f"User ID        : {alert.get('user_id')}\n"
                f"Amount         : {float(alert.get('amount', 0)):.2f} {alert.get('currency', '')}\n"
                f"Merchant       : {alert.get('merchant')}\n"
                f"Country        : {alert.get('merchant_country')}\n"
                f"Type           : {alert.get('transaction_type')}\n"
                f"Time           : {alert.get('event_time')}\n"
                f"Alert Reason   : {alert.get('alert_reason')}\n"
            )
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(
                    self.alert_from_email, self.alert_to_email, msg.as_string()
                )

            logger.info(
                "Email alert sent for transaction %s", alert.get("transaction_id")
            )
        except Exception as e:
            logger.error("Failed to send email alert: %s", str(e))

    def _log_alert(self, alert: dict):
        logger.warning(
            "FRAUD ALERT | txn_id=%s | user=%s | amount=%.2f | merchant=%s | type=%s | reason=%s",
            alert.get("transaction_id"),
            alert.get("user_id"),
            float(alert.get("amount", 0)),
            alert.get("merchant"),
            alert.get("transaction_type"),
            alert.get("alert_reason"),
        )

    def run(self):
        self.running = True
        logger.info(
            "Notification service started. Listening on topic: %s", self.fraud_topic
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
                    alert = json.loads(msg.value().decode("utf-8"))
                    self._log_alert(alert)
                    self._send_email(alert)
                    self.consumer.commit(asynchronous=False)
                except json.JSONDecodeError as e:
                    logger.error("Failed to decode alert message: %s", str(e))
                except Exception as e:
                    logger.error("Error processing alert: %s", str(e))

        except KafkaException as e:
            logger.error("Kafka exception: %s", str(e))
        finally:
            self.consumer.close()
            logger.info("Notification service shut down cleanly")

    def shutdown(self, signum, frame):
        logger.info("Shutting down notification service...")
        self.running = False


if __name__ == "__main__":
    service = FraudNotificationService()
    service.run()

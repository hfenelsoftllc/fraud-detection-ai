# ==============================================================================
# Fraud Detection System — Makefile
# ------------------------------------------------------------------------------
# Usage:
#   make help                  Show this help
#
#   # Docker Compose (recommended for full stack)
#   make up                    Build images and start all services
#   make down                  Stop and remove containers
#   make build                 Rebuild all images without starting
#   make logs                  Tail logs for all services
#   make logs-<svc>            Tail logs for a specific service
#   make restart-<svc>         Restart a specific service
#   make ps                    Show running containers and health status
#
#   # Local Python development (outside Docker)
#   make venvs                 Create one venv per application
#   make install               Install requirements in every venv
#   make install-producer      Install only producer requirements
#   make install-consumer      Install only consumer requirements
#   make install-notification  Install only notification requirements
#   make install-models        Install only model-training requirements
#   make install-airflow       Install only Airflow requirements
#   make install-mlflow        Install only MLflow server requirements
#
#   # Local sequential execution (outside Docker — Kafka must be reachable)
#   make run-all               Run all services sequentially (background daemons)
#   make run-train             Train the model locally
#   make run-producer          Start the producer (foreground)
#   make run-consumer          Start the consumer (foreground)
#   make run-notification      Start the notification service (foreground)
#
#   make clean                 Remove all local venvs and __pycache__
# ==============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON       ?= python3
PIP          ?= pip
COMPOSE      ?= docker compose
COMPOSE_FILE  = docker-compose.yml
SRC_DIR      := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

APPS := producer consumer notification models airflow mlflow

# Venv paths (one per app, isolated)
VENV_PRODUCER     = $(SRC_DIR)producer/.venv
VENV_CONSUMER     = $(SRC_DIR)consumer/.venv
VENV_NOTIFICATION = $(SRC_DIR)notification/.venv
VENV_MODELS       = $(SRC_DIR)models/.venv
VENV_AIRFLOW      = $(SRC_DIR)airflow/.venv
VENV_MLFLOW       = $(SRC_DIR)mlflow/.venv

# Python interpreter inside each venv
PY_PROD  = $(VENV_PRODUCER)/bin/python
PY_CONS  = $(VENV_CONSUMER)/bin/python
PY_NOTIF = $(VENV_NOTIFICATION)/bin/python
PY_MODEL = $(VENV_MODELS)/bin/python
PY_AFLOW = $(VENV_AIRFLOW)/bin/python
PY_MFLOW = $(VENV_MLFLOW)/bin/python

# On Windows, Python lives under Scripts/ not bin/
ifeq ($(OS),Windows_NT)
  PY_PROD  = $(VENV_PRODUCER)/Scripts/python
  PY_CONS  = $(VENV_CONSUMER)/Scripts/python
  PY_NOTIF = $(VENV_NOTIFICATION)/Scripts/python
  PY_MODEL = $(VENV_MODELS)/Scripts/python
  PY_AFLOW = $(VENV_AIRFLOW)/Scripts/python
  PY_MFLOW = $(VENV_MLFLOW)/Scripts/python
endif

.DEFAULT_GOAL := help
.PHONY: help \
        up down build logs ps restart \
        venvs \
        install install-producer install-consumer install-notification \
        install-models install-airflow install-mlflow \
        run-all run-train run-producer run-consumer run-notification \
        clean

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Fraud Detection System — available targets"
	@echo "  ─────────────────────────────────────────────────────────────────"
	@echo "  Docker Compose"
	@echo "    up                   Build images and start all services"
	@echo "    down                 Stop and remove containers"
	@echo "    build                Rebuild images without starting"
	@echo "    logs                 Tail logs for all services"
	@echo "    ps                   Show container status"
	@echo ""
	@echo "  Local Python (venv per application)"
	@echo "    venvs                Create venvs for all applications"
	@echo "    install              Install all requirements"
	@echo "    install-producer     Install producer requirements"
	@echo "    install-consumer     Install consumer requirements"
	@echo "    install-notification Install notification requirements"
	@echo "    install-models       Install model-training requirements"
	@echo "    install-airflow      Install Airflow requirements"
	@echo "    install-mlflow       Install MLflow server requirements"
	@echo ""
	@echo "  Local Sequential Execution (Kafka must be reachable)"
	@echo "    run-all              Run train → producer → consumer → notification"
	@echo "    run-train            Train the fraud detection model"
	@echo "    run-producer         Run the transaction producer (foreground)"
	@echo "    run-consumer         Run the fraud-detection consumer (foreground)"
	@echo "    run-notification     Run the notification service (foreground)"
	@echo ""
	@echo "    clean                Remove venvs and __pycache__"
	@echo ""

# ── Docker Compose ────────────────────────────────────────────────────────────
up:
	$(COMPOSE) -f $(COMPOSE_FILE) up --build -d
	@echo ""
	@echo "  Services started.  UIs:"
	@echo "    Airflow  → http://localhost:8080"
	@echo "    MLflow   → http://localhost:5500"
	@echo "    Kafka UI → http://localhost:8090"
	@echo "    MinIO    → http://localhost:9001"
	@echo ""

down:
	$(COMPOSE) -f $(COMPOSE_FILE) down --remove-orphans

build:
	$(COMPOSE) -f $(COMPOSE_FILE) build

logs:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f

## Convenience: make logs-producer  / make logs-consumer  / etc.
logs-%:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f $*

## Convenience: make restart-producer / make restart-consumer / etc.
restart-%:
	$(COMPOSE) -f $(COMPOSE_FILE) restart $*

ps:
	$(COMPOSE) -f $(COMPOSE_FILE) ps

# ── Virtual environments (one per service) ────────────────────────────────────
$(VENV_PRODUCER):
	$(PYTHON) -m venv $(VENV_PRODUCER)

$(VENV_CONSUMER):
	$(PYTHON) -m venv $(VENV_CONSUMER)

$(VENV_NOTIFICATION):
	$(PYTHON) -m venv $(VENV_NOTIFICATION)

$(VENV_MODELS):
	$(PYTHON) -m venv $(VENV_MODELS)

$(VENV_AIRFLOW):
	$(PYTHON) -m venv $(VENV_AIRFLOW)

$(VENV_MLFLOW):
	$(PYTHON) -m venv $(VENV_MLFLOW)

venvs: $(VENV_PRODUCER) $(VENV_CONSUMER) $(VENV_NOTIFICATION) \
       $(VENV_MODELS) $(VENV_AIRFLOW) $(VENV_MLFLOW)
	@echo "All virtual environments created."

# ── Install requirements ──────────────────────────────────────────────────────
install-producer: $(VENV_PRODUCER)
	$(PY_PROD) -m pip install --upgrade pip
	$(PY_PROD) -m pip install -r $(SRC_DIR)producer/requirements.txt
	@echo "[producer] requirements installed."

install-consumer: $(VENV_CONSUMER)
	$(PY_CONS) -m pip install --upgrade pip
	$(PY_CONS) -m pip install -r $(SRC_DIR)consumer/requirements.txt
	@echo "[consumer] requirements installed."

install-notification: $(VENV_NOTIFICATION)
	$(PY_NOTIF) -m pip install --upgrade pip
	$(PY_NOTIF) -m pip install -r $(SRC_DIR)notification/requirements.txt
	@echo "[notification] requirements installed."

install-models: $(VENV_MODELS)
	$(PY_MODEL) -m pip install --upgrade pip
	$(PY_MODEL) -m pip install -r $(SRC_DIR)models/requirements.txt
	@echo "[models] requirements installed."

install-airflow: $(VENV_AIRFLOW)
	$(PY_AFLOW) -m pip install --upgrade pip
	$(PY_AFLOW) -m pip install -r $(SRC_DIR)airflow/requirements.txt
	@echo "[airflow] requirements installed."

install-mlflow: $(VENV_MLFLOW)
	$(PY_MFLOW) -m pip install --upgrade pip
	$(PY_MFLOW) -m pip install -r $(SRC_DIR)mlflow/requirements.txt
	@echo "[mlflow] requirements installed."

## Install all requirements sequentially (safe order: models first, then services)
install: install-models install-mlflow install-producer \
         install-consumer install-notification install-airflow
	@echo ""
	@echo "All requirements installed successfully."

# ── Local sequential execution ────────────────────────────────────────────────
## 1. Train the model (blocking — waits until training completes)
run-train: install-models
	@echo ">>> [1/4] Training fraud detection model..."
	$(PY_MODEL) $(SRC_DIR)models/train_model.py \
		--n-samples 100000 \
		--model-dir  $(SRC_DIR)models
	@echo ">>> Model training complete."

## 2. Start the transaction producer in the background
run-producer: install-producer
	@echo ">>> [2/4] Starting transaction producer (background)..."
	$(PY_PROD) $(SRC_DIR)producer/main.py &
	@echo ">>> Producer started (PID $$!)."

## 3. Start the fraud-detection consumer in the background
run-consumer: install-consumer
	@echo ">>> [3/4] Starting fraud detection consumer (background)..."
	$(PY_CONS) $(SRC_DIR)consumer/main.py &
	@echo ">>> Consumer started (PID $$!)."

## 4. Start the notification service in the foreground (last — keeps terminal alive)
run-notification: install-notification
	@echo ">>> [4/4] Starting notification service (foreground)..."
	$(PY_NOTIF) $(SRC_DIR)notification/main.py

## Full sequential run: train → producer (bg) → consumer (bg) → notification (fg)
## Prerequisite: Kafka broker must be reachable (run `make up` first or have a
## local broker on localhost:9092).
run-all: run-train run-producer run-consumer run-notification

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "Removing virtual environments..."
	rm -rf $(VENV_PRODUCER) $(VENV_CONSUMER) $(VENV_NOTIFICATION) \
	       $(VENV_MODELS) $(VENV_AIRFLOW) $(VENV_MLFLOW)
	@echo "Removing __pycache__ directories..."
	find $(SRC_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find $(SRC_DIR) -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."

FROM python:3.11-slim-bookworm

# set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=UTF-8 \
    APP_HOME=/app

# set work directory
WORKDIR $APP_HOME

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gnupg2 \
    gcc \
    g++ \
    libpq-dev \
    python3-dev \
    libssl-dev \
    libsasl2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Manually install librdkafka with SASL/SSL support
RUN curl -fsSL https://packages.confluent.io/deb/6.2/archive.key | apt-key add - \
    && echo "deb [arch=amd64] https://packages.confluent.io/deb/6.2 stable main" >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

#Copy the requirements file and install Python dependencies
COPY requirements.txt ${APP_HOME}/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#`COPY` the rest of the application code
COPY . ${APP_HOME}

#Entrypoint
ENTRYPOINT ["python3", "main.py"]
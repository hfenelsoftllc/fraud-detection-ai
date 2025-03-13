FROM apache/airflow:2.10.5

#create model directory and set ownership
USER root
RUN mkdir -p /app/models

USER airflow

#copy requirements file
COPY requirements.txt /tmp/requirements.txt

#Install python packages as root
RUN pip install --no-cache-dir -r /tmp/requirements.txt

#Switch back to airflow user for safety
USER airflow
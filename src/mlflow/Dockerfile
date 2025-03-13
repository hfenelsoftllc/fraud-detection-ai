FROM python:3.11-slim-bookworm

WORKDIR /app
COPY requirements.txt /app/requirements.txt

#install python packages dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the code
COPY . .

# Set the command to run you app
EXPOSE 5500

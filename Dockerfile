# Use the official Python image from DockerHub
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the application files into the container
COPY . /app

# Add the /app folder to PYTHONPATH so that the `surimi` module is found
ENV PYTHONPATH=/app

# Expose server port
EXPOSE 5001

ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:18889
ENV OTEL_SERVICE_NAME=market

# Runthe server 
CMD ["python", "server/app.py"]

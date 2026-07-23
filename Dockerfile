FROM python:3.12-slim AS builder

WORKDIR /app

COPY ./requirements.txt ./requirements.txt 

RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------

# Use the official Python image from DockerHub
FROM python:3.12-slim AS runner

# Upgrade system packages to address vulnerabilities
# RUN apt-get update && apt-get upgrade -y && apt-get clean

# Set the working directory inside the container
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copy the requirements.txt file into the container
COPY ./src ./src
COPY ./server ./server
COPY ./util ./util


# Add the /app folder to PYTHONPATH so that the `surimi` module is found
ENV PYTHONPATH=/app

# Expose server port
EXPOSE ${GRPC_PORT-5001}

ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
ENV OTEL_SERVICE_NAME=market

# Run the server -u option is used to ensure that output (logs) is flushed immediately
CMD ["python", "-u", "src/app.py"]
# Use Python 3.12 as base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy dependencies first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto files
COPY ./proto /app/proto/

# Copy the entire src directory
COPY ./src /app/src/

# Copy the entire src pb
COPY ./pb /app/pb/

# Copy the market_client.py file
COPY ./client/market_client.py /app/client/

# Debugging: Check if Market.py exists inside the container
RUN ls -l /app/src/Market.py || echo "Market.py NOT FOUND!"

# Add src to Python's module search path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Compile the gRPC protobuf file
RUN python -m grpc_tools.protoc -I/app/proto --python_out=. --grpc_python_out=. /app/proto/market.proto

# Expose the gRPC port
EXPOSE 50051

# Run the gRPC client
CMD ["python", "/app/client/market_client.py"]

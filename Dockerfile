# Use the official Python image from DockerHub
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the application files into the container, including pb directory
COPY . /app

# Ensure the pb directory is inside /app/pb
RUN ls /app/pb  # This is just a test step to confirm pb exists inside the container

# Expose both the client and server ports
EXPOSE 5000 5001

# Run both the server and client with the correct order
CMD python server/market_service.py & python client/market_client.py

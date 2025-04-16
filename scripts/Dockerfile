# Use official Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt and Makefile for caching
COPY requirements.txt Makefile ./

# Install Python dependencies using Makefile
RUN make install

# Copy the rest of the application code (including the Data folder and ingest_to_s3.py)
COPY Data/ /app/Data/
COPY ingest_to_s3.py /app/ingest_to_s3.py

# Optional: Run code formatting and linting during build
RUN make refactor

# Set default command to run your script
CMD ["python", "ingest_to_s3.py"]

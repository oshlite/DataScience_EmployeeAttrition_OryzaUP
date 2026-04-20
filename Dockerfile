# Use Python 3.11 slim image as base
FROM python:3.11.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directories if they don't exist
RUN mkdir -p model static/dashboards

# Expose ports
EXPOSE 5000 8501

# Set environment variables
ENV FLASK_APP=web_app.py
ENV PYTHONUNBUFFERED=1

# Default command runs with Gunicorn
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 4 web_app:app

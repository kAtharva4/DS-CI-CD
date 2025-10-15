# Use a lightweight, stable Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (including model file)
COPY . .

# Expose FastAPI's default port
EXPOSE 8000

# Optional: set environment variables
ENV APP_NAME="McDonalds_Sentiment_API"

# Run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

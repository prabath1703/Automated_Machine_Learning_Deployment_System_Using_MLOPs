# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Command to run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

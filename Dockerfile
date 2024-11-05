# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set Python path to include the app directory
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose the port your application runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "run.py"]
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# CRITICAL: Tell Python to look in the current directory for modules
ENV PYTHONPATH=/app

# Run the script
CMD ["python", "inference.py"]

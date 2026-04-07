FROM python:3.11-slim

WORKDIR /app

# 1. Ensure requirements are installed first (standard practice)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the rest of the project
COPY . .

# 3. Environment Variables

ENV OPENAI_API_KEY=""
ENV API_BASE_URL=""
ENV MODEL_NAME=""


CMD ["python", "inference.py"]
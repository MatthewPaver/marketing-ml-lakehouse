# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY lakehouse/requirements.txt /app/lakehouse/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/lakehouse/requirements.txt

# Copy project
COPY . /app

ENV PYTHONPATH=/app

# Default command can be overridden; provide a help text
CMD ["bash", "-lc", "echo 'Use: python -m lakehouse.run_all or streamlit run lakehouse/dashboard/app.py' && bash"]

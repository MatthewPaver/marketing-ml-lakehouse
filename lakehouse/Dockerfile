# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY lakehouse/requirements.txt /app/lakehouse/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/lakehouse/requirements.txt

# Copy project
COPY . /app

ENV PYTHONPATH=/app
ARG APP_UID=10001
ARG APP_GID=10001
RUN test "$APP_UID" -gt 0 && test "$APP_GID" -gt 0 \
    && (getent group "$APP_GID" || groupadd --gid "$APP_GID" lakehouse) \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --create-home lakehouse \
    && mkdir -p /app/lakehouse/artifacts /app/lakehouse/models /app/docs/generated \
    && chown "$APP_UID:$APP_GID" /app/lakehouse /app/lakehouse/artifacts \
        /app/lakehouse/models /app/docs/generated \
    && find /app/lakehouse/artifacts /app/lakehouse/models /app/docs/generated \
        -maxdepth 1 -type f -exec chown "$APP_UID:$APP_GID" {} + \
    && find /app/lakehouse -maxdepth 1 -type f -name 'lakehouse.duckdb*' \
        -exec chown "$APP_UID:$APP_GID" {} +
ENV HOME=/home/lakehouse
ENV PYTHONDONTWRITEBYTECODE=1
USER ${APP_UID}:${APP_GID}
ENTRYPOINT ["python", "-B", "-m", "lakehouse.container_entrypoint"]

# Default command can be overridden; provide a help text
CMD ["bash", "-lc", "echo 'Use: python -m lakehouse.run_all or streamlit run lakehouse/dashboard/app.py' && bash"]

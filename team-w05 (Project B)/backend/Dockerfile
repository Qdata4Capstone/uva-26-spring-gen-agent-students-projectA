FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY mcp_server/ /mcp_server/

EXPOSE 8000

CMD ["sh", "-c", "cd /app && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
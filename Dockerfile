FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
# CPU-only torch: the default PyPI wheel bundles CUDA libs the Cloud Run
# instance can never use, at a cost of several GB of image size.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m spacy download en_core_web_sm

COPY download_model.py /tmp/download_model.py
RUN python /tmp/download_model.py \
    && rm /tmp/download_model.py

COPY backend/ /app/backend/

WORKDIR /app/backend

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
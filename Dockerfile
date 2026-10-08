FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements-ocr.txt
COPY pipeline ./pipeline
COPY frontend ./frontend
COPY benchmarks ./benchmarks
ENV DATABASE_PATH=/app/data/scrawlguard.sqlite3
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "pipeline.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

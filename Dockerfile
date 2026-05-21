FROM python:3.11-slim

WORKDIR /app
ENV RAG_CORPUS_PATH=data/scifact/corpus.jsonl
ENV RAG_STORAGE_DIR=storage/scifact
ENV RAG_LOG_DIR=logs

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app .
COPY data ./data
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

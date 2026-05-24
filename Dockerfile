FROM python:3.11-slim

WORKDIR /app
ENV RAG_CORPUS_PATH=data/scifact/corpus.jsonl
ENV RAG_STORAGE_DIR=storage/scifact
ENV RAG_LOG_DIR=logs
ENV RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# Reranker is off by default; the eval profile / explicit env can enable it.
ENV RAG_ENABLE_RERANKER=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Warm the HF cache for the embedding + cross-encoder so cold-start latency
# (especially when the eval profile flips ENABLE_RERANKER on) stays
# predictable inside Docker. Failure here must not break the image build.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${RAG_EMBED_MODEL}'); \
CrossEncoder('${RAG_RERANKER_MODEL}')" || true

COPY app .
COPY data ./data
COPY tests ./tests
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

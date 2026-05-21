from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if __package__:
    from .scifact import (
        DEFAULT_TOP_K,
        MAX_TOP_K,
        peek_scifact_retriever,
        scifact_status,
        start_scifact_retriever_initialization,
    )
else:
    from scifact import (
        DEFAULT_TOP_K,
        MAX_TOP_K,
        peek_scifact_retriever,
        scifact_status,
        start_scifact_retriever_initialization,
    )


app = FastAPI(title="SciFact Retrieval API", version="2.0.0")


class RetrieveRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class Context(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str


class RetrieveResponse(BaseModel):
    query_id: str
    contexts: list[Context]


@app.on_event("startup")
def startup():
    start_scifact_retriever_initialization()


@app.get("/health")
def health():
    start_scifact_retriever_initialization()
    return scifact_status()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    start_scifact_retriever_initialization()
    retriever = peek_scifact_retriever()
    if retriever is None or not retriever.ready:
        raise HTTPException(
            status_code=503,
            detail=(retriever.error if retriever else None) or "SciFact retriever is still initializing.",
        )

    contexts = retriever.retrieve(request.question, request.top_k)
    if not contexts:
        raise HTTPException(status_code=503, detail="No relevant SciFact contexts found.")

    return {"query_id": request.query_id, "contexts": contexts}

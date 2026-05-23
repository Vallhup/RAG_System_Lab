import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


LOG_DIR = Path(os.getenv("RAG_LOG_DIR", "logs"))
_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return

    app_dir = LOG_DIR / "app"
    retrieval_dir = LOG_DIR / "retrieval"
    evaluation_dir = LOG_DIR / "evaluation"
    security_dir = LOG_DIR / "security"
    error_event_dir = LOG_DIR / "errors"
    index_dir = LOG_DIR / "indexing"
    for directory in (app_dir, retrieval_dir, evaluation_dir, security_dir, error_event_dir, index_dir):
        directory.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    app_handler = _file_handler(app_dir / "app.log", logging.INFO, formatter)
    error_handler = _file_handler(app_dir / "error.log", logging.ERROR, formatter)
    root.addHandler(app_handler)
    root.addHandler(error_handler)

    index_logger = logging.getLogger("rag.index")
    index_logger.setLevel(logging.INFO)
    index_logger.propagate = True
    index_logger.addHandler(_file_handler(index_dir / "index.log", logging.INFO, formatter))

    _configured = True


def log_retrieval_event(event: dict[str, Any]) -> None:
    log_event("retrieval", "retrieve.jsonl", event)


def log_security_event(event: dict[str, Any]) -> None:
    log_event("security", "audit.jsonl", event)


def log_failure_event(event: dict[str, Any]) -> None:
    log_event("errors", "failures.jsonl", event)


def log_evaluation_event(event: dict[str, Any]) -> None:
    log_event("evaluation", "runs.jsonl", event)


def log_event(category: str, filename: str, event: dict[str, Any]) -> None:
    event_dir = LOG_DIR / category
    event_dir.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _now(), **event}
    with (event_dir / filename).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _file_handler(
    path: Path,
    level: int,
    formatter: logging.Formatter,
) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        path,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

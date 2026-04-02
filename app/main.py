import logging
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from llama_index.core import Settings, SimpleDirectoryReader, SummaryIndex
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms import MockLLM
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.file import (
    DocxReader,
    FlatReader,
    HWPReader,
    ImageReader,
    PDFReader,
)
from llama_index.readers.wikipedia import WikipediaReader

PAGE_TITLES = ["Artificial intelligence"]
WIKI_QUERY = "What is this article mainly about?"
DATA_QUERY = "Summarize the main topics covered across all files in the data folder."
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".hwp", ".png"]
DEFAULT_LLM_PROVIDER = "extractive"
WORD_PATTERN = r"[A-Za-z\uac00-\ud7a3]{2,}"
PROMPT_NOISE_PHRASES = (
    "context information is below",
    "given the context information",
    "using both the context information",
    "we have provided an existing answer",
    "we have the opportunity to refine",
    "the original query is as follows",
    "query:",
    "existing answer:",
    "refined answer",
    "answer the question",
    "*note:",
)

FILE_QUERY_TESTS = {
    "Text.txt": [
        {
            "question": "한계기업의 정의는 무엇인가?",
            "expected": "최근 3개 회계연도 말 이자보상비율이 연속으로 1 미만인 기업",
        },
        {
            "question": "공동연구개발기관이 제조기업인 경우 최근 3개 회계연도 평균 매출액 기준은 얼마 이상인가?",
            "expected": "3개년 평균 50억 이상",
        },
    ],
    "Policy.pdf": [
        {
            "question": "제1조 목적은 무엇인가?",
            "expected": "연구과제 관리 및 연구비 집행에 관한 세부사항을 정하는 것",
        },
        {
            "question": "연구자의 책임과 의무 중 연구비와 직접 관련된 항목은 무엇인가?",
            "expected": "연구비의 투명한 집행 및 비목별 집행기준 준수",
        },
    ],
    "SRS.docx": [
        {
            "question": "이 문서의 목적은 무엇인가?",
            "expected": "소프트웨어 요구사항을 식별하고 명세하는 것",
        },
        {
            "question": "이 문서에서 요구사항 명세를 위해 사용하는 다이어그램은 무엇인가?",
            "expected": "Data Flow Diagram 및 Sequence Diagram",
        },
    ],
    "CRA.xlsx": [
        {
            "question": "DPR-Req.-001: 2의 변경 상태와 수용 여부는 무엇인가?",
            "expected": "Change Status는 Modified, 수용여부는 accepted",
        },
        {
            "question": "Requirements for mechanical loads 항목의 변경 상태는 무엇인가?",
            "expected": "Added",
        },
    ],
    "DESIGN.hwp": [
        {
            "question": "공고 번호와 공고 제목은 무엇인가?",
            "expected": "산업통상부 공고 제2026-150호, 2026년도 디자인산업기술개발사업 신규지원 대상과제 공고",
        },
        {
            "question": "1-1. 사업목적의 핵심 내용은 무엇인가?",
            "expected": "디자인융합 혁신 기술개발 지원을 통해 고부가가치를 창출하고 미래성장동력을 확보하는 것",
        },
    ],
    "Image.png": [
        {
            "question": "mermaids의 migration pattern은 무엇에 따라 달라지는가?",
            "expected": "species, age, environmental conditions",
        },
        {
            "question": "겨울철 mermaids는 어디로 이동하는가?",
            "expected": "Arctic 인근 summer feeding grounds에서 남쪽으로 이동",
        },
    ],
}


class ExtractiveLLM(CustomLLM):
    max_tokens: int = 256
    model_name: str = "extractive-llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=False,
        )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        return CompletionResponse(text=self._build_answer(prompt))

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponseGen:
        text = self._build_answer(prompt)

        def gen() -> CompletionResponseGen:
            partial = ""
            for char in text:
                partial += char
                yield CompletionResponse(text=partial, delta=char)

        return gen()

    def _build_answer(self, prompt: str) -> str:
        query = self._extract_query(prompt)
        context = self._extract_context(prompt)
        candidates = self._split_candidates(context)
        ranked = self._rank_candidates(candidates, query)

        if ranked:
            answer = " ".join(ranked[:3])
        else:
            answer = context[: self.max_tokens * 4].strip()

        return answer or "No useful context found."

    def _extract_query(self, prompt: str) -> str:
        patterns = [
            r"Query:\s*(.+)",
            r"question\s*[:=]\s*(.+)",
            r"query\s*[:=]\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_context(self, prompt: str) -> str:
        block_patterns = [
            (
                r"Context information is below\.\s*-+\s*(.*?)\s*-+\s*"
                r"(?:Given the context information|Using both the context information|Query:)"
            ),
            (
                r"We have provided context information below\.\s*-+\s*(.*?)\s*-+\s*"
                r"(?:Given the context information|Query:)"
            ),
        ]
        for pattern in block_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return prompt.strip()

    def _split_candidates(self, context: str) -> list[str]:
        raw_parts = re.split(r"\n\s*\n+", context)
        if len(raw_parts) <= 1:
            raw_parts = re.split(r"(?<=[.!?])\s+|\n+", context)

        candidates = []
        seen = set()
        for part in raw_parts:
            cleaned = " ".join(part.split()).strip()
            lower = cleaned.lower()
            if len(cleaned) < 30:
                continue
            if any(phrase in lower for phrase in PROMPT_NOISE_PHRASES):
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            candidates.append(cleaned)
        return candidates

    def _rank_candidates(self, candidates: list[str], query: str) -> list[str]:
        query_terms = {
            term.lower()
            for term in re.findall(WORD_PATTERN, query)
            if len(term) >= 2
        }

        def score(text: str) -> tuple[int, int]:
            words = set(re.findall(WORD_PATTERN, text.lower()))
            overlap = len(query_terms & words)
            return overlap, len(text)

        return sorted(candidates, key=score, reverse=True)


class StructuredExcelReader(BaseReader):
    def load_data(
        self, file: Path, extra_info: dict | None = None, fs=None
    ) -> list[Document]:
        if fs:
            with fs.open(file, "rb") as handle:
                df = pd.read_excel(handle, header=None)
        else:
            df = pd.read_excel(file, header=None)

        cleaned = df.fillna("").map(self._clean_cell)
        data_start = self._find_data_start(cleaned)
        header_rows = cleaned.iloc[max(0, data_start - 3) : data_start]

        column_headers = self._build_column_headers(header_rows)
        row_texts = []

        for _, row in cleaned.iloc[data_start:].iterrows():
            fields = []
            for column_index, value in enumerate(row):
                if not value or value == "-":
                    continue
                header = column_headers.get(column_index, f"column_{column_index}")
                fields.append(f"{header}: {value}")

            if fields:
                row_texts.append("\n".join(fields))

        metadata = dict(extra_info or {})
        metadata["structured_format"] = "header_value_rows"
        text = "\n\n".join(row_texts)
        return [Document(text=text, metadata=metadata)]

    def _clean_cell(self, value) -> str:
        return " ".join(str(value).split()).strip()

    def _find_data_start(self, df: pd.DataFrame) -> int:
        requirement_pattern = re.compile(r"[A-Za-z]+-Req", re.IGNORECASE)

        for index, row in df.iterrows():
            first_cell = row.iloc[0]
            if isinstance(first_cell, str) and requirement_pattern.search(first_cell):
                return index

        for index, row in df.iterrows():
            non_empty_count = sum(1 for value in row if value not in ("", "-"))
            if index > 0 and non_empty_count >= 3:
                return index

        return 0

    def _build_column_headers(self, header_rows: pd.DataFrame) -> dict[int, str]:
        headers = {}
        ignored_values = {"", "-", "example"}

        for column_index in header_rows.columns:
            values = []
            seen = set()
            for value in header_rows[column_index]:
                if not value:
                    continue
                lower = value.lower() if isinstance(value, str) else value
                if value in ignored_values or lower in ignored_values:
                    continue
                if value in seen:
                    continue
                seen.add(value)
                values.append(value)

            if not values:
                headers[column_index] = f"column_{column_index}"
            elif len(values) == 1:
                headers[column_index] = values[0]
            else:
                headers[column_index] = " / ".join(values[-2:])

        return headers


class CleanHWPReader(BaseReader):
    def __init__(self):
        self._reader = HWPReader()

    def load_data(
        self, file: Path, extra_info: dict | None = None, fs=None
    ) -> list[Document]:
        documents = self._reader.load_data(file, extra_info=extra_info, fs=fs)
        cleaned_documents = []

        for document in documents:
            cleaned_text = self._clean_text(document.text)
            cleaned_documents.append(
                Document(text=cleaned_text, metadata=dict(document.metadata))
            )

        return cleaned_documents

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
        text = re.sub(
            r"[^A-Za-z0-9\uac00-\ud7a3\s\.,!?;:'\"()\[\]{}<>\-–—~/%&+·•※○□△:]",
            " ",
            text,
        )

        lines = []
        for raw_line in text.splitlines():
            line = " ".join(raw_line.split()).strip()
            if not line:
                continue
            if not re.search(WORD_PATTERN, line):
                continue
            lines.append(line)

        return "\n".join(lines)


def build_llm():
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()

    if provider == "mock":
        return MockLLM(max_tokens=256)

    # Future providers such as Ollama can be added here without changing
    # the rest of the indexing or query pipeline.
    return ExtractiveLLM(max_tokens=256)


def configure_settings():
    log_dir = Path("/logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "debug.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    Settings.llm = build_llm()
    logging.info(
        "Using LLM provider: %s",
        os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
    )


def preview_documents(documents, label):
    print(f"\n=== {label} ===")
    print("len(documents) =", len(documents))
    if documents:
        print("documents[0].metadata =", documents[0].metadata)


def preview_nodes(nodes):
    print("\n=== Node Parsing ===")
    print("len(nodes) =", len(nodes))
    if nodes:
        print("nodes[0].text[:200] =", nodes[0].text[:200])
        print("nodes[0].metadata =", nodes[0].metadata)


def shorten_text(text: str, limit: int = 260) -> str:
    flat = " ".join(text.split()).strip()
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def build_index(documents):
    nodes = SimpleNodeParser().get_nodes_from_documents(documents)
    preview_nodes(nodes)

    print("\n=== SummaryIndex Build ===")
    index = SummaryIndex(nodes)
    print("SummaryIndex build complete")
    return index


def run_query(index, query_text):
    print("\n=== Query Engine ===")
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)

    print("question =", query_text)
    print("response =")
    print()
    print(str(response))
    if response.source_nodes:
        print("\nTop source chunk =")
        print()
        print(response.source_nodes[0].node.get_content())


def run_wikipedia_demo():
    print("=== Block 1: Wikipedia Document Load ===")
    reader = WikipediaReader()
    documents = reader.load_data(pages=PAGE_TITLES)

    preview_documents(documents, "Wikipedia Documents")
    if not documents:
        print("Document load failed")
        return

    index = build_index(documents)
    run_query(index, WIKI_QUERY)


def resolve_data_dir():
    candidates = [
        Path("/data"),
        Path(__file__).resolve().parent.parent / "data",
        Path("data"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the data directory.")


def build_file_extractor():
    return {
        ".txt": FlatReader(),
        ".pdf": PDFReader(),
        ".docx": DocxReader(),
        ".xlsx": StructuredExcelReader(),
        ".hwp": CleanHWPReader(),
        ".png": ImageReader(
            parse_text=True,
            text_type="plain_text",
            pytesseract_model_kwargs={"lang": "kor+eng"},
        ),
    }


def load_documents_for_file(file_path: Path):
    extractor = build_file_extractor()
    reader = extractor[file_path.suffix.lower()]
    return reader.load_data(file_path, extra_info={"file_name": file_path.name})


def get_document_name(document):
    metadata = document.metadata
    return (
        metadata.get("file_name")
        or metadata.get("filename")
        or metadata.get("file_path")
        or "unknown"
    )


def load_data_folder_documents():
    print("\n=== Block 5: Data Folder Load ===")
    data_dir = resolve_data_dir()
    file_extractor = build_file_extractor()

    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        file_extractor=file_extractor,
        required_exts=SUPPORTED_EXTENSIONS,
    )
    documents = reader.load_data()

    preview_documents(documents, "Data Folder Documents")

    counts = Counter(get_document_name(document) for document in documents)
    print("documents_per_file =", dict(counts))

    for index, document in enumerate(documents[:6]):
        text = getattr(document, "text", "") or ""
        print(f"\n--- document[{index}] ---")
        print("metadata =", document.metadata)
        print("text[:200] =", text[:200])

    return documents


def run_data_folder_demo():
    documents = load_data_folder_documents()
    if not documents:
        print("Data folder load failed")
        return

    index = build_index(documents)
    run_query(index, DATA_QUERY)


def run_file_query_evaluation():
    print("\n=== Block 6: File-by-File Query Evaluation ===")
    data_dir = resolve_data_dir()

    for file_name, test_cases in FILE_QUERY_TESTS.items():
        print(f"\n--- {file_name} ---")
        documents = load_documents_for_file(data_dir / file_name)
        print("len(documents) =", len(documents))

        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        for case_index, test_case in enumerate(test_cases, start=1):
            response = query_engine.query(test_case["question"])
            print(f"\nQ{case_index}: {test_case['question']}")
            print("Expected:", test_case["expected"])
            print("Actual:", shorten_text(str(response)))


def main():
    configure_settings()
    run_wikipedia_demo()
    run_data_folder_demo()
    run_file_query_evaluation()


if __name__ == "__main__":
    main()

import logging
from collections import Counter
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, SummaryIndex
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import (
    DocxReader,
    FlatReader,
    HWPReader,
    ImageReader,
    PDFReader,
    PandasExcelReader,
)
from llama_index.readers.wikipedia import WikipediaReader

PAGE_TITLES = ["Artificial intelligence"]
WIKI_QUERY = "What is this article mainly about?"
DATA_QUERY = "Summarize the main topics covered across all files in the data folder."
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".hwp", ".png"]


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

    Settings.llm = MockLLM(max_tokens=256)


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
    print("response =", str(response))
    if response.source_nodes:
        print("Top source chunk:", repr(response.source_nodes[0].node.get_content()))


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
        ".xlsx": PandasExcelReader(),
        ".hwp": HWPReader(),
        ".png": ImageReader(
            parse_text=True,
            text_type="plain_text",
            pytesseract_model_kwargs={"lang": "kor+eng"},
        ),
    }


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


def main():
    configure_settings()
    run_wikipedia_demo()
    run_data_folder_demo()


if __name__ == "__main__":
    main()

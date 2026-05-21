from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
)
if __package__:
    from .core import SUPPORTED_EXTENSIONS
    from .readers import build_file_extractor
else:
    from core import SUPPORTED_EXTENSIONS
    from readers import build_file_extractor

def load_data_folder_documents(data_dir="./data"):
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found.")
    
    reader = SimpleDirectoryReader(
        input_dir=str(path),
        file_extractor=build_file_extractor(),
        required_exts=SUPPORTED_EXTENSIONS
    )
    return reader.load_data()

def build_or_load_index(persist_dir="./storage", data_dir="./data"):
    storage_path = Path(persist_dir)
    if storage_path.exists() and any(storage_path.iterdir()):
        print(f"Loading existing index from {persist_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    
    print("Building new index...")
    documents = load_data_folder_documents(data_dir)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    return index

def shorten_text(text: str, limit: int = 200) -> str:
    return text[:limit] + "..." if len(text) > limit else text

# Singleton-like access to index
_global_index = None

def get_index():
    global _global_index
    if _global_index is None:
        _global_index = build_or_load_index()
    return _global_index

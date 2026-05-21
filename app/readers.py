import re
import pytesseract
from PIL import Image
from pathlib import Path
import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.file import (
    DocxReader, FlatReader, HWPReader, PDFReader
)

class SimpleImageReader(BaseReader):
    """A light-weight image reader using pytesseract directly to avoid heavy torch/transformers dependencies."""
    def load_data(self, file: Path, extra_info: dict = None, fs=None) -> list[Document]:
        try:
            img = Image.open(file)
            text = pytesseract.image_to_string(img, lang='kor+eng')
            metadata = extra_info or {"file_name": file.name}
            return [Document(text=text, metadata=metadata)]
        except Exception as e:
            print(f"Error processing image {file}: {e}")
            return []

class StructuredExcelReader(BaseReader):
    def load_data(self, file: Path, extra_info: dict = None, fs=None) -> list[Document]:
        df = pd.read_excel(fs.open(file) if fs else file, header=None).fillna("").map(lambda x: str(x).strip())
        return [Document(text=df.to_string(), metadata=extra_info or {"file_name": file.name})]

class CleanHWPReader(BaseReader):
    def __init__(self):
        self._reader = HWPReader()
    def load_data(self, file: Path, extra_info: dict = None, fs=None) -> list[Document]:
        docs = self._reader.load_data(file, extra_info=extra_info, fs=fs)
        cleaned_docs = []
        for d in docs:
            # Re-create Document to avoid 'no setter' error
            cleaned_text = re.sub(r"[^\w\s\.,!?;:]", " ", d.text)
            cleaned_docs.append(Document(text=cleaned_text, metadata=d.metadata))
        return cleaned_docs

def build_file_extractor():
    return {
        ".txt": FlatReader(),
        ".pdf": PDFReader(),
        ".docx": DocxReader(),
        ".xlsx": StructuredExcelReader(),
        ".hwp": CleanHWPReader(),
        ".png": SimpleImageReader(),
    }

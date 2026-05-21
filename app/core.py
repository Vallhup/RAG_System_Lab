import logging
import os
import re
from pathlib import Path
from llama_index.core import Settings
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.custom import CustomLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Constants
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".hwp", ".png"]
WORD_PATTERN = r"[A-Za-z\uac00-\ud7a3]{2,}"
PROMPT_NOISE_PHRASES = (
    "context information is below", "given the context information",
    "using both the context information", "we have provided an existing answer",
    "the original query is as follows", "query:", "answer the question",
)

class ExtractiveLLM(CustomLLM):
    max_tokens: int = 256
    model_name: str = "extractive-llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=4096, num_output=self.max_tokens, model_name=self.model_name, is_chat_model=False)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        return CompletionResponse(text=self._build_answer(prompt))

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        text = self._build_answer(prompt)
        def gen():
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
        return " ".join(ranked[:3]) if ranked else context[:1024].strip()

    def _extract_query(self, prompt: str) -> str:
        match = re.search(r"(?:Query|question|query)\s*[:=]\s*(.+)", prompt, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_context(self, prompt: str) -> str:
        match = re.search(r"Context information is below\.\s*-+\s*(.*?)\s*-+\s*", prompt, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else prompt.strip()

    def _split_candidates(self, context: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", context)
        return [p.strip() for p in parts if len(p) > 30 and not any(n in p.lower() for n in PROMPT_NOISE_PHRASES)]

    def _rank_candidates(self, candidates: list[str], query: str) -> list[str]:
        q_terms = set(re.findall(WORD_PATTERN, query.lower()))
        return sorted(candidates, key=lambda c: len(q_terms & set(re.findall(WORD_PATTERN, c.lower()))), reverse=True)

def configure_settings():
    cache_dir = Path(os.getenv("LLAMA_INDEX_CACHE_DIR", ".cache/llama_index"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LLAMA_INDEX_CACHE_DIR"] = str(cache_dir)

    # Use a lightweight local embedding model.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )
    Settings.llm = ExtractiveLLM()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

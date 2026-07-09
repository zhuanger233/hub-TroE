from __future__ import annotations

import hashlib
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


warnings.filterwarnings(
    "ignore",
    message=".*Torch was not compiled with flash attention.*",
    category=UserWarning,
)

DEFAULT_DOC_PATH = Path(r"D:\生活\学习\rag\doc.md")
DEFAULT_DB_DIR = Path(__file__).resolve().parent / "vector_db"
DEFAULT_COLLECTION_NAME = "novel_rag"
DEFAULT_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")
INDEX_VERSION = 1


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class SearchHit:
    text: str
    metadata: dict[str, object]
    distance: float | None


@dataclass(frozen=True)
class BuildStats:
    source_path: Path
    db_dir: Path
    collection_name: str
    chunk_count: int
    rebuilt: bool


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    hits: list[SearchHit]
    used_llm: bool


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 sentence-transformers，请先运行: pip install -r requirements.txt"
                ) from exc

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load_model()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [list(vector) for vector in vectors]


def read_text_file(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise RuntimeError(f"找不到语料文件: {path}")
    if not path.is_file():
        raise RuntimeError(f"语料路径不是文件: {path}")

    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    text = path.read_text(encoding="utf-8", errors="replace")
    if last_error is not None:
        print(f"警告: 自动编码识别失败，已用 utf-8(errors=replace) 读取: {last_error}")
    return text, "utf-8(errors=replace)"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_natural_break(text: str, start: int, hard_end: int, chunk_size: int) -> int:
    if hard_end >= len(text):
        return len(text)

    min_end = start + max(120, int(chunk_size * 0.55))
    min_end = min(min_end, hard_end)
    markers = ("\n\n", "\n", "。", "！", "？", "；", ";", ".", "!", "?")

    best = -1
    best_marker_len = 0
    for marker in markers:
        pos = text.rfind(marker, min_end, hard_end)
        if pos > best:
            best = pos
            best_marker_len = len(marker)

    if best != -1:
        return best + best_marker_len
    return hard_end


def _source_signature(path: Path, text: str) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_mtime_ns}|{len(text)}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def split_into_chunks(
    doc_path: Path,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise RuntimeError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise RuntimeError("overlap 必须大于等于 0，并且小于 chunk_size")

    text, encoding = read_text_file(doc_path)
    text = normalize_text(text)
    if not text:
        raise RuntimeError(f"语料文件为空: {doc_path}")

    signature = _source_signature(doc_path, text)
    chunks: list[TextChunk] = []
    start = 0
    source_path = str(doc_path.resolve())

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = _find_natural_break(text, start, hard_end, chunk_size)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk_index = len(chunks)
            chunks.append(
                TextChunk(
                    chunk_id=f"{signature}-{chunk_index:06d}",
                    text=chunk_text,
                    metadata={
                        "source_path": source_path,
                        "source_name": doc_path.name,
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end,
                        "encoding": encoding,
                    },
                )
            )

        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def batched(items: Sequence[TextChunk], batch_size: int) -> Iterable[list[TextChunk]]:
    for index in range(0, len(items), batch_size):
        yield list(items[index : index + batch_size])


def index_file_path(
    db_dir: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", collection_name).strip("._")
    if not safe_name:
        safe_name = DEFAULT_COLLECTION_NAME
    return db_dir / f"{safe_name}.pkl"


def save_index(db_dir: Path, collection_name: str, payload: dict[str, object]) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_file_path(db_dir, collection_name)
    temp_path = index_path.with_suffix(".tmp")
    with temp_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(index_path)
    return index_path


def load_index(db_dir: Path, collection_name: str) -> dict[str, object]:
    index_path = index_file_path(db_dir, collection_name)
    if not index_path.exists():
        raise RuntimeError("索引不存在，请先运行: python main.py build")
    with index_path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION:
        raise RuntimeError("索引文件版本不兼容，请重新运行: python main.py build")
    return payload


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    score = sum(a * b for a, b in zip(left, right))
    return 1.0 - score


def query_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for word in re.findall(r"[A-Za-z0-9_]{2,}", question.lower()):
        terms.add(word)

    for run in re.findall(r"[\u4e00-\u9fff]{2,}", question):
        max_len = min(4, len(run))
        for size in range(max_len, 1, -1):
            for index in range(0, len(run) - size + 1):
                terms.add(run[index : index + size])
    return terms


def lexical_score(question: str, text: str) -> float:
    terms = query_terms(question)
    if not terms:
        return 0.0

    lowered = text.lower()
    total_weight = 0
    matched_weight = 0
    for term in terms:
        weight = len(term)
        total_weight += weight
        if term.lower() in lowered:
            matched_weight += weight
    return matched_weight / total_weight if total_weight else 0.0


def build_index(
    doc_path: Path = DEFAULT_DOC_PATH,
    db_dir: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 900,
    overlap: int = 120,
    batch_size: int = 64,
    rebuild: bool = True,
) -> BuildStats:
    index_path = index_file_path(db_dir, collection_name)
    if index_path.exists() and not rebuild:
        payload = load_index(db_dir, collection_name)
        records = payload.get("records", [])
        return BuildStats(
            source_path=doc_path,
            db_dir=db_dir,
            collection_name=collection_name,
            chunk_count=len(records) if isinstance(records, list) else 0,
            rebuilt=False,
        )

    chunks = split_into_chunks(doc_path, chunk_size=chunk_size, overlap=overlap)
    embedder = EmbeddingService(embedding_model)
    total = len(chunks)
    written = 0
    records: list[dict[str, object]] = []

    for batch in batched(chunks, batch_size):
        embeddings = embedder.encode([chunk.text for chunk in batch])
        for chunk, embedding in zip(batch, embeddings):
            records.append(
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                }
            )
        written += len(batch)
        print(f"已写入索引: {written}/{total}")

    save_index(
        db_dir,
        collection_name,
        {
            "version": INDEX_VERSION,
            "embedding_model": embedding_model,
            "source_path": str(doc_path.resolve()),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "records": records,
        },
    )

    return BuildStats(
        source_path=doc_path,
        db_dir=db_dir,
        collection_name=collection_name,
        chunk_count=total,
        rebuilt=True,
    )


def search(
    question: str,
    db_dir: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
) -> list[SearchHit]:
    question = question.strip()
    if not question:
        raise RuntimeError("问题不能为空")
    if top_k <= 0:
        raise RuntimeError("top_k 必须大于 0")

    payload = load_index(db_dir, collection_name)
    indexed_model = payload.get("embedding_model")
    if indexed_model != embedding_model:
        raise RuntimeError(
            f"索引使用的向量模型是 {indexed_model}，当前参数是 {embedding_model}。"
            "请保持一致，或重新运行 python main.py build。"
        )

    records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError("索引为空，请先运行: python main.py build")

    embedder = EmbeddingService(embedding_model)
    query_embedding = embedder.encode([question])[0]
    ranked: list[tuple[float, float, dict[str, object]]] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        embedding = record.get("embedding")
        if not isinstance(embedding, list):
            continue
        distance = cosine_distance(query_embedding, embedding)
        keyword_score = lexical_score(question, str(record.get("text", "")))
        hybrid_score = (1.0 - distance) + (0.75 * keyword_score)
        ranked.append((hybrid_score, distance, record))

    ranked.sort(key=lambda item: item[0], reverse=True)
    hits: list[SearchHit] = []
    for _, distance, record in ranked[:top_k]:
        text = str(record.get("text", ""))
        metadata = record.get("metadata", {})
        hits.append(
            SearchHit(
                text=text,
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
                distance=distance,
            )
        )
    return hits


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path if env_path.exists() else None)


def build_prompt(question: str, hits: Sequence[SearchHit]) -> str:
    context = "\n\n".join(
        f"[{index}]\n{hit.text}" for index, hit in enumerate(hits, start=1)
    )
    return f"""你是一个严谨的中文 RAG 问答助手。请只依据下面给出的原文片段回答问题。
如果片段不足以支持答案，请明确说“根据当前片段无法确定”。
回答要简洁，并在关键事实后标注片段编号，例如 [1]。

问题：
{question}

原文片段：
{context}
"""


def generate_with_gemini(prompt: str, model_name: str = DEFAULT_LLM_MODEL) -> str | None:
    load_dotenv_if_available()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("缺少 google-genai，请先运行: pip install -r requirements.txt") from exc

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
    except Exception as exc:
        print(f"警告: Gemini 调用失败，已改用仅检索模式: {exc}", file=sys.stderr)
        return None

    text = getattr(response, "text", None)
    if text:
        return text.strip()
    return str(response).strip()


def answer_question(
    question: str,
    db_dir: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    top_k: int = 5,
    use_llm: bool = True,
) -> AnswerResult:
    hits = search(
        question=question,
        db_dir=db_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    if use_llm:
        prompt = build_prompt(question, hits)
        llm_answer = generate_with_gemini(prompt, model_name=llm_model)
        if llm_answer:
            return AnswerResult(answer=llm_answer, hits=hits, used_llm=True)

    top_excerpt = compact_text(hits[0].text, max_chars=320) if hits else "未检索到相关片段。"
    answer = (
        "未调用大模型生成自然语言答案，以下是最相关的原文摘录：\n"
        f"{top_excerpt}\n\n"
        "如需自动组织成答案，请在 .env 中配置有效的 GOOGLE_API_KEY 或 GEMINI_API_KEY。"
    )
    return AnswerResult(answer=answer, hits=hits, used_llm=False)


def compact_text(text: str, max_chars: int = 360) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def format_sources(hits: Sequence[SearchHit]) -> str:
    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        chunk_index = hit.metadata.get("chunk_index", "?")
        char_start = hit.metadata.get("char_start", "?")
        char_end = hit.metadata.get("char_end", "?")
        distance = "?" if hit.distance is None else f"{hit.distance:.4f}"
        lines.append(
            f"[{index}] chunk={chunk_index} chars={char_start}-{char_end} distance={distance}\n"
            f"{compact_text(hit.text)}"
        )
    return "\n\n".join(lines)

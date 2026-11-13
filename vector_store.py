import os
import re
import threading
import time
import uuid

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # must match EMBEDDING_MODEL
COLLECTION_PREFIX = "papeer_"
TITLE_KEY = "metadata.title"   # where langchain-qdrant stores Document.metadata["title"]
TITLES_CACHE_TTL = 300         # seconds; also cleared on every write/delete


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable {name}. Set it in your .env file.")
    return value


# ── Singletons ────────────────────────────────────────────────────────────────

base_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
embedding_file_store = LocalFileStore("./embedding_cache/")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

qdrant_client = QdrantClient(
    url=_require_env("QDRANT_URL"),
    api_key=_require_env("QDRANT_API_KEY"),
    timeout=120,
)

# In-memory caches (per process) so we don't hit the network on every call.
_lock = threading.Lock()
_ready: set[str] = set()                              # collections known to exist
_stores: dict[str, QdrantVectorStore] = {}            # collection -> vector store
_titles_cache: dict[str, tuple[float, list[str]]] = {}  # collection -> (timestamp, titles)


# ── Collection management ─────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", session_id)
    return f"{COLLECTION_PREFIX}{safe}"


def _ensure_collection(name: str) -> None:
    """Create the collection (and title index) if needed. Cheap after the first call."""
    if name in _ready:
        return
    with _lock:
        if name in _ready:
            return
        if not qdrant_client.collection_exists(name):
            try:
                qdrant_client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
            except Exception:
                # Another process may have created it between the check and now.
                if not qdrant_client.collection_exists(name):
                    raise
        try:
            qdrant_client.create_payload_index(
                collection_name=name,
                field_name=TITLE_KEY,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # index already exists, or not supported: filtering still works, just slower
        _ready.add(name)


def _collection_ready(name: str) -> bool:
    """True if the collection exists (without creating it)."""
    if name in _ready:
        return True
    if qdrant_client.collection_exists(name):
        _ensure_collection(name)
        return True
    return False


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    name = get_collection_name(session_id)
    _ensure_collection(name)
    store = _stores.get(name)
    if store is None:
        store = QdrantVectorStore(client=qdrant_client, collection_name=name, embedding=embeddings)
        _stores[name] = store
    return store


def _forget(name: str) -> None:
    _titles_cache.pop(name, None)


def _title_filter(title: str) -> Filter:
    return Filter(must=[FieldCondition(key=TITLE_KEY, match=MatchValue(value=title))])


def _doc_id(doc: Document) -> str:
    """Deterministic chunk ID: re-adding the same chunk overwrites instead of duplicating."""
    m = doc.metadata
    key = "|".join(
        str(m.get(k)) for k in ("title", "source", "url", "page", "start_index")
    ) + "|" + doc.page_content
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


# ── Public API ────────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> int:
    """Embed and store chunks. Idempotent: the same chunk is never stored twice."""
    if not docs:
        return 0
    store = get_vectorstore(session_id)
    store.add_documents(docs, ids=[_doc_id(d) for d in docs])
    _forget(get_collection_name(session_id))
    return len(docs)


def list_papers(session_id: str) -> list[str]:
    """Unique document titles in this session, in first-seen order (cached)."""
    name = get_collection_name(session_id)
    cached = _titles_cache.get(name)
    if cached and time.time() - cached[0] < TITLES_CACHE_TTL:
        return list(cached[1])
    if not _collection_ready(name):
        return []

    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=name,
            with_payload=[TITLE_KEY],  # fetch only the title, not the chunk text
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break

    _titles_cache[name] = (time.time(), titles)
    return list(titles)


def search(
    query: str,
    session_id: str,
    k: int = 4,
    mmr: bool = True,
    title: str | None = None,
) -> list[Document]:
    """Top-k chunks for `query`.

    mmr=True re-ranks for diversity (avoids near-duplicate overlapping chunks).
    title restricts the search to a single document.
    """
    name = get_collection_name(session_id)
    if not _collection_ready(name):
        return []
    store = get_vectorstore(session_id)
    flt = _title_filter(title) if title else None

    if mmr:
        try:
            return store.max_marginal_relevance_search(
                query, k=k, fetch_k=max(20, k * 4), lambda_mult=0.6, filter=flt
            )
        except Exception:
            pass  # fall back to plain similarity search
    return store.similarity_search(query, k=k, filter=flt)


def delete_paper(title: str, session_id: str) -> None:
    """Remove every chunk of one document from this session."""
    name = get_collection_name(session_id)
    if not _collection_ready(name):
        return
    qdrant_client.delete(collection_name=name, points_selector=_title_filter(title))
    _forget(name)


def delete_session(session_id: str) -> None:
    """Drop the whole collection for a session (call when a chat is deleted)."""
    name = get_collection_name(session_id)
    with _lock:
        if qdrant_client.collection_exists(name):
            qdrant_client.delete_collection(name)
        _ready.discard(name)
        _stores.pop(name, None)
        _forget(name)

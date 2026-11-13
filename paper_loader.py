import ipaddress
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:  # PyMuPDF >= 1.24 exposes `pymupdf`; older versions only `fitz`
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf

# ── Config ────────────────────────────────────────────────────────────────────

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 50                 # drop tiny, noisy chunks
MAX_PDF_BYTES = 50_000_000           # 50 MB download cap
USER_AGENT = "Mozilla/5.0 (compatible; Papeer/1.0; research-paper-assistant)"
ALLOW_PRIVATE_URLS = os.getenv("PAPEER_ALLOW_PRIVATE_URLS") == "1"  # for local development only

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)

# ── ArXiv identifiers ─────────────────────────────────────────────────────────

_NEW_ID = r"\d{4}\.\d{4,5}"                       # 1706.03762
_OLD_ID = r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}"        # hep-th/9901001
_ARXIV_ID_RE = re.compile(rf"^(?:arxiv:\s*)?({_NEW_ID}|{_OLD_ID})(?:v\d+)?$", re.I)
_ARXIV_PATH_RE = re.compile(rf"/(?:abs|pdf)/({_NEW_ID}|{_OLD_ID})(?:v\d+)?", re.I)
_ATOM = {"a": "http://www.w3.org/2005/Atom"}

# ── Text cleaning ─────────────────────────────────────────────────────────────

_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00ad": "",    # soft hyphen
    "\u00a0": " ",   # non-breaking space
})
# Join words split by line-end hyphenation (exam-\nple -> example), but keep real compounds
# ("state-of-the-\nart"): the fragment must not itself be preceded by a hyphen.
_HYPHEN_BREAK = re.compile(r"(?<![\w-])([A-Za-z]+)-\n([a-z])")
_COMPOUND_BREAK = re.compile(r"(\w)-\n([a-z])")             # end-to-\nend -> end-to-end
_SOFT_WRAP = re.compile(r"([^\n.!?:;])\n([a-z])")           # mid-sentence line wrap
_PAGE_NUM_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.I)


def _clean_text(text: str, *, pdf: bool = False) -> str:
    text = text.translate(_LIGATURES)
    if pdf:
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
        text = _COMPOUND_BREAK.sub(r"\1-\2", text)
        text = _SOFT_WRAP.sub(r"\1 \2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _drop_edge_page_numbers(text: str) -> str:
    """Remove a page-number-only line at the very top or bottom of a page."""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and _PAGE_NUM_LINE.match(lines[-1]):
        lines.pop()
    if lines and _PAGE_NUM_LINE.match(lines[0]):
        lines.pop(0)
    return "\n".join(lines)


def _strip_repeated_lines(pages: list[Document]) -> None:
    """Remove running headers/footers: short lines near page edges that repeat on >= 50% of pages."""
    if len(pages) < 4:
        return

    def norm(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip())

    def edge_lines(text: str) -> list[str]:
        lines = [l for l in text.splitlines() if l.strip()]
        return lines[:3] + lines[-3:]

    counts: Counter[str] = Counter()
    for p in pages:
        for n in {norm(l) for l in edge_lines(p.page_content) if 3 <= len(l.strip()) <= 100}:
            counts[n] += 1
    threshold = max(3, len(pages) // 2)
    repeated = {n for n, c in counts.items() if c >= threshold}
    if not repeated:
        return

    for p in pages:
        edges = set(edge_lines(p.page_content))
        p.page_content = "\n".join(
            line for line in p.page_content.splitlines()
            if not (line in edges and norm(line) in repeated)
        )


# ── Networking helpers ────────────────────────────────────────────────────────

def _validate_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts that resolve to private/local addresses (SSRF guard).

    Note: redirects are not re-checked.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid URL: {url}")
    if ALLOW_PRIVATE_URLS:
        return
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host: {parsed.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("URLs pointing to private or local network addresses are not allowed.")


def _http_get(url: str, *, timeout: int, max_bytes: int, retries: int = 2) -> bytes:
    """GET with a size cap and retries on transient errors (5xx, 429, timeouts)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read(max_bytes + 1)
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise ValueError(f"Download failed (HTTP {e.code}): {url}") from e
            last_error = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
        else:
            if len(data) > max_bytes:
                raise ValueError(f"File is larger than {max_bytes // 1_000_000} MB: {url}")
            return data
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise ValueError(f"Could not download {url}: {last_error}")


# ── Chunk post-processing ─────────────────────────────────────────────────────

def _finalize(
    chunks: list[Document],
    *,
    title: str,
    source: str,
    source_type: str,
    url: str | None = None,
    extra: dict | None = None,
) -> list[Document]:
    """Keep only useful metadata and drop tiny chunks."""
    out: list[Document] = []
    for c in chunks:
        text = c.page_content.strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        md = {k: c.metadata[k] for k in ("page", "total_pages", "start_index") if k in c.metadata}
        md.update(title=title, source=source, source_type=source_type)
        if url:
            md["url"] = url
        if extra:
            md.update(extra)
        out.append(Document(page_content=text, metadata=md))
    if not out:
        raise ValueError("No readable text could be extracted from this source.")
    return out


# ── PDF ───────────────────────────────────────────────────────────────────────

def _pdf_pages(*, path: str | None = None, data: bytes | None = None) -> list[Document]:
    pdf = pymupdf.open(path) if path else pymupdf.open(stream=data, filetype="pdf")
    try:
        if pdf.needs_pass:
            raise ValueError("This PDF is password protected.")
        total = len(pdf)
        return [
            Document(page_content=page.get_text(), metadata={"page": i, "total_pages": total})
            for i, page in enumerate(pdf)
        ]
    finally:
        pdf.close()


def _pdf_docs(
    pages: list[Document],
    *,
    title: str,
    source: str,
    source_type: str,
    url: str | None = None,
    extra: dict | None = None,
) -> list[Document]:
    _strip_repeated_lines(pages)
    for p in pages:
        p.page_content = _clean_text(_drop_edge_page_numbers(p.page_content), pdf=True)
    pages = [p for p in pages if p.page_content]
    if not pages:
        raise ValueError("No extractable text found. This looks like a scanned PDF (OCR is required).")
    chunks = _splitter.split_documents(pages)
    return _finalize(chunks, title=title, source=source, source_type=source_type, url=url, extra=extra)


def load_pdf(file_path: str, title: str | None = None) -> list[Document]:
    name = title or Path(file_path).stem
    return _pdf_docs(_pdf_pages(path=file_path), title=name, source=name, source_type="pdf")


# ── Plain text & Markdown ─────────────────────────────────────────────────────

def _read_text_file(file_path: str) -> str:
    raw = Path(file_path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _load_plain(file_path: str, splitter: RecursiveCharacterTextSplitter,
                title: str | None, source_type: str) -> list[Document]:
    name = title or Path(file_path).stem
    text = _clean_text(_read_text_file(file_path))
    chunks = splitter.split_documents([Document(page_content=text)])
    return _finalize(chunks, title=name, source=name, source_type=source_type)


def load_text(file_path: str, title: str | None = None) -> list[Document]:
    return _load_plain(file_path, _splitter, title, "text")


def load_markdown(file_path: str, title: str | None = None) -> list[Document]:
    return _load_plain(file_path, _md_splitter, title, "markdown")


# ── ArXiv ─────────────────────────────────────────────────────────────────────

def _extract_arxiv_id(query: str) -> str | None:
    """Return a bare ArXiv ID (no version) if `query` is an ID, 'arxiv:ID', or an arxiv.org URL."""
    q = query.strip()
    if q.lower().startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(q)
        host = (parsed.hostname or "").lower()
        if host == "arxiv.org" or host.endswith(".arxiv.org"):
            m = _ARXIV_PATH_RE.search(parsed.path)
            return m.group(1) if m else None
        return None
    m = _ARXIV_ID_RE.match(q)
    return m.group(1) if m else None


def _parse_arxiv_entries(xml: bytes) -> list[tuple[str, str]]:
    """Parse an ArXiv Atom feed into [(bare_id, title), ...]."""
    entries = []
    for entry in ET.fromstring(xml).findall("a:entry", _ATOM):
        entry_id = entry.findtext("a:id", "", _ATOM)
        if "/api/errors" in entry_id:
            continue
        m = re.search(r"/abs/(.+?)(?:v\d+)?$", entry_id.strip())
        if not m:
            continue
        title = " ".join(entry.findtext("a:title", "", _ATOM).split())
        entries.append((m.group(1), title))
    return entries


def _arxiv_query(params: dict) -> list[tuple[str, str]]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    return _parse_arxiv_entries(_http_get(url, timeout=15, max_bytes=2_000_000))


def _arxiv_search(query: str) -> tuple[str, str]:
    """Find a paper by title. Tries an exact title phrase first, then a keyword search."""
    phrase = query.strip().strip('"')
    words = re.findall(r"\w+", phrase)[:12]
    attempts = [f'ti:"{phrase}"']
    if words:
        attempts.append(" AND ".join(f"all:{w}" for w in words))
    for search_query in attempts:
        results = _arxiv_query({"search_query": search_query, "max_results": 1, "sortBy": "relevance"})
        if results:
            return results[0]
    raise ValueError(f"No ArXiv paper found for: {query}")


def _load_arxiv_by_id(arxiv_id: str, title: str | None = None) -> list[Document]:
    """Download, clean and chunk an ArXiv paper by its bare ID."""
    if title is None:
        try:
            found = _arxiv_query({"id_list": arxiv_id})
            title = found[0][1] if found else None
        except Exception:
            title = None
    pdf_bytes = _http_get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=60, max_bytes=MAX_PDF_BYTES)
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    return _pdf_docs(
        _pdf_pages(data=pdf_bytes),
        title=title or f"arXiv:{arxiv_id}",
        source=abs_url,
        source_type="arxiv",
        url=abs_url,
        extra={"arxiv_id": arxiv_id},
    )


def load_arxiv(query: str) -> list[Document]:
    """Load an ArXiv paper from an ID ('1706.03762'), an arxiv.org URL, or a title."""
    arxiv_id = _extract_arxiv_id(query)
    if arxiv_id:
        return _load_arxiv_by_id(arxiv_id)
    arxiv_id, title = _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id, title)


# ── Web pages ─────────────────────────────────────────────────────────────────

def load_webpage(url: str, title: str | None = None) -> list[Document]:
    """Load a URL. arXiv links and direct .pdf links are detected and handled as papers."""
    url = url.strip()
    _validate_url(url)

    arxiv_id = _extract_arxiv_id(url)
    if arxiv_id:
        return _load_arxiv_by_id(arxiv_id)

    path = urllib.parse.urlparse(url).path
    if path.lower().endswith(".pdf"):
        data = _http_get(url, timeout=60, max_bytes=MAX_PDF_BYTES)
        name = title or Path(urllib.parse.unquote(path)).stem or url
        return _pdf_docs(_pdf_pages(data=data), title=name, source=url, source_type="web_pdf", url=url)

    from langchain_community.document_loaders import WebBaseLoader  # lazy: only needed here

    docs = WebBaseLoader(
        url,
        requests_kwargs={"timeout": 30},
        header_template={"User-Agent": USER_AGENT},
        bs_get_text_kwargs={"separator": "\n", "strip": True},
    ).load()
    text = _clean_text(docs[0].page_content) if docs else ""
    if not text:
        raise ValueError(f"No readable text found at {url}")
    page_title = title or (docs[0].metadata.get("title") or "").strip() or url
    chunks = _splitter.split_documents([Document(page_content=text)])
    return _finalize(chunks, title=page_title, source=url, source_type="web", url=url)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def load_document(source: str, title: str | None = None) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source, title)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source, title)
    if ext == ".txt":
        return load_text(source, title)
    if ext in (".md", ".markdown"):
        return load_markdown(source, title)
    raise ValueError(f"Unsupported file type: {ext}")

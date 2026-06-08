
import os
import re
import json
import sqlite3
import logging
import hashlib
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("research_copilot")

@dataclass
class AppConfig:
    """Centralised configuration with sensible defaults."""

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "all-MiniLM-L6-v2"
    qdrant_path: str = ""
    sqlite_path: str = ""
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    def __post_init__(self) -> None:
        base = Path(tempfile.gettempdir()) / "research_copilot"
        base.mkdir(parents=True, exist_ok=True)
        if not self.qdrant_path:
            self.qdrant_path = str(base / "qdrant_db")
        if not self.sqlite_path:
            self.sqlite_path = str(base / "metadata.db")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")


@st.cache_resource
def get_config() -> AppConfig:
    """Return a singleton AppConfig (cached across Streamlit reruns)."""
    cfg = AppConfig()
    if not cfg.groq_api_key:
        logger.warning("GROQ_API_KEY is not set — LLM features will be unavailable.")
    return cfg



class MetadataDB:
    """Lightweight SQLite wrapper for paper and chunk metadata."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_schema()

    # -- internal helpers ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id   TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    authors    TEXT DEFAULT '',
                    abstract   TEXT DEFAULT '',
                    arxiv_id   TEXT DEFAULT '',
                    pdf_url    TEXT DEFAULT '',
                    filepath   TEXT DEFAULT '',
                    ingested_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id    TEXT PRIMARY KEY,
                    paper_id    TEXT NOT NULL,
                    title       TEXT DEFAULT '',
                    section     TEXT DEFAULT '',
                    page_number INTEGER DEFAULT 0,
                    text        TEXT NOT NULL,
                    char_start  INTEGER DEFAULT 0,
                    char_end    INTEGER DEFAULT 0,
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_paper
                    ON chunks(paper_id);
                """
            )
        logger.info("SQLite schema initialised at %s", self.db_path)

    # -- public API ---------------------------------------------------------

    def upsert_paper(
        self,
        paper_id: str,
        title: str,
        authors: str = "",
        abstract: str = "",
        arxiv_id: str = "",
        pdf_url: str = "",
        filepath: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (paper_id, title, authors, abstract, arxiv_id, pdf_url, filepath)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title=excluded.title, authors=excluded.authors,
                    abstract=excluded.abstract, pdf_url=excluded.pdf_url,
                    filepath=excluded.filepath
                """,
                (paper_id, title, authors, abstract, arxiv_id, pdf_url, filepath),
            )

    def insert_chunk(
        self,
        chunk_id: str,
        paper_id: str,
        title: str,
        section: str,
        page_number: int,
        text: str,
        char_start: int,
        char_end: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO chunks
                    (chunk_id, paper_id, title, section, page_number, text, char_start, char_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, paper_id, title, section, page_number, text, char_start, char_end),
            )

    def get_paper(self, paper_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
            return dict(row) if row else None

    def list_papers(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM papers ORDER BY ingested_at DESC").fetchall()
            return [dict(r) for r in rows]

    def get_chunks_by_paper(self, paper_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE paper_id=? ORDER BY page_number, char_start",
                (paper_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_paper(self, paper_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
            conn.execute("DELETE FROM papers WHERE paper_id=?", (paper_id,))

    def chunk_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def paper_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]


@st.cache_resource
def get_db() -> MetadataDB:
    return MetadataDB(get_config().sqlite_path)



class VectorStore:
    """Thin wrapper around Qdrant local client for dense vector storage."""

    COLLECTION = "research_chunks"
    DEMO_COLLECTIONS = {
        1: "demo_paper_collection_1",
        2: "demo_paper_collection_2",
    }
    DIMENSION = 384  # all-MiniLM-L6-v2

    def __init__(self, qdrant_path: str) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(path=qdrant_path)
        self._ensure_collection()
        logger.info("Qdrant vector store ready at %s", qdrant_path)

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION not in existing:
            self.client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=self.DIMENSION, distance=Distance.COSINE),
            )

    def _ensure_demo_collection(self, paper_num: int) -> None:
        """Create the dedicated demo_paper_collection_N if it does not exist."""
        from qdrant_client.models import Distance, VectorParams

        collection_name = self.DEMO_COLLECTIONS[paper_num]
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.DIMENSION, distance=Distance.COSINE),
            )

    def has_demo_collection(self, paper_num: int) -> bool:
        """Check whether demo_paper_collection_N exists."""
        collection_name = self.DEMO_COLLECTIONS[paper_num]
        existing = [c.name for c in self.client.get_collections().collections]
        return collection_name in existing

    def demo_collection_point_count(self, paper_num: int) -> int:
        """Return the number of points in a demo collection."""
        try:
            info = self.client.get_collection(self.DEMO_COLLECTIONS[paper_num])
            return info.points_count
        except Exception:
            return 0

    def upsert(self, chunk_id: str, vector: list[float], payload: dict) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self.COLLECTION,
            points=[
                PointStruct(
                    id=hashlib.md5(chunk_id.encode()).hexdigest()[:16],
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def batch_upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
        collection_name: Optional[str] = None,
    ) -> None:

        import uuid
        from qdrant_client.models import PointStruct

        collection = collection_name or self.COLLECTION

        points = [
            PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        i
                    )
                ),
                vector=v,
                payload=p,
            )
            for i, v, p in zip(ids, vectors, payloads)
        ]

        self.client.upsert(
            collection_name=collection,
            points=points
        )

    def restore_demo_collection_from_snapshot(
        self, paper_num: int, snapshot_path: Path
    ) -> bool:
        """Restore a demo collection from a pre-indexed snapshot JSON file.

        The snapshot file contains precomputed vectors and payloads so that
        demo papers NEVER require re-embedding after restarts.

        Returns True on success, False on failure.
        """
        import uuid
        from qdrant_client.models import PointStruct

        collection_name = self.DEMO_COLLECTIONS[paper_num]

        if not snapshot_path.exists():
            logger.error("Snapshot file not found: %s", snapshot_path)
            return False

        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snapshot_data = json.load(f)
        except Exception as exc:
            logger.error("Failed to read snapshot %s: %s", snapshot_path, exc)
            return False

        points_data = snapshot_data.get("points", [])
        if not points_data:
            logger.warning("Snapshot %s contains no points", snapshot_path)
            return False

        # Ensure the collection exists
        self._ensure_demo_collection(paper_num)

        # Batch upsert from snapshot — no embedding model needed
        batch_size = 100
        for i in range(0, len(points_data), batch_size):
            batch = points_data[i : i + batch_size]
            points = [
                PointStruct(
                    id=pt["id"],
                    vector=pt["vector"],
                    payload=pt["payload"],
                )
                for pt in batch
            ]
            self.client.upsert(collection_name=collection_name, points=points)

        logger.info(
            "Restored demo collection '%s' from snapshot — %d points loaded (no re-embedding)",
            collection_name,
            len(points_data),
        )
        return True

    def search(self, query_vector: list[float], limit: int = 10, collection_name: Optional[str] = None) -> list[dict]:
        collection = collection_name or self.COLLECTION
        hits = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
        )
        return [
            {
                "chunk_id": h.payload.get("chunk_id", ""),
                "score": h.score,
                "paper_id": h.payload.get("paper_id", ""),
                "title": h.payload.get("title", ""),
                "section": h.payload.get("section", ""),
                "page_number": h.payload.get("page_number", 0),
                "text": h.payload.get("text", ""),
            }
            for h in hits
        ]

    def search_all(self, query_vector: list[float], limit: int = 10) -> list[dict]:
        """Search main + all demo collections, merge and deduplicate by score."""
        results = self.search(query_vector, limit=limit)
        seen = {r.get("chunk_id", "") for r in results}
        for paper_num, coll_name in self.DEMO_COLLECTIONS.items():
            existing = [c.name for c in self.client.get_collections().collections]
            if coll_name not in existing:
                continue
            demo_results = self.search(query_vector, limit=limit, collection_name=coll_name)
            for r in demo_results:
                if r.get("chunk_id", "") not in seen:
                    results.append(r)
                    seen.add(r.get("chunk_id", ""))
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:limit]

    def delete_by_paper(self, paper_id: str) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.client.delete(
            collection_name=self.COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
            ),
        )


@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore(get_config().qdrant_path)



import urllib.request
import urllib.error


def download_arxiv_pdf(arxiv_id: str, dest_dir: str) -> Optional[str]:
    """Download a PDF from arXiv given an ID."""

    import ssl
    import certifi

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    arxiv_id = arxiv_id.strip().split("/")[-1]
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    dest_path = os.path.join(
        dest_dir,
        f"{arxiv_id.replace('/', '_')}.pdf"
    )

    if os.path.exists(dest_path):
        logger.info("PDF already cached: %s", dest_path)
        return dest_path

    try:
        logger.info("Downloading arXiv PDF: %s", url)

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchCopilot/1.0"}
        )

        with urllib.request.urlopen(
            req,
            timeout=60,
            context=ssl_context
        ) as resp:

            with open(dest_path, "wb") as f:
                f.write(resp.read())

        logger.info("Downloaded → %s", dest_path)
        return dest_path

    except (urllib.error.URLError,
            urllib.error.HTTPError,
            OSError) as exc:

        logger.error(
            "arXiv download failed for %s: %s",
            arxiv_id,
            exc
        )

        return None


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """Fetch title, authors, abstract from the arXiv API."""

    import xml.etree.ElementTree as ET
    import ssl
    import certifi

    ssl_context = ssl.create_default_context(
        cafile=certifi.where()
    )

    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchCopilot/1.0"}
        )

        with urllib.request.urlopen(
            req,
            timeout=30,
            context=ssl_context
        ) as resp:

            root = ET.fromstring(resp.read())

        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entry = root.find("atom:entry", ns)

        if entry is None:
            return {}

        title = (
            entry.findtext("atom:title", "", ns)
            or ""
        ).strip().replace("\n", " ")

        authors = ", ".join(
            a.findtext("atom:name", "", ns)
            for a in entry.findall("atom:author", ns)
        )

        abstract = (
            entry.findtext("atom:summary", "", ns)
            or ""
        ).strip().replace("\n", " ")

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract
        }

    except Exception as exc:
        logger.warning(
            "arXiv metadata fetch failed: %s",
            exc
        )

        return {}

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF using PyMuPDF.

    Returns a list of dicts: [{page_number, text}, ...]
    """
    import fitz  # PyMuPDF — lazy import to reduce startup time

    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            if text.strip():
                pages.append({"page_number": page_num + 1, "text": text})
        doc.close()
    except Exception as exc:
        logger.error("PDF extraction error: %s", exc)
    return pages



# Common section headers in CS / ML papers
_SECTION_PATTERNS = [
    r"(?i)^(abstract|introduction|background|related work|method|methodology|"
    r"approach|experiment|experiments|results|evaluation|discussion|"
    r"conclusion|conclusions|references|acknowledgements|appendix)\b"
]


def _detect_section(line: str, current: str) -> str:
    """Return a new section name if *line* looks like a heading, else *current*."""
    for pat in _SECTION_PATTERNS:
        if re.match(pat, line.strip()):
            return line.strip().title()
    return current


def smart_chunk(
    pages: list[dict],
    paper_id: str,
    title: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """Section-aware, overlapping chunking that preserves citations.

    Returns list of chunk dicts ready for storage.
    """
    chunks: list[dict] = []
    current_section = "Unknown"
    chunk_counter = 0

    for page in pages:
        page_num = page["page_number"]
        lines = page["text"].split("\n")
        page_text_parts: list[str] = []
        char_offset = 0

        for line in lines:
            candidate = _detect_section(line, current_section)
            if candidate != current_section and page_text_parts:
                # Flush current buffer as a chunk
                _flush_chunk(
                    chunks, page_text_parts, paper_id, title,
                    current_section, page_num, chunk_counter, char_offset,
                )
                chunk_counter += 1
                char_offset += sum(len(l) + 1 for l in page_text_parts)
                page_text_parts = []
                current_section = candidate
                continue
            if candidate != current_section:
                current_section = candidate
            page_text_parts.append(line)

        if page_text_parts:
            _flush_chunk(
                chunks, page_text_parts, paper_id, title,
                current_section, page_num, chunk_counter, char_offset,
            )
            chunk_counter += 1

    # ---- second pass: split oversized chunks with overlap ----
    final_chunks: list[dict] = []
    for ch in chunks:
        text = ch["text"]
        if len(text) <= chunk_size:
            final_chunks.append(ch)
        else:
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                sub_text = " ".join(words[start:end])
                # Preserve citation-like tokens at boundaries
                if end < len(words) and re.search(r"\[\d+\]$", sub_text) is None:
                    # try to extend to next citation
                    lookahead = " ".join(words[end : end + 5])
                    cite_match = re.search(r"\[(\d+)\]", lookahead)
                    if cite_match:
                        extra = lookahead[: cite_match.end()]
                        sub_text += " " + extra
                sub_id = f"{ch['paper_id']}_{ch['chunk_id']}_{start}"
                final_chunks.append(
                    {
                        "chunk_id": sub_id,
                        "paper_id": ch["paper_id"],
                        "title": ch["title"],
                        "section": ch["section"],
                        "page_number": ch["page_number"],
                        "text": sub_text,
                        "char_start": start,
                        "char_end": end,
                    }
                )
                start += chunk_size - chunk_overlap

    logger.info("Smart-chunked into %d final chunks", len(final_chunks))
    return final_chunks


def _flush_chunk(
    target: list,
    parts: list[str],
    paper_id: str,
    title: str,
    section: str,
    page_num: int,
    counter: int,
    offset: int,
) -> None:
    text = "\n".join(parts).strip()
    if not text:
        return
    target.append(
        {
            "chunk_id": f"{paper_id}_c{counter}",
            "paper_id": paper_id,
            "title": title,
            "section": section,
            "page_number": page_num,
            "text": text,
            "char_start": offset,
            "char_end": offset + len(text),
        }
    )



@st.cache_resource
def get_embedding_model():
    """Lazy-load and cache the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    cfg = get_config()
    model = SentenceTransformer(cfg.embedding_model)
    logger.info("Loaded embedding model: %s", cfg.embedding_model)
    return model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Generate embeddings with batching to keep memory low."""
    model = get_embedding_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()



class HybridRetriever:
    """BM25 + Dense retrieval with Reciprocal Rank Fusion."""

    def __init__(self) -> None:
        self.bm25_corpus: list[str] = []
        self.bm25_ids: list[str] = []
        self._bm25 = None

    # -- BM25 indexing -------------------------------------------------------

    def index_bm25(self, texts: list[str], chunk_ids: list[str]) -> None:
        """(Re)build the BM25 index.  Called after new chunks are added."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 retrieval disabled.")
            return

        self.bm25_corpus.extend(texts)
        self.bm25_ids.extend(chunk_ids)
        tokenized = [t.lower().split() for t in self.bm25_corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index rebuilt with %d docs", len(self.bm25_corpus))

    def bm25_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return BM25 results as [{chunk_id, score}]."""
        if self._bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"chunk_id": self.bm25_ids[i], "score": float(s)} for i, s in ranked if s > 0]

    # -- Dense search -------------------------------------------------------

    def dense_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search Qdrant (main + demo collections) with the query embedding."""
        q_vec = embed_texts([query])[0]
        vs = get_vector_store()
        return vs.search_all(q_vec, limit=top_k)

    # -- Reciprocal Rank Fusion ---------------------------------------------

    @staticmethod
    def rrf(
        bm25_results: list[dict],
        dense_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        Each item must have 'chunk_id' and 'score' keys.
        Returns items sorted by fused score, each with 'rrf_score'.
        """
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}

        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            meta.setdefault(cid, item)

        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            meta.setdefault(cid, item)
            # carry forward richer dense payload
            if "text" in item:
                meta[cid] = item

        fused = []
        for cid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            entry = dict(meta.get(cid, {}))
            entry["chunk_id"] = cid
            entry["rrf_score"] = rrf_score
            fused.append(entry)
        return fused

    # -- Full hybrid search --------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Run hybrid search and return the top-k fused results."""
        cfg = get_config()
        bm25_hits = self.bm25_search(query, top_k=top_k * 3)
        dense_hits = self.dense_search(query, top_k=top_k * 3)
        fused = self.rrf(bm25_hits, dense_hits, k=cfg.rrf_k)
        return fused[:top_k]


@st.cache_resource
def get_retriever() -> HybridRetriever:
    retriever = HybridRetriever()
    # Re-hydrate BM25 from SQLite chunks on startup
    db = get_db()
    papers = db.list_papers()
    all_texts, all_ids = [], []
    for p in papers:
        for ch in db.get_chunks_by_paper(p["paper_id"]):
            all_texts.append(ch["text"])
            all_ids.append(ch["chunk_id"])
    if all_texts:
        retriever.index_bm25(all_texts, all_ids)
    return retriever



def call_groq(system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """Call Groq chat completions. Returns the assistant message text."""
    cfg = get_config()
    if not cfg.groq_api_key:
        return "⚠️ GROQ_API_KEY is not configured. Please set it in .env or Streamlit secrets."

    try:
        from groq import Groq

        client = Groq(api_key=cfg.groq_api_key)
        resp = client.chat.completions.create(
            model=cfg.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Groq API call failed: %s", exc)
        return f"❌ Groq API error: {exc}"



def ingest_arxiv_paper(arxiv_id: str) -> Optional[str]:
    """End-to-end: download arXiv PDF → extract → chunk → embed → store.

    Returns the paper_id on success, None on failure.
    """
    cfg = get_config()
    db = get_db()
    vs = get_vector_store()

    # 1. Download
    pdf_dir = os.path.join(tempfile.gettempdir(), "research_copilot", "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    with st.spinner("Downloading PDF from arXiv..."):
        pdf_path = download_arxiv_pdf(arxiv_id, pdf_dir)
    if not pdf_path:
        return None

    # 2. Fetch metadata
    with st.spinner("Fetching metadata from arXiv API..."):
        meta = fetch_arxiv_metadata(arxiv_id)

    paper_id = hashlib.md5(arxiv_id.encode()).hexdigest()[:12]
    title = meta.get("title", arxiv_id)
    authors = meta.get("authors", "")
    abstract = meta.get("abstract", "")

    # Skip if already ingested
    if db.get_paper(paper_id):
        st.warning(f"Paper **{title}** is already in the database.")
        return paper_id

    # 3. Extract text
    with st.spinner("Extracting text from PDF..."):
        pages = extract_text_from_pdf(pdf_path)
    if not pages:
        st.error("Could not extract text from the PDF.")
        return None

    # 4. Smart chunking
    with st.spinner("Smart-chunking the paper..."):
        chunks = smart_chunk(pages, paper_id, title, cfg.chunk_size, cfg.chunk_overlap)

    # 5. Store metadata
    db.upsert_paper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        abstract=abstract,
        arxiv_id=arxiv_id,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        filepath=pdf_path,
    )

    # 6. Embed and store in Qdrant + SQLite
    with st.spinner(f"Embedding {len(chunks)} chunks (this may take a moment)..."):
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)
        payloads = [
            {
                "chunk_id": c["chunk_id"],
                "paper_id": c["paper_id"],
                "title": c["title"],
                "section": c["section"],
                "page_number": c["page_number"],
                "text": c["text"],
            }
            for c in chunks
        ]
        vs.batch_upsert(
            ids=[c["chunk_id"] for c in chunks],
            vectors=vectors,
            payloads=payloads,
        )

    for c in chunks:
        db.insert_chunk(
            chunk_id=c["chunk_id"],
            paper_id=c["paper_id"],
            title=c["title"],
            section=c["section"],
            page_number=c["page_number"],
            text=c["text"],
            char_start=c["char_start"],
            char_end=c["char_end"],
        )

    # 7. Re-index BM25
    retriever = get_retriever()
    retriever.index_bm25(texts, [c["chunk_id"] for c in chunks])

    st.success(f"Ingested **{title}** — {len(chunks)} chunks indexed.")
    return paper_id


def ingest_uploaded_pdf(uploaded_file, paper_title: str = "") -> Optional[str]:
    """Ingest a user-uploaded PDF through the same pipeline."""
    cfg = get_config()
    db = get_db()
    vs = get_vector_store()

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded_file.read())
    tmp.close()

    title = paper_title.strip() or uploaded_file.name
    paper_id = hashlib.md5(title.encode()).hexdigest()[:12]

    if db.get_paper(paper_id):
        st.warning(f"Paper **{title}** is already in the database.")
        return paper_id

    with st.spinner("Extracting text from PDF..."):
        pages = extract_text_from_pdf(tmp.name)
    if not pages:
        st.error("Could not extract text from the PDF.")
        return None

    with st.spinner("Smart-chunking the paper..."):
        chunks = smart_chunk(pages, paper_id, title, cfg.chunk_size, cfg.chunk_overlap)

    db.upsert_paper(paper_id=paper_id, title=title, filepath=tmp.name)

    with st.spinner(f"Embedding {len(chunks)} chunks..."):
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)
        payloads = [
            {
                "chunk_id": c["chunk_id"],
                "paper_id": c["paper_id"],
                "title": c["title"],
                "section": c["section"],
                "page_number": c["page_number"],
                "text": c["text"],
            }
            for c in chunks
        ]
        vs.batch_upsert(ids=[c["chunk_id"] for c in chunks], vectors=vectors, payloads=payloads)

    for c in chunks:
        db.insert_chunk(
            chunk_id=c["chunk_id"],
            paper_id=c["paper_id"],
            title=c["title"],
            section=c["section"],
            page_number=c["page_number"],
            text=c["text"],
            char_start=c["char_start"],
            char_end=c["char_end"],
        )

    retriever = get_retriever()
    retriever.index_bm25(texts, [c["chunk_id"] for c in chunks])

    st.success(f"Ingested **{title}** — {len(chunks)} chunks indexed.")
    return paper_id



# Per-demo-paper configuration — deterministic IDs, never conflict with user uploads
_DEMO_PAPER_CONFIG = {
    1: {
        "paper_id": hashlib.md5("__demo_paper_1_builtin__".encode()).hexdigest()[:12],
        "title": "Demo Research Paper 1",
        "pdf_file": "demo_paper_1.pdf",
        "metadata_file": "demo_paper_1_metadata.json",
        "snapshot_file": "demo_paper_1_collection_snapshot.json",
        "not_found_msg": "Demo Paper 1 file not found.",
        "success_msg": "Demo Paper 1 loaded successfully.",
    },
    2: {
        "paper_id": hashlib.md5("__demo_paper_2_builtin__".encode()).hexdigest()[:12],
        "title": "Demo Research Paper 2",
        "pdf_file": "demo_paper_2.pdf",
        "metadata_file": "demo_paper_2_metadata.json",
        "snapshot_file": "demo_paper_2_collection_snapshot.json",
        "not_found_msg": "Demo Paper 2 file not found.",
        "success_msg": "Demo Paper 2 loaded successfully.",
    },
}

# Directory for pre-indexed demo collection snapshots (version-controlled)
_DEMO_ASSETS_DIR = "demo_collection_assets"


def _demo_asset_path(filename: str) -> Path:
    """Return a path relative to the directory containing app.py."""
    return Path(__file__).resolve().parent / filename


def _demo_snapshot_path(paper_num: int) -> Path:
    """Return the path to the pre-indexed snapshot file for a demo paper."""
    dcfg = _DEMO_PAPER_CONFIG[paper_num]
    return _demo_asset_path(_DEMO_ASSETS_DIR) / dcfg["snapshot_file"]


def _restore_demo_sqlite_and_bm25(paper_num: int, paper_id: str, title: str) -> None:
    """Restore SQLite records and BM25 index for a demo paper from the snapshot.

    This reads chunk data (text, section, page_number, etc.) from the snapshot
    file and populates SQLite + BM25 — without calling the embedding model.

    This is needed when Qdrant has the vectors (restored from snapshot) but
    SQLite and BM25 have not been hydrated yet (e.g. after container restart
    where /tmp is wiped).
    """
    db = get_db()
    retriever = get_retriever()
    snapshot_path = _demo_snapshot_path(paper_num)

    # First try to hydrate from the snapshot file (authoritative source)
    if snapshot_path.exists():
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snapshot_data = json.load(f)
            points = snapshot_data.get("points", [])

            # Ensure the paper record exists in SQLite
            demo_pdf_path = _demo_asset_path(_DEMO_PAPER_CONFIG[paper_num]["pdf_file"])
            if not db.get_paper(paper_id):
                db.upsert_paper(paper_id=paper_id, title=title, filepath=str(demo_pdf_path))

            # Insert chunks into SQLite and collect texts for BM25
            existing_bm25_ids = set(retriever.bm25_ids)
            new_texts, new_ids = [], []
            for pt in points:
                payload = pt.get("payload", {})
                cid = payload.get("chunk_id", "")
                # Insert chunk into SQLite (idempotent)
                db.insert_chunk(
                    chunk_id=cid,
                    paper_id=payload.get("paper_id", paper_id),
                    title=payload.get("title", title),
                    section=payload.get("section", ""),
                    page_number=payload.get("page_number", 0),
                    text=payload.get("text", ""),
                    char_start=0,
                    char_end=len(payload.get("text", "")),
                )
                # Collect for BM25 if not already indexed
                if cid and cid not in existing_bm25_ids:
                    new_texts.append(payload.get("text", ""))
                    new_ids.append(cid)

            if new_texts:
                retriever.index_bm25(new_texts, new_ids)

            logger.info(
                "Demo paper %d: restored %d chunks to SQLite + BM25 from snapshot",
                paper_num, len(points),
            )
            return
        except Exception as exc:
            logger.warning(
                "Demo paper %d: failed to restore SQLite from snapshot (%s) — "
                "falling back to Qdrant scroll",
                paper_num, exc,
            )

    # Fallback: re-populate from Qdrant vectors (scroll with payloads)
    vs = get_vector_store()
    if vs.has_demo_collection(paper_num):
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            collection_name = vs.DEMO_COLLECTIONS[paper_num]
            records, _ = vs.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
                ),
                limit=500,
                with_payload=True,
                with_vectors=False,
            )

            demo_pdf_path = _demo_asset_path(_DEMO_PAPER_CONFIG[paper_num]["pdf_file"])
            if not db.get_paper(paper_id):
                db.upsert_paper(paper_id=paper_id, title=title, filepath=str(demo_pdf_path))

            existing_bm25_ids = set(retriever.bm25_ids)
            reindex_texts, reindex_ids = [], []
            for rec in records:
                p = rec.payload
                cid = p.get("chunk_id", "")
                db.insert_chunk(
                    chunk_id=cid,
                    paper_id=p.get("paper_id", paper_id),
                    title=p.get("title", title),
                    section=p.get("section", ""),
                    page_number=p.get("page_number", 0),
                    text=p.get("text", ""),
                    char_start=0,
                    char_end=len(p.get("text", "")),
                )
                if cid not in existing_bm25_ids:
                    reindex_texts.append(p.get("text", ""))
                    reindex_ids.append(cid)
            if reindex_texts:
                retriever.index_bm25(reindex_texts, reindex_ids)

            logger.info(
                "Demo paper %d: restored %d chunks to SQLite + BM25 from Qdrant scroll",
                paper_num, len(records),
            )
        except Exception as exc:
            logger.error(
                "Demo paper %d: failed to restore from Qdrant scroll: %s",
                paper_num, exc,
            )


def _initialize_single_demo(paper_num: int) -> Optional[str]:
    """Initialise or restore a single demo paper by number (1 or 2).

    STARTUP BEHAVIOUR (NO re-embedding guarantee):
    ──────────────────────────────────────────────
    1. If Qdrant collection already exists locally with points → load directly.
    2. If missing, restore from the pre-indexed snapshot file in the repo.
    3. Ensure SQLite and BM25 are hydrated from the snapshot (or Qdrant scroll
       as fallback).
    4. NEVER: re-read PDFs, re-chunk, re-generate embeddings, recreate vectors.
    """
    db = get_db()
    vs = get_vector_store()
    dcfg = _DEMO_PAPER_CONFIG[paper_num]

    demo_pdf_path = _demo_asset_path(dcfg["pdf_file"])
    demo_metadata_path = _demo_asset_path(dcfg["metadata_file"])
    snapshot_path = _demo_snapshot_path(paper_num)

    # Load metadata (used for paper_id and title regardless of path taken)
    metadata: dict = {}
    if demo_metadata_path.exists():
        try:
            with open(demo_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:
            logger.warning("Could not read %s: %s", dcfg["metadata_file"], exc)

    paper_id = metadata.get("paper_id", dcfg["paper_id"])
    title = metadata.get("title", dcfg["title"])

    # ───────────────────────────────────────────────────────────────────────
    # PATH A: Qdrant collection exists locally with points — instant load
    # ───────────────────────────────────────────────────────────────────────
    if vs.has_demo_collection(paper_num) and vs.demo_collection_point_count(paper_num) > 0:
        logger.info(
            "Demo paper %d: Qdrant collection exists with %d points — loading directly",
            paper_num, vs.demo_collection_point_count(paper_num),
        )

        # Ensure SQLite + BM25 are hydrated
        _restore_demo_sqlite_and_bm25(paper_num, paper_id, title)

        st.session_state[f"demo_paper_{paper_num}_loaded"] = True
        return paper_id

    if snapshot_path.exists():
        logger.info(
            "Demo paper %d: Qdrant collection missing — restoring from snapshot %s",
            paper_num, snapshot_path,
        )
        success = vs.restore_demo_collection_from_snapshot(paper_num, snapshot_path)
        if success:
            # Hydrate SQLite + BM25 from the same snapshot
            _restore_demo_sqlite_and_bm25(paper_num, paper_id, title)

            st.session_state[f"demo_paper_{paper_num}_loaded"] = True
            logger.info(
                "Demo paper %d: fully restored from snapshot — ready for retrieval",
                paper_num,
            )
            return paper_id
        else:
            logger.error(
                "Demo paper %d: snapshot restore failed — cannot load demo paper",
                paper_num,
            )
            st.error(
                f"Demo Paper {paper_num} could not be restored from snapshot. "
                "Please re-export snapshots or check the demo_collection_assets directory."
            )
            return None

    logger.info(
        "Demo paper %d: first-time initialisation — will embed and auto-export snapshot",
        paper_num,
    )

    if not demo_pdf_path.exists():
        st.error(dcfg["not_found_msg"])
        logger.warning("%s not found at %s", dcfg["pdf_file"], demo_pdf_path)
        return None

    try:
        cfg = get_config()
        paper_id = dcfg["paper_id"]
        title = dcfg["title"]

        # 1. Extract text from the demo PDF
        with st.spinner(f"Demo Paper {paper_num}: extracting text from PDF (first-time setup)…"):
            pages = extract_text_from_pdf(str(demo_pdf_path))
        if not pages:
            st.error(f"Could not extract text from {dcfg['pdf_file']}.")
            return None

        # 2. Smart chunking
        with st.spinner(f"Demo Paper {paper_num}: chunking (first-time setup)…"):
            chunks = smart_chunk(pages, paper_id, title, cfg.chunk_size, cfg.chunk_overlap)
        if not chunks:
            st.error(f"Demo Paper {paper_num} produced no chunks.")
            return None

        # 3. Create the dedicated demo collection
        vs._ensure_demo_collection(paper_num)

        # 4. Store paper metadata in SQLite (idempotent)
        db.upsert_paper(paper_id=paper_id, title=title, filepath=str(demo_pdf_path))

        # 5. Generate embeddings & store in demo_paper_collection_N
        with st.spinner(f"Demo Paper {paper_num}: embedding {len(chunks)} chunks (first-time setup — this won't happen again)…"):
            texts = [c["text"] for c in chunks]
            vectors = embed_texts(texts)
            payloads = [
                {
                    "chunk_id": c["chunk_id"],
                    "paper_id": c["paper_id"],
                    "title": c["title"],
                    "section": c["section"],
                    "page_number": c["page_number"],
                    "text": c["text"],
                }
                for c in chunks
            ]
            collection_name = vs.DEMO_COLLECTIONS[paper_num]
            vs.batch_upsert(
                ids=[c["chunk_id"] for c in chunks],
                vectors=vectors,
                payloads=payloads,
                collection_name=collection_name,
            )

        # 6. Store chunks in SQLite for BM25, literature review, etc.
        for c in chunks:
            db.insert_chunk(
                chunk_id=c["chunk_id"],
                paper_id=c["paper_id"],
                title=c["title"],
                section=c["section"],
                page_number=c["page_number"],
                text=c["text"],
                char_start=c["char_start"],
                char_end=c["char_end"],
            )

        # 7. Index BM25
        retriever = get_retriever()
        retriever.index_bm25(texts, [c["chunk_id"] for c in chunks])

        _auto_export_demo_snapshot(paper_num)

        # 8. Save metadata JSON alongside the PDF
        metadata_out = {
            "paper_id": paper_id,
            "title": title,
            "num_chunks": len(chunks),
            "source": dcfg["pdf_file"],
            "collection_name": collection_name,
        }
        with open(str(demo_metadata_path), "w", encoding="utf-8") as f:
            json.dump(metadata_out, f, indent=2)
        logger.info("Demo paper %d metadata saved to %s", paper_num, demo_metadata_path)

        st.session_state[f"demo_paper_{paper_num}_loaded"] = True
        logger.info(
            "Demo Paper %d initialised — %d chunks in %s (snapshot auto-exported)",
            paper_num, len(chunks), collection_name,
        )
        return paper_id

    except Exception as exc:
        logger.error("Demo Paper %d first-time initialisation failed: %s", paper_num, exc)
        st.error(f"Unable to initialise demo paper {paper_num}: {exc}")
        return None


def initialize_demo_paper_1() -> Optional[str]:
    """Load Demo Paper 1."""
    result = _initialize_single_demo(1)
    if result:
        st.success("Demo Paper 1 loaded successfully.")
    return result


def initialize_demo_paper_2() -> Optional[str]:
    """Load Demo Paper 2."""
    result = _initialize_single_demo(2)
    if result:
        st.success("Demo Paper 2 loaded successfully.")
    return result


def initialize_demo_paper_set() -> Optional[list[str]]:
    """Load both demo papers for comparison demonstrations."""
    results = []
    pid1 = _initialize_single_demo(1)
    if pid1:
        results.append(pid1)
    pid2 = _initialize_single_demo(2)
    if pid2:
        results.append(pid2)
    if results:
        st.success("Demo paper set loaded successfully. You can now compare papers immediately.")
    return results if results else None



def _auto_export_demo_snapshot(paper_num: int) -> bool:
    """Auto-export a single demo collection to a snapshot JSON file.

    Called automatically after Path C (first-time embedding) completes.
    Creates the demo_collection_assets/ directory if needed and writes
    the snapshot file containing all precomputed vectors and payloads.

    Returns True on success, False on failure.
    """
    vs = get_vector_store()
    dcfg = _DEMO_PAPER_CONFIG[paper_num]
    collection_name = vs.DEMO_COLLECTIONS[paper_num]

    if not vs.has_demo_collection(paper_num) or vs.demo_collection_point_count(paper_num) == 0:
        logger.warning(
            "Demo paper %d: cannot auto-export — collection '%s' is missing or empty",
            paper_num, collection_name,
        )
        return False

    # Ensure the assets directory exists
    assets_dir = _demo_asset_path(_DEMO_ASSETS_DIR)
    assets_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Demo paper %d: auto-exporting snapshot from '%s'...",
        paper_num, collection_name,
    )

    try:
        import datetime as _dt

        all_points = []
        offset = None
        while True:
            records, next_offset = vs.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for rec in records:
                all_points.append({
                    "id": rec.id if isinstance(rec.id, str) else str(rec.id),
                    "vector": rec.vector,
                    "payload": rec.payload,
                })
            if next_offset is None:
                break
            offset = next_offset

        snapshot_path = assets_dir / dcfg["snapshot_file"]
        snapshot_data = {
            "collection_name": collection_name,
            "dimension": vs.DIMENSION,
            "point_count": len(all_points),
            "export_timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "points": all_points,
        }
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, indent=2)

        logger.info(
            "Demo paper %d: auto-exported %d points → %s (commit this file to git!)",
            paper_num, len(all_points), snapshot_path,
        )
        return True

    except Exception as exc:
        logger.error("Demo paper %d: auto-export failed: %s", paper_num, exc)
        return False


def export_demo_snapshots() -> dict[int, bool]:
    """Manually export all demo Qdrant collections to JSON snapshot files.

    This is a developer utility.  It reads all vectors and payloads from
    each demo collection and writes them to JSON files in
    demo_collection_assets/.

    Returns a dict mapping paper_num → success (bool).
    """
    logger.info("=== DEMO SNAPSHOT EXPORT (one-time developer tool) ===")

    # Ensure the assets directory exists
    assets_dir = _demo_asset_path(_DEMO_ASSETS_DIR)
    assets_dir.mkdir(parents=True, exist_ok=True)

    vs = get_vector_store()
    results: dict[int, bool] = {}

    for paper_num in _DEMO_PAPER_CONFIG:
        collection_name = vs.DEMO_COLLECTIONS[paper_num]
        dcfg = _DEMO_PAPER_CONFIG[paper_num]

        if not vs.has_demo_collection(paper_num):
            logger.warning(
                "Demo paper %d: collection '%s' does not exist — skipping export",
                paper_num, collection_name,
            )
            results[paper_num] = False
            continue

        point_count = vs.demo_collection_point_count(paper_num)
        if point_count == 0:
            logger.warning(
                "Demo paper %d: collection '%s' is empty — skipping export",
                paper_num, collection_name,
            )
            results[paper_num] = False
            continue

        logger.info(
            "Demo paper %d: exporting %d points from '%s'...",
            paper_num, point_count, collection_name,
        )

        try:
            # Scroll through ALL points with vectors + payloads
            all_points = []
            offset = None
            while True:
                records, next_offset = vs.client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                for rec in records:
                    all_points.append({
                        "id": rec.id if isinstance(rec.id, str) else str(rec.id),
                        "vector": rec.vector,
                        "payload": rec.payload,
                    })

                if next_offset is None:
                    break
                offset = next_offset

            # Write snapshot file
            snapshot_path = assets_dir / dcfg["snapshot_file"]
            snapshot_data = {
                "collection_name": collection_name,
                "dimension": vs.DIMENSION,
                "point_count": len(all_points),
                "export_timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "points": all_points,
            }
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(snapshot_data, f, indent=2)

            logger.info(
                "Demo paper %d: exported %d points → %s",
                paper_num, len(all_points), snapshot_path,
            )
            results[paper_num] = True

        except Exception as exc:
            logger.error(
                "Demo paper %d: export failed: %s", paper_num, exc,
            )
            results[paper_num] = False

    logger.info("=== SNAPSHOT EXPORT COMPLETE: %s ===", results)
    return results



def _init_session_state() -> None:
    """Set defaults for session-state keys."""
    defaults = {
        "page": "Dashboard",
        "query_history": [],
        "demo_paper_1_loaded": False,
        "demo_paper_2_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _sidebar() -> None:
    """Render the sidebar navigation."""
    with st.sidebar:
        st.title("🔬 AI Research Copilot")
        st.caption("RAG-powered research assistant")
        st.divider()

        pages = ["Dashboard", "Ingest Papers", "Ask Questions", "Literature Review", "Compare Papers"]
        choice = st.radio("Navigate", pages, label_visibility="collapsed")
        st.session_state.page = choice

        st.divider()
        cfg = get_config()
        db = get_db()
        st.metric("Papers", db.paper_count())
        st.metric("Chunks", db.chunk_count())

        if cfg.groq_api_key:
            st.success("Groq API ✓", icon="✅")
        else:
            st.error("Groq API ✗", icon="❌")

        st.divider()
        st.caption("Built with ❤️ for GenAI interviews")


# -- Dashboard --------------------------------------------------------------

def page_dashboard() -> None:
    st.header("📊 Dashboard")
    db = get_db()
    papers = db.list_papers()

    col1, col2, col3 = st.columns(3)
    col1.metric("Papers Ingested", len(papers))
    col2.metric("Total Chunks", db.chunk_count())
    col3.metric("Embedding Model", get_config().embedding_model)

    st.divider()

    if papers:
        st.subheader("Recently Ingested Papers")
        for p in papers[:10]:
            with st.expander(f"📄 {p['title']}"):
                st.markdown(f"**Authors:** {p.get('authors', 'N/A')}")
                st.markdown(f"**arXiv ID:** `{p.get('arxiv_id', 'N/A')}`")
                st.markdown(f"**Ingested:** {p.get('ingested_at', 'N/A')}")
                if p.get("abstract"):
                    st.markdown("**Abstract:**")
                    st.write(p["abstract"][:500] + ("..." if len(p["abstract"]) > 500 else ""))
    else:
        st.info("No papers ingested yet. Head to **Ingest Papers** to get started!")

    st.divider()
    st.subheader("Architecture Overview")
    st.markdown(
        """
        | Component | Technology |
        |---|---|
        | LLM | Groq — llama-3.1-8b-instant |
        | Embeddings | sentence-transformers — all-MiniLM-L6-v2 |
        | Vector DB | Qdrant (local mode) |
        | Metadata DB | SQLite |
        | PDF Parsing | PyMuPDF |
        | Sparse Retrieval | BM25Okapi |
        | Dense Retrieval | Qdrant COSINE search |
        | Fusion | Reciprocal Rank Fusion (RRF) |
        | Chunking | Section-aware + overlap + citation-preserving |
        | Demo Papers | Pre-indexed snapshots (no re-embedding on restart) |
        """
    )


# -- Ingest Papers ----------------------------------------------------------

def page_ingest() -> None:
    st.header("📥 Ingest Papers")

    tab_upload, tab_arxiv = st.tabs(["Upload PDF", "arXiv Paper"])

    with tab_upload:
        st.subheader("Upload a PDF")
        uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
        paper_title = st.text_input("Paper title (optional)", placeholder="My Custom Paper")
        if st.button("Ingest Upload", type="primary", disabled=uploaded is None):
            result = ingest_uploaded_pdf(uploaded, paper_title)
            if result is None:
                st.error("Ingestion failed. Could not process the PDF.")

        st.divider()
        st.subheader("Demo Papers")
        st.caption(
            "Load built-in demo papers for quick testing — no upload needed. "
            "Demo papers are pre-indexed and load instantly without re-embedding."
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Load Demo Paper 1"):
                initialize_demo_paper_1()
        with col2:
            if st.button("📄 Load Demo Paper 2"):
                initialize_demo_paper_2()
        if st.button("📚 Load Demo Paper Set"):
            initialize_demo_paper_set()

    with tab_arxiv:
        st.subheader("Ingest from arXiv")
        arxiv_id = st.text_input(
            "arXiv ID",
            placeholder="e.g. 2401.00001 or 2401.00001v2",
            help="Enter the arXiv identifier for the paper you want to ingest.",
        )
        if st.button("Ingest from arXiv", type="primary", disabled=not arxiv_id.strip()):
            result = ingest_arxiv_paper(arxiv_id.strip())
            if result is None:
                st.error("Ingestion failed. Check the arXiv ID and try again.")


# -- Ask Questions ----------------------------------------------------------

def page_ask() -> None:
    st.header("❓ Ask Questions")

    db = get_db()
    if db.paper_count() == 0:
        st.warning("No papers in the database yet. Ingest some papers first!")
        return

    query = st.text_input("Research question", placeholder="e.g. What is the main contribution of this paper?")
    top_k = st.slider("Number of sources", min_value=2, max_value=4, value=2)

    if st.button("Search & Answer", type="primary", disabled=not query.strip()):
        retriever = get_retriever()
        with st.spinner("Retrieving relevant chunks..."):
            results = retriever.search(query.strip(), top_k=top_k)

        if not results:
            st.warning("No relevant chunks found. Try a different query.")
            return

        # Build context with citations
        context_parts = []
        for idx, r in enumerate(results, start=1):
            context_parts.append(
                f"[Source {idx}] Paper: {r.get('title', 'N/A')}\n"
                f"Section: {r.get('section', 'N/A')} | Page: {r.get('page_number', 'N/A')}\n"
                f"RRF Score: {r.get('rrf_score', 0):.4f}\n"
                f"\"{r.get('text', '')}\""
            )
        context = "\n\n".join(context_parts)
        context=context[:7000]

        system_prompt = (
            "You are an expert research assistant. Answer the user's question using ONLY "
            "the provided sources. For every claim, cite the source using [Source N] format. "
            "If the sources do not contain enough information, say so honestly. "
            "Format your answer with:\n"
            "- A clear, concise answer\n"
            "- Citation blocks in this format:\n"
            "  [Paper: XYZ]\n"
            "  Section: ABC\n"
            "  Evidence:\n"
            '  "..."\n'
        )
        user_prompt = f"Sources:\n{context}\n\nQuestion: {query.strip()}\n\nAnswer:"

        with st.spinner("Generating answer with Groq..."):
            answer = call_groq(system_prompt, user_prompt, max_tokens=1024)

        st.subheader("Answer")
        st.markdown(answer)

        st.divider()
        st.subheader("Retrieved Sources")
        for idx, r in enumerate(results, start=1):
            with st.expander(f"Source {idx} — {r.get('title', 'N/A')} (Score: {r.get('rrf_score', 0):.4f})"):
                st.markdown(f"**Section:** {r.get('section', 'N/A')}")
                st.markdown(f"**Page:** {r.get('page_number', 'N/A')}")
                st.markdown(f"**Chunk ID:** `{r.get('chunk_id', 'N/A')}`")
                st.markdown("**Text:**")
                st.write(r.get("text", ""))

        # Save to query history
        st.session_state.query_history.append(
            {"query": query.strip(), "num_sources": len(results)}
        )

    # Query history
    if st.session_state.query_history:
        st.divider()
        st.subheader("Query History")
        for i, h in enumerate(reversed(st.session_state.query_history), start=1):
            st.markdown(f"{i}. **{h['query']}** — {h['num_sources']} sources")


# -- Literature Review ------------------------------------------------------

def page_literature_review() -> None:
    st.header("📚 Literature Review")

    db = get_db()
    papers = db.list_papers()
    if len(papers) < 1:
        st.warning("Ingest at least one paper before generating a literature review.")
        return

    paper_options = {f"{p['title']} ({p['paper_id'][:8]})": p["paper_id"] for p in papers}
    selected = st.multiselect(
        "Select papers for the review",
        list(paper_options.keys()),
        help="Choose 1 or more papers to include in the literature review.",
    )

    topic = st.text_input(
        "Review topic / focus area",
        placeholder="e.g. Transformer architectures in computer vision",
    )

    if st.button("Generate Literature Review", type="primary", disabled=not selected or not topic.strip()):
        # Collect all chunks for selected papers
        all_chunks: list[dict] = []
        for label in selected:
            pid = paper_options[label]
            all_chunks.extend(db.get_chunks_by_paper(pid))

        if not all_chunks:
            st.warning("No chunks found for the selected papers.")
            return

        # Use top chunks by section diversity
        section_chunks: dict[str, list[dict]] = {}
        for ch in all_chunks:
            section_chunks.setdefault(ch["section"], []).append(ch)

        # Pick up to 2 chunks per section
        representative: list[dict] = []
        for sec, chs in section_chunks.items():
            representative.extend(chs[:2])

        context_parts = []
        for ch in representative:
            context_parts.append(
                f"Paper: {ch['title']}\nSection: {ch['section']} (Page {ch['page_number']})\n\"{ch['text']}\""
            )
        context = "\n\n---\n\n".join(context_parts)
        context=context[:7000]

        system_prompt = (
            "You are an expert academic writer. Generate a coherent literature review on the given topic "
            "using ONLY the provided paper excerpts. Synthesize findings, compare approaches, identify "
            "gaps, and cite papers using [Paper: title] format. Structure the review with clear headings."
        )
        user_prompt = (
            f"Topic: {topic.strip()}\n\n"
            f"Paper excerpts:\n{context}\n\n"
            "Write a comprehensive literature review."
        )

        with st.spinner("Generating literature review with Groq..."):
            review = call_groq(system_prompt, user_prompt, max_tokens=2048)

        st.subheader("Literature Review")
        st.markdown(review)

        st.divider()
        st.subheader("Papers Included")
        for label in selected:
            pid = paper_options[label]
            p = db.get_paper(pid)
            if p:
                st.markdown(f"- **{p['title']}** — {p.get('authors', 'N/A')}")


# -- Compare Papers ---------------------------------------------------------

def page_compare() -> None:
    st.header("⚖️ Compare Papers")

    db = get_db()
    papers = db.list_papers()
    if len(papers) < 2:
        st.warning("Ingest at least two papers before comparing.")
        return

    paper_options = {f"{p['title']} ({p['paper_id'][:8]})": p["paper_id"] for p in papers}
    labels = list(paper_options.keys())

    col1, col2 = st.columns(2)
    with col1:
        sel_a = st.selectbox("Paper A", labels, index=0)
    with col2:
        sel_b = st.selectbox("Paper B", labels, index=min(1, len(labels) - 1))

    if sel_a == sel_b:
        st.warning("Please select two different papers to compare.")
        return

    aspect = st.text_input(
        "Comparison aspect (optional)",
        placeholder="e.g. Methodology, Performance, Datasets used",
    )

    if st.button("Compare", type="primary"):
        pid_a = paper_options[sel_a]
        pid_b = paper_options[sel_b]

        chunks_a = db.get_chunks_by_paper(pid_a)
        chunks_b = db.get_chunks_by_paper(pid_b)

        # Pick representative chunks (Abstract, Method, Results)
        key_sections = ["abstract", "introduction", "method", "methodology", "approach", "experiment", "results", "conclusion"]

        def _pick(chunks: list[dict], n: int = 3) -> list[dict]:
            picked: list[dict] = []
            for ks in key_sections:
                for ch in chunks:
                    if ch["section"].lower().startswith(ks) and ch not in picked:
                        picked.append(ch)
                        break
                if len(picked) >= n:
                    break
            # Fill remaining with any chunks
            for ch in chunks:
                if ch not in picked:
                    picked.append(ch)
                if len(picked) >= n:
                    break
            return picked[:n]

        rep_a = _pick(chunks_a)
        rep_b = _pick(chunks_b)

        ctx_a = "\n\n".join(
            f"Section: {c['section']} (Page {c['page_number']})\n\"{c['text']}\""
            for c in rep_a
        )
        ctx_a = ctx_a[:4000]
        ctx_b = "\n\n".join(
            f"Section: {c['section']} (Page {c['page_number']})\n\"{c['text']}\""
            for c in rep_b
        )
        ctx_b = ctx_b[:4000]

        aspect_instruction = f" Focus especially on: {aspect.strip()}." if aspect.strip() else ""

        system_prompt = (
            "You are an expert research analyst. Compare two research papers based on the provided "
            "excerpts. Highlight similarities, differences, strengths, and weaknesses. "
            "Cite papers using [Paper: title] format. Structure the comparison with clear headings."
            f"{aspect_instruction}"
        )
        user_prompt = (
            f"**Paper A excerpts:**\n{ctx_a}\n\n"
            f"**Paper B excerpts:**\n{ctx_b}\n\n"
            "Provide a detailed comparison."
        )

        with st.spinner("Generating comparison with Groq..."):
            comparison = call_groq(system_prompt, user_prompt, max_tokens=512)

        st.subheader("Comparison")
        st.markdown(comparison)



def main() -> None:
    """Application entry point."""
    _init_session_state()
    _sidebar()

    page = st.session_state.page

    if page == "Dashboard":
        page_dashboard()
    elif page == "Ingest Papers":
        page_ingest()
    elif page == "Ask Questions":
        page_ask()
    elif page == "Literature Review":
        page_literature_review()
    elif page == "Compare Papers":
        page_compare()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()

"""高レベルAPI: ingest_all() / ingest_url() / answer() / answer_stream()。

UI（Streamlit）とCLI（ingest.py）の双方がこれを使う。
個々のモジュールの実装詳細はここに漏れない。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from .chunker import chunk_pages
from .config import PAPERS_DIR, TOP_K
from .generator import answer_question, stream_answer_question
from .pdf_loader import load_pdf
from .registry import record_ingestion
from .types import Answer, Hit
from .url_loader import download_pdf
from .vectorstore import VectorStore


def ingest_one(
    pdf_path: Path,
    store: VectorStore,
    *,
    source_url: Optional[str] = None,
) -> int:
    """PDF1本を取り込んで、追加されたチャンク数を返す。

    成功時はレジストリ（SQLite）にメタデータを記録する。
    """
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)
    n_added = store.add_chunks(chunks)
    if pages:
        record_ingestion(
            source_path=str(pdf_path),
            paper_title=pages[0].paper_title,
            n_pages=len(pages),
            n_chunks=n_added,
            source_url=source_url,
        )
    return n_added


def ingest_all() -> dict[str, int]:
    """papers/ 配下のPDFを全て取り込む。{ファイル名: 追加チャンク数} を返す。"""
    store = VectorStore()
    results: dict[str, int] = {}
    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        results[pdf_path.name] = ingest_one(pdf_path, store)
    return results


def ingest_url(url: str) -> tuple[str, int]:
    """URLからPDFを取得して取り込む。(保存ファイル名, 追加チャンク数) を返す。"""
    pdf_path = download_pdf(url)
    store = VectorStore()
    n = ingest_one(pdf_path, store, source_url=url)
    return pdf_path.name, n


def _retrieve(query: str, top_k: int) -> list[Hit]:
    store = VectorStore()
    return store.search(query, top_k=top_k)


def answer(query: str, top_k: int = TOP_K) -> Answer:
    """検索 → 生成（ブロッキング）。"""
    hits = _retrieve(query, top_k=top_k)
    return answer_question(query, hits)


def answer_stream(
    query: str, top_k: int = TOP_K
) -> tuple[Iterator[str], list[Hit]]:
    """検索 → 生成（ストリーミング）。

    呼び出し側は (stream, hits) を受け取り、stream を消費すると逐次テキストが返る。
    UI側で st.write_stream() に渡すユースケースを想定。
    """
    hits = _retrieve(query, top_k=top_k)
    stream = stream_answer_question(query, hits)
    return stream, hits

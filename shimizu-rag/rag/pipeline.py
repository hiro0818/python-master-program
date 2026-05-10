"""高レベルAPI: ingest_all() と answer()。

UI（Streamlit）とCLI（ingest.py）の双方がこれを使う。
個々のモジュールの実装詳細はここに漏れない。
"""
from __future__ import annotations

from pathlib import Path

from .chunker import chunk_pages
from .config import PAPERS_DIR, TOP_K
from .generator import answer_question
from .pdf_loader import load_pdf
from .types import Answer
from .vectorstore import VectorStore


def ingest_one(pdf_path: Path, store: VectorStore) -> int:
    """PDF1本を取り込んで、追加されたチャンク数を返す。"""
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)
    return store.add_chunks(chunks)


def ingest_all() -> dict[str, int]:
    """papers/ 配下のPDFを全て取り込む。{ファイル名: 追加チャンク数} を返す。"""
    store = VectorStore()
    results: dict[str, int] = {}
    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        results[pdf_path.name] = ingest_one(pdf_path, store)
    return results


def answer(query: str, top_k: int = TOP_K) -> Answer:
    """検索 → 生成。UIとCLIの公開エントリポイント。"""
    store = VectorStore()
    hits = store.search(query, top_k=top_k)
    return answer_question(query, hits)

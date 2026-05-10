"""Page -> Chunk リスト。スライディングウィンドウで分割。

ポイント:
    - チャンクは1ページ内に収まる（ページ番号引用の正確性を保つ）
    - chunk_id はハッシュで決定的に生成 → 再取り込み時に重複を避けられる
"""
from __future__ import annotations

import hashlib

from .config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS
from .types import Chunk, Page


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    """各ページを CHUNK_SIZE_CHARS 文字のウィンドウで分割。"""
    step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text
        if not text:
            continue
        if len(text) <= CHUNK_SIZE_CHARS:
            chunks.append(_make_chunk(page, text, 0))
            continue
        for window_start in range(0, len(text), step):
            window = text[window_start : window_start + CHUNK_SIZE_CHARS]
            if not window.strip():
                continue
            chunks.append(_make_chunk(page, window, window_start))
            if window_start + CHUNK_SIZE_CHARS >= len(text):
                break
    return chunks


def _make_chunk(page: Page, text: str, offset: int) -> Chunk:
    chunk_id = hashlib.sha1(
        f"{page.source_path}:{page.page_num}:{offset}".encode("utf-8")
    ).hexdigest()[:16]
    return Chunk(
        paper_title=page.paper_title,
        page_num=page.page_num,
        text=text,
        source_path=page.source_path,
        chunk_id=chunk_id,
    )

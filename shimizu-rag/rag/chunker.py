"""Page -> Chunk リスト。スライディングウィンドウで分割。

チャンクは1ページ内に収まる（ページ番号引用の正確性を保つため）。
実装は次のコミットで埋める。
"""
from __future__ import annotations

from .types import Chunk, Page


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    raise NotImplementedError("Phase 2 で実装")

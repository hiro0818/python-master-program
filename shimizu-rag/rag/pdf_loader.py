"""PDF -> Page リスト。pdfplumberでページ番号を保持したままテキスト抽出。

実装は次のコミットで埋める。
"""
from __future__ import annotations

from pathlib import Path

from .types import Page


def load_pdf(pdf_path: Path) -> list[Page]:
    raise NotImplementedError("Phase 2 で実装")

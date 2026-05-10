"""高レベルAPI: ingest_all() と answer()。

UI（Streamlit）とCLI（ingest.py）の双方がこれを使う。
実装は次のコミットで埋める。
"""
from __future__ import annotations

from .types import Answer


def ingest_all() -> dict[str, int]:
    raise NotImplementedError("Phase 2 で実装")


def answer(query: str) -> Answer:
    raise NotImplementedError("Phase 2 で実装")

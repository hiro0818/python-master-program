"""LanceDB ラッパー。チャンク保存とベクター検索を提供する。

埋め込みモデル: BAAI/bge-m3（sentence-transformers経由）
ストレージ: LanceDB（Lance/Parquet互換、Apache 2.0、ローカル永続化）
実装は次のコミットで埋める。
"""
from __future__ import annotations

from .types import Chunk, Hit


class VectorStore:
    def __init__(self) -> None:
        raise NotImplementedError("Phase 2 で実装")

    def add_chunks(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError("Phase 2 で実装")

    def search(self, query: str, top_k: int = 8) -> list[Hit]:
        raise NotImplementedError("Phase 2 で実装")

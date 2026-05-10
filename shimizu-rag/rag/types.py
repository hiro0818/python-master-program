"""モジュール間の通信に使うデータ型（不変インターフェース）。

ここに定義された dataclass が各モジュールの『契約』。
内部実装（PDF解析、埋め込みモデル、ベクターDB、LLM）が変わっても、
この型は変わらない。これがロックイン回避の核。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    """PDFの1ページ分のテキスト。"""

    paper_title: str
    page_num: int
    text: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    """検索の最小単位。1ページを複数のチャンクに分割する。"""

    paper_title: str
    page_num: int
    text: str
    source_path: str
    chunk_id: str


@dataclass(frozen=True)
class Hit:
    """ベクター検索の結果1件。"""

    text: str
    score: float
    paper_title: str
    page_num: int
    source_path: str


@dataclass(frozen=True)
class Answer:
    """LLMの回答 + 引用元のリスト。"""

    text: str
    citations: list[Hit]

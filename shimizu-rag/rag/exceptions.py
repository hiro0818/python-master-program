"""清水研RAG の例外階層。

UIや上位層がまとめて捕捉できるよう、すべて RAGError を継承する。
メッセージは日本語（UIにそのまま表示する想定）。
"""
from __future__ import annotations


class RAGError(Exception):
    """清水研RAG が投げる全例外の基底。"""


class PDFLoadError(RAGError):
    """PDF の読み込み・解析に失敗した。"""


class EmbeddingError(RAGError):
    """埋め込みモデルのロード／推論に失敗した、または次元が不一致。"""


class URLDownloadError(RAGError):
    """URL からの PDF 取得に失敗した。"""


class CitationError(RAGError):
    """引用検証に関する問題。"""


class RegistryError(RAGError):
    """論文レジストリ（SQLite）の読み書きに失敗した。"""

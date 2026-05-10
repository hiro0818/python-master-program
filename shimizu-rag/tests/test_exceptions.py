"""例外階層が想定通りであることを確認する。

UI層が `except RAGError` で全て捕捉できることが保証されないと、ユーザー向けの
エラーメッセージが日本語にならず想定外Exceptionが赤いトレースバックで出てしまう。
"""
from __future__ import annotations

from rag.exceptions import (
    CitationError,
    EmbeddingError,
    PDFLoadError,
    RAGError,
    RegistryError,
    URLDownloadError,
)


def test_all_subclasses_inherit_from_rag_error() -> None:
    for cls in (
        PDFLoadError,
        EmbeddingError,
        URLDownloadError,
        CitationError,
        RegistryError,
    ):
        assert issubclass(cls, RAGError)


def test_url_download_error_alias_in_url_loader() -> None:
    """app.py が `from rag.url_loader import URLDownloadError` で参照できる。"""
    from rag.url_loader import URLDownloadError as UrlLoaderURLDownloadError

    assert UrlLoaderURLDownloadError is URLDownloadError


def test_rag_error_can_catch_all_descendants() -> None:
    for exc in (
        PDFLoadError("a"),
        EmbeddingError("b"),
        URLDownloadError("c"),
        CitationError("d"),
        RegistryError("e"),
    ):
        try:
            raise exc
        except RAGError as caught:
            assert isinstance(caught, type(exc))

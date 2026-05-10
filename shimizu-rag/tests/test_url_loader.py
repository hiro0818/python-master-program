"""url_loader の純粋ロジック部分（ネットワーク不要部分）のテスト。"""
from __future__ import annotations

import pytest

from rag.exceptions import URLDownloadError
from rag.url_loader import _filename_from_url, download_pdf


def test_filename_extracted_from_url() -> None:
    assert _filename_from_url("https://example.com/papers/foo.pdf") == "foo.pdf"


def test_filename_appends_pdf_extension_if_missing() -> None:
    name = _filename_from_url("https://example.com/foo")
    assert name.endswith(".pdf")


def test_filename_strips_unsafe_characters() -> None:
    name = _filename_from_url("https://example.com/foo bar/中文.pdf")
    # 英数 / . _ - のみが残る → 漢字も空白も消える
    for ch in name:
        assert ch.isalnum() or ch in "._-"


def test_filename_falls_back_when_path_empty() -> None:
    name = _filename_from_url("https://example.com/")
    assert name == "downloaded.pdf"


def test_download_rejects_non_http_scheme() -> None:
    with pytest.raises(URLDownloadError):
        download_pdf("ftp://example.com/foo.pdf")


def test_download_rejects_empty_url() -> None:
    with pytest.raises(URLDownloadError):
        download_pdf("")

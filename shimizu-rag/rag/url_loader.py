"""URL から PDF をダウンロードして papers/ に保存する。

httpx を使用（Anthropic SDK が依存しているため追加インストール不要）。
保存先は papers/ に統一し、既存のロード経路（pdf_loader → chunker）に
そのまま乗せる。
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import PAPERS_DIR

DEFAULT_TIMEOUT_SEC = 60.0
MAX_BYTES = 50 * 1024 * 1024  # 50 MB 上限（暴走対策）


class URLDownloadError(Exception):
    """URL取得に関する全ての失敗を表す。"""


def download_pdf(url: str, timeout: float = DEFAULT_TIMEOUT_SEC) -> Path:
    """指定URLからPDFをダウンロードして papers/ 配下に保存。

    成功時: 保存先 Path を返す
    失敗時: URLDownloadError

    PDF判定は Content-Type もしくは URLの .pdf 拡張子で行う。
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise URLDownloadError("URLは http:// または https:// で始まる必要があります。")

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    filename = _filename_from_url(url)
    save_path = PAPERS_DIR / filename

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            _validate_pdf_response(response, url)
            content = response.content
            if len(content) > MAX_BYTES:
                raise URLDownloadError(
                    f"ファイルサイズが大きすぎます ({len(content)/1e6:.1f} MB > 50 MB)"
                )
            save_path.write_bytes(content)
    except httpx.HTTPStatusError as e:
        raise URLDownloadError(
            f"HTTPエラー {e.response.status_code}: {url}"
        ) from e
    except httpx.RequestError as e:
        raise URLDownloadError(f"ダウンロード失敗: {e}") from e

    return save_path


def _validate_pdf_response(response: httpx.Response, url: str) -> None:
    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type:
        return
    if url.lower().split("?")[0].endswith(".pdf"):
        return
    raise URLDownloadError(
        f"PDFと判定できませんでした (Content-Type: {content_type or '不明'})"
    )


def _filename_from_url(url: str) -> str:
    """URLから安全なファイル名を生成。

    パス末尾を取り、英数と . _ - 以外を除去。.pdf 拡張子を保証する。
    """
    parsed = urlparse(url)
    raw = Path(parsed.path).name or "downloaded.pdf"
    safe = "".join(c for c in raw if c.isalnum() or c in "._-")
    if not safe:
        safe = "downloaded.pdf"
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    return safe

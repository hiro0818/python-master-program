"""PDF -> Page リスト。pdfplumberでページ番号を保持したままテキスト抽出。

なぜpdfplumber:
    - MITライセンス（AGPLのPyMuPDFを避ける）
    - ページ単位の正確な抽出（ページ番号引用の根拠になる）
    - 純Python（インストールが軽量）
"""
from __future__ import annotations

from pathlib import Path

import pdfplumber

from .exceptions import PDFLoadError
from .types import Page


def load_pdf(pdf_path: Path) -> list[Page]:
    """PDFを読み、ページごとに Page を返す。空ページはスキップ。

    失敗時は PDFLoadError を投げる（破損PDF、暗号化、IOエラー等）。
    """
    if not pdf_path.exists():
        raise PDFLoadError(f"PDFが見つかりません: {pdf_path}")
    if pdf_path.stat().st_size == 0:
        raise PDFLoadError(f"PDFが空ファイルです: {pdf_path.name}")

    title = _infer_title(pdf_path)
    pages: list[Page] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = (page.extract_text() or "").strip()
                except Exception:
                    # 個別ページの抽出失敗は致命でない（次ページへ）
                    continue
                if not text:
                    continue
                pages.append(
                    Page(
                        paper_title=title,
                        page_num=i,
                        text=text,
                        source_path=str(pdf_path),
                    )
                )
    except PDFLoadError:
        raise
    except Exception as e:
        raise PDFLoadError(
            f"PDFの解析に失敗しました ({pdf_path.name}): {type(e).__name__}: {e}"
        ) from e

    if not pages:
        raise PDFLoadError(
            f"テキストを抽出できませんでした ({pdf_path.name})。"
            "スキャンPDFや画像のみの可能性があります。"
        )
    return pages


def _infer_title(pdf_path: Path) -> str:
    """PDFメタデータの Title を優先、なければファイル名（拡張子なし）。"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            md = pdf.metadata or {}
            title = md.get("Title") or md.get("title")
            if isinstance(title, str) and title.strip():
                return title.strip()
    except Exception:
        pass
    return pdf_path.stem


def count_pages(pdf_path: Path) -> int:
    """PDFのページ数を返す（レジストリ記録用、失敗時は0）。"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0

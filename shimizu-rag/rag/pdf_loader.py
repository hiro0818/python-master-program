"""PDF -> Page リスト。pdfplumberでページ番号を保持したままテキスト抽出。

なぜpdfplumber:
    - MITライセンス（AGPLのPyMuPDFを避ける）
    - ページ単位の正確な抽出（ページ番号引用の根拠になる）
    - 純Python（インストールが軽量）
"""
from __future__ import annotations

from pathlib import Path

import pdfplumber

from .types import Page


def load_pdf(pdf_path: Path) -> list[Page]:
    """PDFを読み、ページごとに Page を返す。空ページはスキップ。"""
    title = _infer_title(pdf_path)
    pages: list[Page] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
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

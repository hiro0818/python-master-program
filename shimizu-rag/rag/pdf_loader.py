"""PDF -> Page リスト。pdfplumberで本文 + 表を抽出する。

なぜpdfplumber:
    - MITライセンス（AGPLのPyMuPDFを避ける）
    - ページ単位の正確な抽出（ページ番号引用の根拠になる）
    - 純Python（Streamlit Cloud 1GB枠でも動く軽量さ）
    - 表抽出 (extract_tables) が標準で使える

ページテキストの構成:
    [本文]

    ## 表（自動抽出）

    | header1 | header2 | ... |
    | --- | --- | ... |
    | ... |

→ 表もテキストとして検索対象になり、生物学論文の実験データ
  （ホルモン濃度、塩分耐性、体重・体長など）に質問が当たる。

既知の限界:
    - 画像内の文字（OCR）は対象外。スキャンPDFは別途。
    - 図そのものは取れない（キャプションは extract_text に含まれる）。
    - 2段組レイアウトは pdfplumber の reading-order 推定に依存。
      崩れる場合は config.PDF_X_TOLERANCE/PDF_Y_TOLERANCE で調整可能。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pdfplumber

from .exceptions import PDFLoadError
from .types import Page

# pdfplumber の extract_text に渡すデフォルトのトレランス。
# 2段組PDFで読み順が崩れる場合、環境変数で調整可能。
import os

_X_TOLERANCE = float(os.getenv("PDF_X_TOLERANCE", "3"))
_Y_TOLERANCE = float(os.getenv("PDF_Y_TOLERANCE", "3"))


def load_pdf(pdf_path: Path) -> list[Page]:
    """PDFを読み、ページごとに Page を返す。空ページはスキップ。

    各ページのテキストには本文と、抽出された表（Markdown化）が
    両方含まれる。失敗時は PDFLoadError を投げる。
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
                text = _extract_page_content(page)
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


def _extract_page_content(page: Any) -> str:
    """1ページから本文と表を取り出して結合。"""
    body = _safe_extract_text(page).strip()
    tables_md = _extract_tables_as_markdown(page)
    parts: list[str] = []
    if body:
        parts.append(body)
    if tables_md:
        parts.append("## 表（自動抽出）\n\n" + "\n\n".join(tables_md))
    return "\n\n".join(parts).strip()


def _safe_extract_text(page: Any) -> str:
    """ページ単位の抽出失敗を握り潰し、可能な限り続行する。"""
    try:
        return page.extract_text(
            x_tolerance=_X_TOLERANCE, y_tolerance=_Y_TOLERANCE
        ) or ""
    except Exception:
        try:
            return page.extract_text() or ""
        except Exception:
            return ""


def _extract_tables_as_markdown(page: Any) -> list[str]:
    """ページ内の表を Markdown 文字列のリストとして返す。"""
    try:
        tables = page.extract_tables() or []
    except Exception:
        return []
    out: list[str] = []
    for tbl in tables:
        md = table_to_markdown(tbl)
        if md:
            out.append(md)
    return out


def table_to_markdown(table: Iterable[Iterable[Any]] | None) -> str:
    """pdfplumber の表（list[list[str | None]]）を Markdown 文字列に変換。

    ノイズ除去:
        - 行数 < 2 または 列数 < 2 の表は捨てる（フォーム枠などを除外）
        - 全セルが空の行は捨てる
    エスケープ:
        - None → 空文字
        - セル内改行 → 半角スペース
        - セル内 `|` → `\\|`
    """
    if not table:
        return ""
    rows: list[list[str]] = []
    for raw_row in table:
        if not raw_row:
            continue
        row = [_clean_cell(c) for c in raw_row]
        rows.append(row)
    if len(rows) < 2:
        return ""
    max_cols = max(len(r) for r in rows)
    if max_cols < 2:
        return ""
    # 列数を揃え、全セル空の行を除外
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    rows = [r for r in rows if any(cell for cell in r)]
    if len(rows) < 2:
        return ""
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _clean_cell(cell: Any) -> str:
    if cell is None:
        return ""
    s = str(cell).replace("\n", " ").replace("|", "\\|").strip()
    return s


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

"""pdf_loader の純粋ロジック部分（PDF不要部分）のテスト。

PDF パース自体は pdfplumber の責務。ここでは：
    - table_to_markdown のエッジケース
    - _clean_cell の入力種別ハンドリング
    - PDFLoadError が想定どおり投げられること
を検証する。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.exceptions import PDFLoadError
from rag.pdf_loader import _clean_cell, load_pdf, table_to_markdown


def test_table_to_markdown_basic() -> None:
    table = [
        ["Treatment", "GH (ng/mL)", "Notes"],
        ["Control", "1.2", "n=10"],
        ["IGF-1", "3.4", "n=10"],
    ]
    md = table_to_markdown(table)
    lines = md.splitlines()
    assert lines[0] == "| Treatment | GH (ng/mL) | Notes |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| Control | 1.2 | n=10 |"
    assert lines[3] == "| IGF-1 | 3.4 | n=10 |"


def test_table_to_markdown_handles_none_cells() -> None:
    table = [["a", "b", None], [None, "y", "z"]]
    md = table_to_markdown(table)
    assert "| a | b |  |" in md
    assert "|  | y | z |" in md


def test_table_to_markdown_normalizes_newlines() -> None:
    table = [["col1", "col2"], ["multi\nline", "value"]]
    md = table_to_markdown(table)
    assert "multi line" in md
    assert "\nline" not in md.replace("---\n", "").replace(" |\n", "")


def test_table_to_markdown_escapes_pipe() -> None:
    table = [["a", "b"], ["x|y", "z"]]
    md = table_to_markdown(table)
    assert "x\\|y" in md


def test_table_to_markdown_pads_short_rows() -> None:
    table = [["a", "b", "c"], ["x"]]  # 不揃い
    md = table_to_markdown(table)
    # 列数3に揃えられる
    assert "| x |  |  |" in md


def test_table_to_markdown_rejects_single_row() -> None:
    assert table_to_markdown([["only one"]]) == ""
    assert table_to_markdown([["a", "b", "c"]]) == ""


def test_table_to_markdown_rejects_single_column() -> None:
    assert table_to_markdown([["a"], ["b"]]) == ""


def test_table_to_markdown_rejects_empty_input() -> None:
    assert table_to_markdown(None) == ""
    assert table_to_markdown([]) == ""


def test_table_to_markdown_drops_all_empty_rows() -> None:
    table = [["a", "b"], ["", ""], ["x", "y"]]
    md = table_to_markdown(table)
    assert "| x | y |" in md
    # 空行は出力に含まれない
    assert "|  |  |" not in md


def test_clean_cell_handles_various_inputs() -> None:
    assert _clean_cell(None) == ""
    assert _clean_cell("") == ""
    assert _clean_cell("  hello  ") == "hello"
    assert _clean_cell("a\nb") == "a b"
    assert _clean_cell("a|b") == "a\\|b"
    assert _clean_cell(42) == "42"  # 数値も文字列化


def test_load_pdf_missing_file_raises() -> None:
    with pytest.raises(PDFLoadError, match="見つかりません"):
        load_pdf(Path("/no/such/file.pdf"))


def test_load_pdf_empty_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(PDFLoadError, match="空ファイル"):
        load_pdf(empty)


def test_load_pdf_corrupted_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not actually a pdf")
    with pytest.raises(PDFLoadError):
        load_pdf(bad)

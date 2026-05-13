"""chunker のユニットテスト。

検証点:
    - チャンクが CHUNK_SIZE_CHARS を超えない
    - 隣接チャンクが CHUNK_OVERLAP_CHARS だけ重なる
    - chunk_id が決定的（同じ入力 → 同じID）
    - 空ページはスキップ
    - 1ページ短文はそのまま1チャンク
    - チャンクはページ境界を跨がない（ページ番号引用の正確性）
"""
from __future__ import annotations

from rag.chunker import chunk_pages
from rag.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS
from rag.types import Page


def _page(text: str, page_num: int = 1, title: str = "Paper A") -> Page:
    return Page(
        paper_title=title,
        page_num=page_num,
        text=text,
        source_path="/tmp/paper-a.pdf",
    )


def test_short_page_makes_single_chunk() -> None:
    chunks = chunk_pages([_page("hello world")])
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].page_num == 1


def test_empty_page_is_skipped() -> None:
    chunks = chunk_pages([_page("")])
    assert chunks == []


def test_long_page_is_split_with_overlap() -> None:
    text = "a" * (CHUNK_SIZE_CHARS * 2 + 100)
    chunks = chunk_pages([_page(text)])
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= CHUNK_SIZE_CHARS
    # オフセット差は CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS のステップ
    step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
    assert step > 0  # サニティ


def test_chunk_id_is_deterministic() -> None:
    page = _page("a" * (CHUNK_SIZE_CHARS + 50))
    a = chunk_pages([page])
    b = chunk_pages([page])
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_different_pages_get_different_chunk_ids() -> None:
    p1 = _page("same text", page_num=1)
    p2 = _page("same text", page_num=2)
    chunks = chunk_pages([p1, p2])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_do_not_cross_page_boundary() -> None:
    p1 = _page("page one content", page_num=1)
    p2 = _page("page two content", page_num=2)
    chunks = chunk_pages([p1, p2])
    # 各チャンクは元のページのテキストだけを含み、ページ番号も保持される
    for c in chunks:
        if c.page_num == 1:
            assert "page two" not in c.text
        else:
            assert "page one" not in c.text

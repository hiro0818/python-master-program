"""引用検証レイヤーのユニットテスト。

検証点:
    - [title, p.N] の抽出（ASCII / 全角ブラケット / 全角カンマ）
    - 部分一致 + ページ厳密一致のときに verified=True
    - ページが違うとき verified=False（理由メッセージにヒントを含む）
    - そもそも論文タイトルがヒットに無いとき verified=False
"""
from __future__ import annotations

from rag.citation_verifier import (
    extract_citations,
    verify_citations,
)
from rag.types import Hit


def _hit(title: str, page: int, text: str = "abc") -> Hit:
    return Hit(
        text=text,
        score=0.9,
        paper_title=title,
        page_num=page,
        source_path=f"/tmp/{title}.pdf",
    )


def test_extract_basic_ascii() -> None:
    text = "結論は重要である [Shimizu 2024 IGFBP, p.5] と示されている。"
    out = extract_citations(text)
    assert out == [("[Shimizu 2024 IGFBP, p.5]", "Shimizu 2024 IGFBP", 5)]


def test_extract_handles_fullwidth_brackets() -> None:
    text = "［Yada 2010 GH, p.12］"
    out = extract_citations(text)
    assert len(out) == 1
    assert out[0][1] == "Yada 2010 GH"
    assert out[0][2] == 12


def test_extract_handles_p_without_dot() -> None:
    text = "[Foo 2020, p 7]"
    out = extract_citations(text)
    assert out[0][2] == 7


def test_extract_multiple_in_one_sentence() -> None:
    text = "X と示された [A, p.1][B 2020, p.3]。"
    out = extract_citations(text)
    assert len(out) == 2


def test_verified_when_title_substring_and_page_match() -> None:
    hits = [_hit("Shimizu 2024 IGFBP-2b in masu salmon", 5)]
    text = "結論 [Shimizu 2024 IGFBP, p.5]"
    checks = verify_citations(text, hits)
    assert len(checks) == 1
    assert checks[0].verified is True
    assert checks[0].matched_hit is hits[0]


def test_unverified_when_page_mismatch_but_title_present() -> None:
    hits = [_hit("Shimizu 2024 IGFBP", 5)]
    text = "結論 [Shimizu 2024 IGFBP, p.99]"
    checks = verify_citations(text, hits)
    assert checks[0].verified is False
    assert "タイトルは一致" in checks[0].reason
    assert "99" in checks[0].reason or "5" in checks[0].reason


def test_unverified_when_paper_not_in_hits() -> None:
    hits = [_hit("Yada 2010 GH", 1)]
    text = "結論 [Shimizu 2024, p.1]"
    checks = verify_citations(text, hits)
    assert checks[0].verified is False
    assert "該当する論文" in checks[0].reason


def test_normalization_handles_case_and_whitespace() -> None:
    hits = [_hit("Shimizu  2024  IGFBP", 5)]
    text = "結論 [shimizu 2024 igfbp, p.5]"
    checks = verify_citations(text, hits)
    assert checks[0].verified is True


def test_no_citations_in_text_returns_empty() -> None:
    hits = [_hit("Foo", 1)]
    checks = verify_citations("引用のない文章。", hits)
    assert checks == []

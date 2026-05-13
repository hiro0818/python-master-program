"""LLMの回答に含まれる `[論文タイトル, p.N]` を、検索ヒットの (paper_title, page_num)
と照合する後段バリデータ。

このレイヤーが無いと、LLMがコンテキストに無いページ番号を捏造しても気づけない。
ハルシネーション対策の最後の砦。

使い方:
    checks = verify_citations(answer_text, hits)
    unverified = [c for c in checks if not c.verified]
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from .types import Hit


# `[ 任意のタイトル , p.数字 ]` を抽出。全角ブラケット/カンマも許容。
# 例: [Shimizu 2024 IGFBP, p.5]、 ［Shimizu 2024, p.12］
_CITATION_RE = re.compile(
    r"[\[［]\s*"
    r"(?P<title>[^,，\]］]+?)"
    r"\s*[,，]\s*"
    r"p\.?\s*(?P<page>\d+)"
    r"\s*[\]］]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationCheck:
    """1つの引用 [title, p.N] の検証結果。"""

    raw: str
    cited_title: str
    cited_page: int
    verified: bool
    matched_hit: Optional[Hit]
    reason: str  # 検証結果の人間可読な説明


def _normalize(text: str) -> str:
    """部分一致比較のための正規化。NFKC + lowercase + 連続空白の畳み込み。"""
    norm = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", norm).strip()


def extract_citations(text: str) -> list[tuple[str, str, int]]:
    """回答テキストから (raw, title, page) のリストを抽出。重複は保持。"""
    out: list[tuple[str, str, int]] = []
    for m in _CITATION_RE.finditer(text):
        title = m.group("title").strip()
        try:
            page = int(m.group("page"))
        except ValueError:
            continue
        out.append((m.group(0), title, page))
    return out


def verify_citations(answer_text: str, hits: list[Hit]) -> list[CitationCheck]:
    """回答中の各引用について、ヒット集合に同じ (タイトル部分一致, ページ一致) があるか検証。

    タイトルは正規化（NFKC + lower + 空白圧縮）後の双方向部分一致でマッチ判定。
    ページは厳密一致。
    """
    checks: list[CitationCheck] = []
    norm_hits = [(_normalize(h.paper_title), h) for h in hits]

    for raw, cited_title, cited_page in extract_citations(answer_text):
        cited_norm = _normalize(cited_title)
        match: Optional[Hit] = None
        for hit_norm, hit in norm_hits:
            title_ok = (cited_norm in hit_norm) or (hit_norm in cited_norm)
            if title_ok and hit.page_num == cited_page:
                match = hit
                break

        if match is not None:
            checks.append(
                CitationCheck(
                    raw=raw,
                    cited_title=cited_title,
                    cited_page=cited_page,
                    verified=True,
                    matched_hit=match,
                    reason="検索ヒットと一致",
                )
            )
            continue

        # タイトル一致するヒットがあるなら、ページのみ不一致
        title_only_match = next(
            (h for nh, h in norm_hits if (cited_norm in nh) or (nh in cited_norm)),
            None,
        )
        if title_only_match is not None:
            reason = (
                f"タイトルは一致するが p.{cited_page} は検索ヒットに無い"
                f"（実在: p.{title_only_match.page_num}）"
            )
        else:
            reason = "該当する論文が検索ヒットに無い"

        checks.append(
            CitationCheck(
                raw=raw,
                cited_title=cited_title,
                cited_page=cited_page,
                verified=False,
                matched_hit=None,
                reason=reason,
            )
        )
    return checks

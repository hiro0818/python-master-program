"""検索評価ハーネス: Recall@K を計算する。

設計の意図:
    - 「BGE-M3 + chunk 1000字 + Top-K=8 が清水研の論文に最適か」を定量的に検証する。
    - 評価セット（eval/questions.yaml）を変更しないで、設定（埋め込みモデル、TOP_K、
      CHUNK_SIZE_CHARS）を振って差分を見る運用を想定。

使い方:
    python eval/evaluate.py                  # デフォルトの top-k で評価
    python eval/evaluate.py --top-k 12       # K を上書き
    python eval/evaluate.py --questions other.yaml

PyYAML が必要（requirements.txt 参照）。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# eval/ から python eval/evaluate.py で実行されたとき shimizu-rag/ を import path に。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from rag.config import TOP_K  # noqa: E402
from rag.types import Hit  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402


@dataclass
class EvalQuestion:
    id: str
    question: str
    expected: list[dict]  # [{"title_substring": str, "page": int}, ...]


def load_questions(path: Path) -> list[EvalQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: list[EvalQuestion] = []
    for entry in raw.get("questions", []):
        out.append(
            EvalQuestion(
                id=str(entry["id"]),
                question=str(entry["question"]),
                expected=list(entry.get("expected", [])),
            )
        )
    return out


def hit_matches_expected(hit: Hit, expected: dict[str, Any]) -> bool:
    needle = str(expected.get("title_substring", "")).lower().strip()
    page = expected.get("page")
    if not needle:
        return False
    title_ok = needle in hit.paper_title.lower()
    page_ok = (page is None) or (int(page) == hit.page_num)
    return title_ok and page_ok


def recall_for_question(hits: list[Hit], expected_list: list[dict]) -> float:
    """期待エントリのうち、いくつが Top-K の中に入っていたか（0.0〜1.0）。"""
    if not expected_list:
        return 0.0
    matched = 0
    for exp in expected_list:
        if any(hit_matches_expected(h, exp) for h in hits):
            matched += 1
    return matched / len(expected_list)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recall@K を計算")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).parent / "questions.yaml",
        help="評価用YAMLへのパス",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"検索のTop-K（デフォルト: config.TOP_K={TOP_K}）",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print(f"⚠️  {args.questions} に質問がありません。")
        return 1

    store = VectorStore()
    if store.count() == 0:
        print("⚠️  DBが空です。まず ingest を実行してください。")
        return 1

    print(f"🔎 {len(questions)} 問 / Top-K={args.top_k} で評価中...\n")
    recalls: list[float] = []
    for q in questions:
        hits = store.search(q.question, top_k=args.top_k)
        r = recall_for_question(hits, q.expected)
        recalls.append(r)
        marker = "✅" if r == 1.0 else ("⚠️" if r > 0 else "❌")
        print(f"  {marker} [{q.id}] Recall={r:.2f}  Q={q.question}")
        if r < 1.0:
            for exp in q.expected:
                hit_in = any(hit_matches_expected(h, exp) for h in hits)
                print(
                    f"      期待: title~='{exp.get('title_substring')}' "
                    f"p={exp.get('page')} → {'入った' if hit_in else '入らず'}"
                )

    mean = sum(recalls) / len(recalls)
    print()
    print(f"=== 集計 ===")
    print(f"  質問数:      {len(recalls)}")
    print(f"  Top-K:       {args.top_k}")
    print(f"  平均 Recall: {mean:.3f}")
    print(f"  完全一致率:  {sum(1 for r in recalls if r == 1.0) / len(recalls):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

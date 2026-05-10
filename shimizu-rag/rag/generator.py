"""Claude API の隔離レイヤー。Anthropic SDK を使うのは『このファイルだけ』。

将来 OpenAI / Gemini / ローカルLLM に乗り換えるときは、このファイルだけ書き換える。
入出力（query, hits → Answer）は不変。
実装は次のコミットで埋める。
"""
from __future__ import annotations

from .types import Answer, Hit


def answer_question(query: str, hits: list[Hit]) -> Answer:
    raise NotImplementedError("Phase 2 で実装")

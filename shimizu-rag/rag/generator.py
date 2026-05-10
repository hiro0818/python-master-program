"""Claude API の隔離レイヤー。Anthropic SDK を使うのは『このファイルだけ』。

将来 OpenAI / Gemini / ローカルLLM に乗り換えるときは、このファイルだけ書き換える。
入出力 (query, hits) → Answer は不変。
"""
from __future__ import annotations

from anthropic import Anthropic

from .config import LLM_MAX_TOKENS, LLM_MODEL
from .types import Answer, Hit


SYSTEM_PROMPT = """\
あなたは清水研（北海道大学）の研究内容に詳しい研究アシスタントです。
ユーザーの質問に対し、提供された論文の抜粋（コンテキスト）のみを根拠に
**日本語で**回答してください。

# 必須ルール
1. **根拠が抜粋に含まれない場合は、必ず「提供された論文には記載がありません」と答えてください。**
   推測や一般知識での回答は禁止です。
2. **すべての主張に引用を付けてください**。引用形式は `[論文タイトル, p.ページ番号]` です。
   一文に複数の根拠がある場合は複数の引用を併記してください。
3. **英語の論文を引用する場合でも、回答本文は日本語**で書いてください。
4. 専門用語が出てきたら、文脈から分かる範囲で簡潔に補足してください。
5. 回答の最後に「## 参照した論文」セクションを付け、引用した論文をリストアップしてください。
"""


def _build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(
            f"[抜粋 {i} / 論文「{h.paper_title}」, p.{h.page_num}]\n{h.text}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(query: str, hits: list[Hit]) -> Answer:
    """検索済みチャンクを根拠に Claude Opus 4.7 で回答生成。

    Anthropic SDK はこの関数内でのみ使用される。
    """
    client = Anthropic()
    if not hits:
        return Answer(
            text="提供された論文には関連する記載が見つかりませんでした。",
            citations=[],
        )

    context = _build_context(hits)
    user_message = (
        f"# コンテキスト（論文抜粋）\n{context}\n\n"
        f"# ユーザーの質問\n{query}\n\n"
        "上記のコンテキストのみを根拠に、ルールに従って日本語で回答してください。"
    )

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        thinking={"type": "adaptive"},
        # システムプロンプトはほぼ固定 → cache_control で 2回目以降を安く
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    text = next(
        (block.text for block in response.content if block.type == "text"),
        "",
    )
    return Answer(text=text, citations=hits)

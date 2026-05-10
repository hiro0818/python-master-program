"""Streamlit Web UI: 清水研RAG のチャット画面。

使い方:
    streamlit run app.py

オプション:
    .env に APP_PASSWORD を設定すると、起動時にパスワード入力を要求する。
    未設定ならパスワードゲートは無効（ローカル単独利用向け）。
"""
from __future__ import annotations

import hmac
import os

import streamlit as st
from dotenv import load_dotenv

from rag.config import LLM_MODEL, EMBEDDING_MODEL_NAME, TOP_K
from rag.pipeline import answer
from rag.vectorstore import VectorStore

load_dotenv()


@st.cache_resource
def get_store() -> VectorStore:
    """VectorStoreを1度だけ生成（モデルロードは重いため）。"""
    return VectorStore()


def _check_password() -> bool:
    """APP_PASSWORD が設定されていれば認証を要求。

    タイミング攻撃を避けるため hmac.compare_digest を使用。
    未設定（空文字含む）なら常に通す。
    """
    expected = os.getenv("APP_PASSWORD", "").strip()
    if not expected:
        return True

    if st.session_state.get("password_ok"):
        return True

    st.title("🔒 清水研RAG ─ パスワード")
    pw = st.text_input("パスワードを入力してください", type="password")
    if not pw:
        st.stop()
    if hmac.compare_digest(pw, expected):
        st.session_state.password_ok = True
        st.rerun()
    else:
        st.error("パスワードが違います。")
        st.stop()
    return False  # unreachable


def main() -> None:
    st.set_page_config(page_title="清水研RAG", page_icon="🐟", layout="wide")
    if not _check_password():
        return
    st.title("🐟 清水研RAG")
    st.caption(
        "清水研（北大）の論文に基づいて質問に答えます。回答には必ず引用が付きます。"
        "根拠がないときは「わかりません」と答えます。"
    )

    # サイドバー: ステータス表示
    with st.sidebar:
        st.subheader("📊 ステータス")
        store = get_store()
        st.metric("DB内チャンク数", store.count())
        st.markdown(f"**生成モデル**: `{LLM_MODEL}`")
        st.markdown(f"**埋め込みモデル**: `{EMBEDDING_MODEL_NAME}`")
        st.markdown(f"**Top-K**: {TOP_K}")
        st.divider()
        st.markdown("論文を追加するには、`papers/` にPDFを置いて `python ingest.py` を実行。")

    # チャット履歴
    if "history" not in st.session_state:
        st.session_state.history = []

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant" and entry.get("citations"):
                _render_citations(entry["citations"])

    # 入力
    query = st.chat_input("質問を入力（例: サクラマスのスモルト化に関わるホルモンは？）")
    if not query:
        return

    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("検索 + 回答生成中..."):
            try:
                result = answer(query)
            except Exception as e:
                st.error(f"エラー: {type(e).__name__}: {e}")
                return

            st.markdown(result.text)
            if result.citations:
                _render_citations(result.citations)
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": result.text,
                    "citations": result.citations,
                }
            )


def _render_citations(citations: list) -> None:
    with st.expander(f"📄 検索された抜粋 ({len(citations)} 件)"):
        for i, h in enumerate(citations, 1):
            st.markdown(
                f"**抜粋 {i}**: 「{h.paper_title}」, p.{h.page_num}  "
                f"(score={h.score:.3f})"
            )
            preview = h.text[:500] + ("…" if len(h.text) > 500 else "")
            st.text(preview)
            st.caption(f"📁 {h.source_path}")
            st.divider()


if __name__ == "__main__":
    main()

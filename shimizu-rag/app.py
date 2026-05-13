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

from rag.citation_verifier import CitationCheck, verify_citations
from rag.config import EMBEDDING_MODEL_NAME, LLM_MODEL, TOP_K
from rag.exceptions import RAGError
from rag.pipeline import answer_stream, ingest_url
from rag.registry import list_papers
from rag.types import Hit
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

    _render_sidebar()

    # チャット履歴
    if "history" not in st.session_state:
        st.session_state.history = []

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                if entry.get("citation_checks"):
                    _render_citation_badges(entry["citation_checks"])
                if entry.get("citations"):
                    _render_citations(entry["citations"])

    # 入力
    query = st.chat_input("質問を入力（例: サクラマスのスモルト化に関わるホルモンは？）")
    if not query:
        return

    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("検索中..."):
                stream, hits = answer_stream(query)
            full_text = st.write_stream(stream)
        except RAGError as e:
            st.error(f"❌ {e}")
            return
        except Exception as e:
            st.error(f"想定外エラー: {type(e).__name__}: {e}")
            return

        checks = verify_citations(full_text, hits)
        _render_citation_badges(checks)
        if hits:
            _render_citations(hits)
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": full_text,
                "citations": hits,
                "citation_checks": checks,
            }
        )


def _render_sidebar() -> None:
    with st.sidebar:
        st.subheader("📊 ステータス")
        store = get_store()
        st.metric("DB内チャンク数", store.count())
        st.markdown(f"**生成モデル**: `{LLM_MODEL}`")
        st.markdown(f"**埋め込みモデル**: `{EMBEDDING_MODEL_NAME}`")
        st.markdown(f"**Top-K**: {TOP_K}")
        st.divider()

        st.subheader("➕ URLから論文を追加")
        with st.form("url_ingest_form", clear_on_submit=True):
            url_input = st.text_input(
                "PDFのURL",
                placeholder="https://example.com/paper.pdf",
                help="直リンクPDFのみ対応。著作権・利用規約に注意してください。",
            )
            submit = st.form_submit_button("取り込む")
        if submit and url_input:
            with st.spinner("ダウンロード + 取り込み中…"):
                try:
                    filename, n_chunks = ingest_url(url_input.strip())
                    st.success(f"✅ {filename}: {n_chunks} 新規チャンク")
                    get_store.clear()
                    st.rerun()
                except RAGError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ 想定外エラー: {type(e).__name__}: {e}")

        st.divider()
        st.subheader("📚 取り込み済み論文")
        try:
            papers = list_papers()
        except RAGError as e:
            st.warning(f"レジストリ読み込み失敗: {e}")
            papers = []
        if not papers:
            st.caption("まだありません。")
        else:
            for p in papers:
                with st.expander(f"📄 {p.paper_title}"):
                    st.markdown(f"- ページ数: {p.n_pages}")
                    st.markdown(f"- チャンク数（累積）: {p.n_chunks}")
                    st.markdown(f"- 取り込み日時: `{p.ingested_at}`")
                    if p.source_url:
                        st.markdown(f"- URL: {p.source_url}")
                    st.caption(f"📁 `{p.source_path}`")

        st.divider()
        st.caption(
            "ローカルPDFは `papers/` に置いて `python ingest.py` を実行することでも追加できます。"
        )


def _render_citation_badges(checks: list[CitationCheck]) -> None:
    if not checks:
        return
    verified = sum(1 for c in checks if c.verified)
    total = len(checks)
    if verified == total:
        st.success(f"✅ 引用検証: {verified}/{total} 一致")
        return
    st.warning(f"⚠️ 引用検証: {verified}/{total} 一致 ─ 未検証の引用があります")
    with st.expander("未検証の引用一覧"):
        for c in checks:
            if c.verified:
                continue
            st.markdown(f"- `{c.raw}` ─ {c.reason}")


def _render_citations(citations: list[Hit]) -> None:
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

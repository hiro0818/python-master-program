"""CLI: papers/ 配下のPDFを取り込んでLanceDBに保存する。

使い方:
    python ingest.py
"""
from __future__ import annotations

from rag.config import PAPERS_DIR
from rag.pipeline import ingest_all


def main() -> None:
    pdfs = list(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"⚠️  {PAPERS_DIR} に PDF がありません。")
        print("   論文PDFを papers/ に置いてから再実行してください。")
        return

    print(f"📚 {len(pdfs)} 本のPDFを取り込みます...")
    print("   （初回はBGE-M3モデルのダウンロードで数分かかります）\n")

    results = ingest_all()

    print("\n=== 取り込み結果 ===")
    for name, n in results.items():
        marker = "✅" if n > 0 else "⏭️ "
        print(f"  {marker} {name}: {n} 新規チャンク")

    total_new = sum(results.values())
    print(f"\n完了: {total_new} 新規チャンクをDBに追加しました。")
    print("   次は: streamlit run app.py")


if __name__ == "__main__":
    main()

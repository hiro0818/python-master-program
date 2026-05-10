# 🐟 清水研RAG

清水研究室（北海道大学）の論文に基づいて質問に答えるRAG（Retrieval-Augmented Generation）システム。

## 目的

- 研究室の論文を横断検索し、**引用付き**で日本語回答を返す
- 配属前のキャッチアップ、ゼミ準備、新規実験の文献調査を高速化する
- 根拠がないときは「わかりません」と答える（ハルシネーション抑制）

## 技術スタック（5基準で評価済み: 目的適合・エコシステム・流動性・非ロックイン・5年後の生存）

| レイヤー | 技術 | ライセンス | 役割 |
|---|---|---|---|
| PDF解析 | pdfplumber | MIT | PDF → ページごとのテキスト抽出 |
| 埋め込み | BAAI/bge-m3（via sentence-transformers） | Apache 2.0 | 日英両対応、ローカル実行で無料 |
| ベクターDB | LanceDB | Apache 2.0 | Lance/Parquet互換、フォーマット公開 |
| 生成 | Claude Opus 4.7（Anthropic API） | API | 引用付き回答生成（`rag/generator.py` 内に隔離） |
| UI | Streamlit | Apache 2.0 | チャット風Webアプリ |

設計の詳細・選定理由は議論ログ参照。

## ディレクトリ構成

```
shimizu-rag/
├── README.md
├── requirements.txt
├── .env.example         # ANTHROPIC_API_KEY=
├── app.py               # Streamlit エントリポイント
├── ingest.py            # CLI: papers/ → ベクターDB
├── papers/              # PDFをここに置く（gitignore）
└── rag/
    ├── __init__.py
    ├── types.py         # 不変インターフェース（dataclass）
    ├── config.py        # パラメータ集約
    ├── pdf_loader.py    # PDF → Page
    ├── chunker.py       # Page → Chunk
    ├── vectorstore.py   # Chunk → LanceDB / 検索
    ├── generator.py     # Claude API 隔離
    └── pipeline.py      # 高レベルAPI: ingest_all() / answer()
```

## ステータス

- [x] **Phase 1 (土台)**: ディレクトリ構成・設定・インターフェース型・スタブ
- [ ] **Phase 2 (実装)**: 各モジュールの中身
- [ ] **Phase 3 (改善)**: 検索フィルタ、履歴保存、論文一覧画面

## セットアップ（Phase 2 完了後に有効）

```bash
cd shimizu-rag
python -m venv .venv
source .venv/bin/activate          # Windowsなら .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# .env の ANTHROPIC_API_KEY に実際のキーを書く

# PDFを papers/ に置く
# 例: papers/Shimizu2024_IGFBP.pdf

# 取り込み（初回のみ）
python ingest.py

# 起動
streamlit run app.py
```

## 設計原則

1. **インターフェースは安定、実装は交換可能に** — `types.py` の dataclass が契約
2. **ライセンスは MIT / Apache 2.0 で揃える** — AGPL や独自ライセンスを避ける
3. **データフォーマットは公開仕様のものを選ぶ** — LanceDB（Parquet互換）、SQLite、JSON
4. **流行に乗らず、コモディティを選ぶ** — 埋め込みモデル・LLM プロバイダーは交換前提
5. **UIロジックとビジネスロジックを分離** — `app.py` は表示だけ、`rag/` がコア

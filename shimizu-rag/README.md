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
- [x] **Phase 2 (実装)**: 全モジュールの中身（PDCA で検証済み）
- [ ] **Phase 3 (改善)**: 検索フィルタ、履歴保存、論文一覧画面、Streaming回答

## セットアップ

### 1. 依存関係のインストール

```bash
cd shimizu-rag
python -m venv .venv
source .venv/bin/activate      # Windowsなら .venv\Scripts\activate
pip install -r requirements.txt
```

> **初回の注意**:
> - BGE-M3 モデルが初回 ingest 時に自動ダウンロードされます（約 568MB、数分）
> - HuggingFace Hub への接続が必要

### 2. APIキー設定

```bash
cp .env.example .env
# .env を開いて ANTHROPIC_API_KEY=sk-ant-... を実際のキーに書き換える
```

APIキーは https://console.anthropic.com/ から取得できます。

### 3. 論文を置く

```bash
# papers/ にPDFをコピー
cp ~/Downloads/Shimizu2024_IGFBP.pdf papers/
```

### 4. 取り込み（初回のみ／PDF追加時）

```bash
python ingest.py
```

冪等です。同じPDFを再取り込みしても重複保存されません。

### 5. UI 起動

```bash
streamlit run app.py
```

ブラウザが自動で開きます。

## 使い方の例

質問例:
- 「サクラマスのスモルト化に関わる主要なホルモンは？」
- 「IGFBP-2b の機能について、清水研は何を明らかにしたか？」
- 「シロザケの長期海水飼育の意義は？」

回答には必ず `[論文タイトル, p.ページ番号]` 形式の引用が付きます。
根拠が論文中に見つからない場合は「提供された論文には記載がありません」と返ります。

## 設計原則

1. **インターフェースは安定、実装は交換可能に** — `types.py` の dataclass が契約
2. **ライセンスは MIT / Apache 2.0 で揃える** — AGPL や独自ライセンスを避ける
3. **データフォーマットは公開仕様のものを選ぶ** — LanceDB（Parquet互換）、SQLite、JSON
4. **流行に乗らず、コモディティを選ぶ** — 埋め込みモデル・LLM プロバイダーは交換前提
5. **UIロジックとビジネスロジックを分離** — `app.py` は表示だけ、`rag/` がコア

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `ModuleNotFoundError: No module named 'rag'` | `cd shimizu-rag` してから実行する |
| `ANTHROPIC_API_KEY not set` | `.env` を作成しキーを設定 |
| BGE-M3 ダウンロードが遅い | 初回のみ。ネット接続を確認 |
| 検索結果が0件 | `papers/` にPDFを置いて `python ingest.py` を実行 |

## 次にやりたいこと（Phase 3）

- [ ] 質問・回答の履歴保存（SQLite）
- [ ] 論文ごとのフィルタリング（「IGFBP関連の論文だけから探す」）
- [ ] 取り込み済み論文の一覧画面
- [ ] Streamingで回答を逐次表示
- [ ] 引用クリックで該当PDFをプレビュー
- [ ] 簡易テスト（pytest）整備

# 🐟 清水研RAG

清水研究室（北海道大学）の論文に基づいて質問に答えるRAG（Retrieval-Augmented Generation）システム。

## 目的

- 研究室の論文を横断検索し、**引用付き**で日本語回答を返す
- 配属前のキャッチアップ、ゼミ準備、新規実験の文献調査を高速化する
- 根拠がないときは「わかりません」と答える（ハルシネーション抑制）

## 技術スタック（5基準で評価済み: 目的適合・エコシステム・流動性・非ロックイン・5年後の生存）

| レイヤー | 技術 | ライセンス | 役割 |
|---|---|---|---|
| PDF解析 | pdfplumber | MIT | PDF → ページごとのテキスト + 表（Markdown化） |
| 埋め込み | BAAI/bge-m3（via sentence-transformers） | Apache 2.0 | 日英両対応、ローカル実行で無料 |
| ベクターDB | LanceDB | Apache 2.0 | Lance/Parquet互換、フォーマット公開 |
| 生成 | Claude Opus 4.7（Anthropic API） | API | 引用付き回答生成（`rag/generator.py` 内に隔離） |
| UI | Streamlit | Apache 2.0 | チャット風Webアプリ |

## ディレクトリ構成

```
shimizu-rag/
├── README.md
├── requirements.txt          # 本番依存
├── requirements-dev.txt      # 開発依存（pytest）
├── .env.example              # ANTHROPIC_API_KEY=
├── app.py                    # Streamlit エントリポイント
├── ingest.py                 # CLI: papers/ → ベクターDB
├── papers/                   # PDFをここに置く（gitignore）
├── eval/
│   ├── questions.yaml        # 検索評価用の質問セット
│   └── evaluate.py           # Recall@K 計算スクリプト
├── tests/                    # pytest ユニットテスト
└── rag/
    ├── __init__.py
    ├── types.py              # 不変インターフェース（dataclass）
    ├── config.py             # パラメータ集約
    ├── exceptions.py         # 例外階層（RAGError 配下）
    ├── pdf_loader.py         # PDF → Page
    ├── url_loader.py         # URL → ダウンロード → papers/
    ├── chunker.py            # Page → Chunk
    ├── vectorstore.py        # Chunk → LanceDB / 検索
    ├── generator.py          # Claude API 隔離（ストリーミング対応）
    ├── citation_verifier.py  # [title, p.N] とヒットの照合
    ├── registry.py           # 取り込み済み論文の SQLite レジストリ
    └── pipeline.py           # 高レベルAPI: ingest_all/url/answer/answer_stream
```

## ステータス

- [x] **Phase 1 (土台)**: ディレクトリ構成・設定・インターフェース型・スタブ
- [x] **Phase 2 (実装)**: 全モジュールの中身（PDCA で検証済み）
- [x] **Phase 3 (品質強化)**: pytest（54件）、カスタム例外、SQLite論文レジストリ、ストリーミング、引用検証、Recall@K評価ハーネス、**表（Table）抽出**
- [ ] **Phase 4 (運用改善)**: 検索フィルタ、履歴保存、引用クリックでPDFプレビュー、Railway/Render永続化、layout-aware抽出（docling）

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

#### （任意）パスワード保護

ローカル外（ngrok / SSH トンネル / デプロイ等）に公開する場合は、
`.env` に `APP_PASSWORD=好きなパスワード` を設定すれば、
起動時にパスワード入力が要求されます。空のままなら無効です。

### 3. 論文を置く

2通りの方法があります。

**方法A: ローカルPDFを papers/ にコピー**
```bash
cp ~/Downloads/Shimizu2024_IGFBP.pdf papers/
```

**方法B: Streamlit UI からURLで追加**
アプリ起動後、サイドバー「➕ URLから論文を追加」にPDFの直リンクURLを貼り付け。
オープンアクセス論文（Nature OA、bioRxiv、Frontiers、PMC等）に有効。

> **著作権・利用規約に注意**:
> 大学契約越しのpaywall論文URLをローカル蓄積するのは契約違反になりうる場合があります。
> オープンアクセス論文か、自分が合法的に取得済みのPDFを使ってください。

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

## クラウド公開（Streamlit Cloud）

### 1. リポジトリ準備

PRをmainにマージするか、デプロイしたいブランチを決めておく。

### 2. Streamlit Cloud にサインイン

https://share.streamlit.io/ → GitHubアカウントでサインイン

### 3. 新規アプリ作成

「**New app**」→ 以下を入力：

| 項目 | 値 |
|---|---|
| Repository | `hiro0818/python-master-program` |
| Branch | `main`（またはデプロイ対象のブランチ） |
| Main file path | `shimizu-rag/app.py` |
| App URL | お好みのサブドメイン（例: `shimizu-rag`） |

### 4. シークレット設定

「**Advanced settings**」→ Secrets タブに以下を貼る：

```toml
ANTHROPIC_API_KEY = "sk-ant-xxxxxxxxxxxxxx"
APP_PASSWORD = "好きなパスワード"

# 1GB RAMで BGE-M3 が苦しい場合は軽量モデルに差し替え
# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_DIM = 384
```

### 5. Deploy をクリック

数分でビルド完了。`https://<your-app>.streamlit.app/` でアクセス可能になります。
パスワード設定済みなら認証画面が出ます。

### 注意点

- **永続化なし**: コンテナ再起動でDBがリセットされます。論文URLは都度サイドバーから再追加する形になります。
  本格運用にはRailway/Render等の永続ボリュームが必要。
- **メモリ**: BGE-M3で苦しい場合、シークレットの `EMBEDDING_MODEL_NAME` を軽量モデルに切り替え。
- **コールドスタート**: 7日間アクセスがないとスリープ。初回起動は数十秒かかります。

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `ModuleNotFoundError: No module named 'rag'` | `cd shimizu-rag` してから実行する |
| `ANTHROPIC_API_KEY not set` | `.env` を作成しキーを設定 |
| BGE-M3 ダウンロードが遅い | 初回のみ。ネット接続を確認 |
| 検索結果が0件 | `papers/` にPDFを置いて `python ingest.py` を実行 |

## 品質保証

### テスト

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

41 件のユニットテスト：

| ファイル | カバー範囲 |
|---|---|
| `test_chunker.py` | スライディングウィンドウ、決定的 chunk_id、ページ境界の保持 |
| `test_vectorstore_score.py` | L2² → cosine 変換式の単調性・境界値 |
| `test_citation_verifier.py` | `[title, p.N]` 抽出（全角ブラケット含む）、検証ロジック |
| `test_registry.py` | SQLite upsert、URL保持、件数・ソート |
| `test_url_loader.py` | ファイル名サニタイズ、スキーム検証 |
| `test_exceptions.py` | 例外階層、`except RAGError` で全捕捉できること |

ネットワーク・APIキー・モデルダウンロード不要で完結する。

### 検索精度の評価（Recall@K）

`eval/questions.yaml` に「期待される論文・ページ」を書いて：

```bash
python eval/evaluate.py             # config.TOP_K で評価
python eval/evaluate.py --top-k 12  # K を上書きして比較
```

- 設定（埋め込みモデル / TOP_K / CHUNK_SIZE_CHARS）を振って Recall@K の変化を見る
- まず 20 問程度書くと統計的に意味がある
- 配属後に論文を読みながらエントリを足す運用

### 引用検証

LLM が `[論文タイトル, p.5]` と書いたとき、その (タイトル部分一致, ページ厳密一致) の
チャンクが実際に検索ヒットに含まれていたかを後段で検証し、UIにバッジ表示する：

- ✅ 全引用が一致 → 緑バッジ
- ⚠️ 未検証あり → 黄色バッジ + 詳細を expander 表示

ハルシネーションがプロンプト層をすり抜けても、ここで気付ける。

### PDF表抽出

`pdfplumber.extract_tables()` でページ内の表を抽出し、Markdown 形式に変換して
ページテキストに統合する。生物学論文の実験データ表（ホルモン濃度、塩分耐性、
体重・体長、サンプル数 N）が検索対象になる：

```
（本文…）

## 表（自動抽出）

| Treatment | GH (ng/mL) | Notes |
| --- | --- | --- |
| Control | 1.2 | n=10 |
| IGF-1 | 3.4 | n=10 |
```

2段組の reading order が崩れるときは環境変数で調整できる：

```bash
export PDF_X_TOLERANCE=2  # デフォルト 3
export PDF_Y_TOLERANCE=2
```

> **既知の限界**:
> - 画像内の文字（OCR）は対象外。スキャンPDFは別途処理が必要。
> - 図そのものは取れない（キャプションは extract_text に含まれる）。
> - layout-aware な抽出が必要なら Phase 4 で `docling` (MIT) を検討。
>   モデル数百MBが必要なので Streamlit Cloud 1GB 枠だと厳しい。

## 次にやりたいこと（Phase 4）

- [ ] 質問・回答の履歴保存（既存の `papers.sqlite` に history テーブル追加）
- [ ] 論文ごとのフィルタリング（「IGFBP関連の論文だけから探す」）
- [ ] 引用クリックで該当PDFをプレビュー
- [ ] PDF抽出を `unstructured` or `marker` に切替（図表・2段組対応）
- [ ] Railway/Render に移して永続ボリューム化
- [ ] 検索精度のドメイン特化（PubMedBERT / SciBERT 比較）

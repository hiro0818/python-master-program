"""清水研RAGの設定。パラメータを変更するときはここだけ触る。

環境変数で上書き可能（クラウドデプロイ時の調整用）:
    EMBEDDING_MODEL_NAME, EMBEDDING_DIM, TOP_K, LLM_MODEL, LLM_MAX_TOKENS
"""
from __future__ import annotations

import os
from pathlib import Path

# プロジェクトのベースディレクトリ（このファイルから見て一つ上の親）
BASE_DIR = Path(__file__).resolve().parent.parent

# データの置き場所
PAPERS_DIR = BASE_DIR / "papers"
DB_DIR = BASE_DIR / "db"

# 埋め込みモデル（環境変数で差し替え可能：クラウドなど省メモリ環境向け）
# 例: EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#     EMBEDDING_DIM=384
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# チャンキング（1チャンクの長さと、隣接チャンクとの重複量、文字数）
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200

# 検索で取得する上位件数
TOP_K = int(os.getenv("TOP_K", "8"))

# 生成LLM
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-4-7")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# LanceDBのテーブル名
TABLE_NAME = "chunks"

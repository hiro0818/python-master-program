"""清水研RAGの設定。パラメータを変更するときはここだけ触る。"""
from __future__ import annotations

from pathlib import Path

# プロジェクトのベースディレクトリ（このファイルから見て一つ上の親）
BASE_DIR = Path(__file__).resolve().parent.parent

# データの置き場所
PAPERS_DIR = BASE_DIR / "papers"
DB_DIR = BASE_DIR / "db"

# 埋め込みモデル（差し替えたいときはこの2行）
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# チャンキング（1チャンクの長さと、隣接チャンクとの重複量、文字数）
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200

# 検索で取得する上位件数
TOP_K = 8

# 生成LLM
LLM_MODEL = "claude-opus-4-7"
LLM_MAX_TOKENS = 4096

# LanceDBのテーブル名
TABLE_NAME = "chunks"

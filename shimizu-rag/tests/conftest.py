"""pytest 共通設定。

shimizu-rag/ をimport pathに入れて、`from rag.xxx import ...` を可能にする。
ネットワーク・LLMキー・PDFモデル等の重い依存は触らない（ユニットテストは軽量に保つ）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

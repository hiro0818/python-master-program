"""LanceDB ラッパー。チャンク保存とベクター検索を提供する。

埋め込みモデル: BAAI/bge-m3（sentence-transformers経由）
ストレージ: LanceDB（Lance/Parquet互換、Apache 2.0、ローカル永続化）

設計:
    - 同じ chunk_id を複数回追加しても重複保存しない（idempotent ingest）
    - normalize_embeddings=True で正規化済みベクターを保存し、
      cosine相当の類似度で検索する
    - search() の戻り値 score は distance_to_score() で cosine 相当に変換
"""
from __future__ import annotations

from typing import Any

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

from .config import (
    DB_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    TABLE_NAME,
    TOP_K,
)
from .exceptions import EmbeddingError
from .types import Chunk, Hit


def distance_to_score(distance: float) -> float:
    """LanceDB の L2² 距離を cosine 相当の類似度 [0, 1] に変換。

    正規化済みベクターのとき ||a-b||² = 2(1 - cos_sim) なので
    score = 1 - distance/2 が cosine_similarity と一致する。
    分かりやすさのため 0..1 にクランプ。
    """
    score = 1.0 - distance / 2.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


class VectorStore:
    def __init__(self) -> None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(DB_DIR))
        try:
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            raise EmbeddingError(
                f"埋め込みモデル '{EMBEDDING_MODEL_NAME}' のロードに失敗しました: "
                f"{type(e).__name__}: {e}"
            ) from e
        self._table = self._open_or_create_table()

    def _open_or_create_table(self) -> Any:
        if TABLE_NAME in self._db.table_names():
            return self._db.open_table(TABLE_NAME)
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
                pa.field("text", pa.string()),
                pa.field("paper_title", pa.string()),
                pa.field("page_num", pa.int32()),
                pa.field("source_path", pa.string()),
                pa.field("chunk_id", pa.string()),
            ]
        )
        return self._db.create_table(TABLE_NAME, schema=schema)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """チャンクを埋め込んで保存。既存IDはスキップ。追加件数を返す。"""
        if not chunks:
            return 0
        existing = self._existing_chunk_ids({c.chunk_id for c in chunks})
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            return 0
        texts = [c.text for c in new_chunks]
        try:
            vectors = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
        except Exception as e:
            raise EmbeddingError(
                f"埋め込み計算に失敗しました: {type(e).__name__}: {e}"
            ) from e

        if vectors and len(vectors[0]) != EMBEDDING_DIM:
            raise EmbeddingError(
                f"埋め込み次元の不一致: モデル出力 {len(vectors[0])} 次元 / "
                f"設定 EMBEDDING_DIM={EMBEDDING_DIM}。"
                "config.py または環境変数 EMBEDDING_DIM を見直してください。"
            )

        rows = [
            {
                "vector": vec,
                "text": c.text,
                "paper_title": c.paper_title,
                "page_num": c.page_num,
                "source_path": c.source_path,
                "chunk_id": c.chunk_id,
            }
            for c, vec in zip(new_chunks, vectors)
        ]
        self._table.add(rows)
        return len(new_chunks)

    def _existing_chunk_ids(self, candidates: set[str]) -> set[str]:
        if not candidates or self._table.count_rows() == 0:
            return set()
        arrow_tbl = self._table.to_arrow()
        if arrow_tbl.num_rows == 0 or "chunk_id" not in arrow_tbl.column_names:
            return set()
        existing = set(arrow_tbl.column("chunk_id").to_pylist())
        return existing & candidates

    def search(self, query: str, top_k: int = TOP_K) -> list[Hit]:
        """質問に近いチャンクを top_k 件取得。"""
        try:
            query_vec = self._model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0].tolist()
        except Exception as e:
            raise EmbeddingError(
                f"クエリの埋め込みに失敗しました: {type(e).__name__}: {e}"
            ) from e

        results = self._table.search(query_vec).limit(top_k).to_arrow()
        hits: list[Hit] = []
        for row in results.to_pylist():
            distance = float(row.get("_distance", 0.0))
            hits.append(
                Hit(
                    text=row["text"],
                    score=distance_to_score(distance),
                    paper_title=row["paper_title"],
                    page_num=int(row["page_num"]),
                    source_path=row["source_path"],
                )
            )
        return hits

    def count(self) -> int:
        return self._table.count_rows()

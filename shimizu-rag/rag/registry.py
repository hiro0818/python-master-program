"""取り込み済み論文のレジストリ（SQLite）。

LanceDB（ベクター本体）とは別管理で、論文単位のメタデータを保持する。
コンテナ再起動でDBが消えるクラウド環境でも、レジストリだけ別途バックアップすれば
「どのURLから何を取り込んだか」が復元できる。

スキーマ:
    papers(
        source_path  TEXT PRIMARY KEY,  -- papers/foo.pdf の絶対パス
        paper_title  TEXT NOT NULL,
        source_url   TEXT,              -- URL取り込みの場合のみ
        ingested_at  TEXT NOT NULL,     -- ISO8601 UTC
        n_pages      INTEGER NOT NULL,
        n_chunks     INTEGER NOT NULL
    )

設計上の注意:
    - SQLite は Python 標準ライブラリ → 追加依存なし
    - PRIMARY KEY = source_path により upsert で重複登録を防ぐ
    - LanceDB と同じ DB_DIR に置く（papers.sqlite）
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .config import DB_DIR
from .exceptions import RegistryError

REGISTRY_FILENAME = "papers.sqlite"


@dataclass(frozen=True)
class PaperRecord:
    source_path: str
    paper_title: str
    source_url: Optional[str]
    ingested_at: str
    n_pages: int
    n_chunks: int


def _registry_path() -> Path:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return DB_DIR / REGISTRY_FILENAME


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _registry_path()
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    except sqlite3.Error as e:
        raise RegistryError(f"レジストリの接続に失敗しました: {e}") from e


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            source_path  TEXT PRIMARY KEY,
            paper_title  TEXT NOT NULL,
            source_url   TEXT,
            ingested_at  TEXT NOT NULL,
            n_pages      INTEGER NOT NULL,
            n_chunks     INTEGER NOT NULL
        )
        """
    )
    conn.commit()


def record_ingestion(
    *,
    source_path: str,
    paper_title: str,
    n_pages: int,
    n_chunks: int,
    source_url: Optional[str] = None,
) -> None:
    """1本分の取り込みイベントを記録。同じ source_path は upsert。"""
    if not source_path:
        raise RegistryError("source_path が空です。")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO papers
                    (source_path, paper_title, source_url, ingested_at, n_pages, n_chunks)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    paper_title = excluded.paper_title,
                    source_url  = COALESCE(excluded.source_url, papers.source_url),
                    ingested_at = excluded.ingested_at,
                    n_pages     = excluded.n_pages,
                    n_chunks    = papers.n_chunks + excluded.n_chunks
                """,
                (source_path, paper_title, source_url, now, n_pages, n_chunks),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise RegistryError(f"レジストリの書き込みに失敗しました: {e}") from e


def list_papers() -> list[PaperRecord]:
    """取り込み済み論文を新しい順で返す。"""
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            rows = conn.execute(
                "SELECT source_path, paper_title, source_url, ingested_at, "
                "n_pages, n_chunks FROM papers ORDER BY ingested_at DESC"
            ).fetchall()
    except sqlite3.Error as e:
        raise RegistryError(f"レジストリの読み込みに失敗しました: {e}") from e
    return [
        PaperRecord(
            source_path=r["source_path"],
            paper_title=r["paper_title"],
            source_url=r["source_url"],
            ingested_at=r["ingested_at"],
            n_pages=r["n_pages"],
            n_chunks=r["n_chunks"],
        )
        for r in rows
    ]


def count_papers() -> int:
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            row = conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()
            return int(row["n"]) if row else 0
    except sqlite3.Error as e:
        raise RegistryError(f"レジストリの読み込みに失敗しました: {e}") from e

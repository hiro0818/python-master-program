"""SQLite レジストリのユニットテスト。

DB_DIR を tmp_path で差し替えて、本番DBを汚染しないように分離する。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def registry_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """rag.registry を tmp_path 配下のDBで使えるように再ロード。"""
    from rag import config as config_module

    monkeypatch.setattr(config_module, "DB_DIR", tmp_path / "db")
    from rag import registry as registry_module

    importlib.reload(registry_module)
    return registry_module


def test_record_and_list_roundtrip(registry_module) -> None:
    registry_module.record_ingestion(
        source_path="/tmp/foo.pdf",
        paper_title="Foo Title",
        n_pages=10,
        n_chunks=42,
        source_url="https://example.com/foo.pdf",
    )
    papers = registry_module.list_papers()
    assert len(papers) == 1
    assert papers[0].source_path == "/tmp/foo.pdf"
    assert papers[0].paper_title == "Foo Title"
    assert papers[0].source_url == "https://example.com/foo.pdf"
    assert papers[0].n_pages == 10
    assert papers[0].n_chunks == 42


def test_upsert_accumulates_chunks(registry_module) -> None:
    registry_module.record_ingestion(
        source_path="/tmp/bar.pdf", paper_title="Bar", n_pages=5, n_chunks=3
    )
    registry_module.record_ingestion(
        source_path="/tmp/bar.pdf", paper_title="Bar", n_pages=5, n_chunks=2
    )
    papers = registry_module.list_papers()
    assert len(papers) == 1
    assert papers[0].n_chunks == 5  # 累積


def test_upsert_preserves_url_when_later_call_omits_it(registry_module) -> None:
    registry_module.record_ingestion(
        source_path="/tmp/baz.pdf",
        paper_title="Baz",
        n_pages=1,
        n_chunks=1,
        source_url="https://example.com/baz.pdf",
    )
    registry_module.record_ingestion(
        source_path="/tmp/baz.pdf", paper_title="Baz", n_pages=1, n_chunks=1
    )
    papers = registry_module.list_papers()
    assert papers[0].source_url == "https://example.com/baz.pdf"


def test_count_and_ordering(registry_module) -> None:
    import time

    registry_module.record_ingestion(
        source_path="/tmp/a.pdf", paper_title="A", n_pages=1, n_chunks=1
    )
    time.sleep(1.1)  # ingested_at の秒精度差を確実にする
    registry_module.record_ingestion(
        source_path="/tmp/b.pdf", paper_title="B", n_pages=1, n_chunks=1
    )
    assert registry_module.count_papers() == 2
    papers = registry_module.list_papers()
    # 新しい順（DESC）
    assert papers[0].source_path == "/tmp/b.pdf"
    assert papers[1].source_path == "/tmp/a.pdf"


def test_empty_source_path_raises(registry_module) -> None:
    from rag.exceptions import RegistryError

    with pytest.raises(RegistryError):
        registry_module.record_ingestion(
            source_path="", paper_title="X", n_pages=1, n_chunks=1
        )

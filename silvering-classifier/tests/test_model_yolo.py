"""YOLOv8-cls wrapper のテスト。

ultralytics は重いので実際の訓練はテストしない。
代わりにデータ準備ユーティリティ（純pythonで完結）を確認する。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.model_yolo import build_yolo_split


def _make_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(path)


def test_build_yolo_split_creates_expected_structure(tmp_path):
    src_dir = tmp_path / "src_imgs"
    paths, labels = [], []
    for i in range(4):
        p = src_dir / f"img_parr_{i}.png"
        _make_dummy_image(p)
        paths.append(p)
        labels.append("parr")
    for i in range(4):
        p = src_dir / f"img_smolt_{i}.png"
        _make_dummy_image(p)
        paths.append(p)
        labels.append("smolt")

    train_paths = paths[:3] + paths[4:7]
    train_labels = labels[:3] + labels[4:7]
    val_paths = [paths[3], paths[7]]
    val_labels = [labels[3], labels[7]]

    root = tmp_path / "yolo_dataset"
    build_yolo_split(train_paths, val_paths, train_labels, val_labels, root)

    assert (root / "train" / "parr").is_dir()
    assert (root / "train" / "smolt").is_dir()
    assert (root / "val" / "parr").is_dir()
    assert (root / "val" / "smolt").is_dir()

    assert len(list((root / "train" / "parr").iterdir())) == 3
    assert len(list((root / "train" / "smolt").iterdir())) == 3
    assert len(list((root / "val" / "parr").iterdir())) == 1
    assert len(list((root / "val" / "smolt").iterdir())) == 1


def test_build_yolo_split_rejects_mismatched_lengths(tmp_path):
    src = tmp_path / "x.png"
    _make_dummy_image(src)
    with pytest.raises(ValueError):
        build_yolo_split([src, src], [src], ["parr"], ["parr"], tmp_path / "out")


def test_build_yolo_split_raises_on_missing_image(tmp_path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError):
        build_yolo_split([missing], [], ["parr"], [], tmp_path / "out")


def test_build_yolo_split_idempotent_on_repeated_call(tmp_path):
    src = tmp_path / "a.png"
    _make_dummy_image(src)
    root = tmp_path / "out"
    build_yolo_split([src], [src], ["parr"], ["smolt"], root)
    build_yolo_split([src], [src], ["parr"], ["smolt"], root)
    assert len(list((root / "train" / "parr").iterdir())) == 1
    assert len(list((root / "val" / "smolt").iterdir())) == 1

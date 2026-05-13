"""dataset.py のテスト。PIL/PyTorch 重依存をテスト時に避けるため、
合成画像 (NumPy で作って PNG 保存) を使う。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.dataset import FishImageDataset, discover_labels, load_image_rgb
from src.exceptions import DataError


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:] = color
    Image.fromarray(arr).save(path)


@pytest.fixture
def small_raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "raw"
    (raw / "parr").mkdir(parents=True)
    (raw / "smolt").mkdir(parents=True)
    _make_image(raw / "parr" / "a.png", (50, 50, 50))
    _make_image(raw / "parr" / "b.png", (60, 60, 60))
    _make_image(raw / "smolt" / "x.png", (200, 200, 200))
    _make_image(raw / "smolt" / "y.png", (220, 220, 220))

    from src import config as cfg

    monkeypatch.setattr(cfg, "RAW_DIR", raw)
    monkeypatch.setattr(cfg, "LABELS_CSV", tmp_path / "labels.csv")
    # discover_labels が再 import 時のスナップショットを取らないようリロード
    import importlib

    from src import dataset as ds_module

    importlib.reload(ds_module)
    return raw


def test_load_image_rgb(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    _make_image(p, (100, 150, 200))
    arr = load_image_rgb(p)
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.uint8


def test_load_image_rgb_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="読み込み"):
        load_image_rgb(tmp_path / "no.png")


def test_discover_labels_from_directory(small_raw_dir: Path) -> None:
    from src.dataset import discover_labels as discover

    df = discover()
    assert len(df) == 4
    assert set(df["label"]) == {"parr", "smolt"}


def test_dataset_iteration(small_raw_dir: Path) -> None:
    from src.dataset import FishImageDataset as DS
    from src.dataset import discover_labels as discover

    df = discover()
    ds = DS(df, transform=None)
    assert len(ds) == 4
    image, label = ds[0]
    assert image.shape == (64, 64, 3)
    assert label in (0, 1)


def test_dataset_with_transform_returns_tensor(small_raw_dir: Path) -> None:
    """albumentations + ToTensorV2 が C×H×W テンソルを返すこと。"""
    from src.augment import eval_transform
    from src.dataset import FishImageDataset as DS
    from src.dataset import discover_labels as discover

    df = discover()
    ds = DS(df, transform=eval_transform())
    image, label = ds[0]
    assert image.ndim == 3
    assert image.shape[0] == 3  # チャンネル先頭
    assert isinstance(label, int)


def test_dataset_empty_df_raises() -> None:
    with pytest.raises(DataError):
        FishImageDataset(pd.DataFrame(columns=["image_path", "label"]))


def test_labels_csv_takes_precedence_over_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    (raw / "parr").mkdir(parents=True)
    (raw / "smolt").mkdir(parents=True)
    img = tmp_path / "fish.png"
    _make_image(img, (128, 128, 128))

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [{"image_path": str(img), "label": "parr"}]
    ).to_csv(csv_path, index=False)

    from src import config as cfg

    monkeypatch.setattr(cfg, "RAW_DIR", raw)
    monkeypatch.setattr(cfg, "LABELS_CSV", csv_path)
    import importlib

    from src import dataset as ds_module

    importlib.reload(ds_module)

    df = ds_module.discover_labels()
    assert len(df) == 1
    assert df.iloc[0]["label"] == "parr"


def test_labels_csv_unknown_label_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [{"image_path": "/tmp/x.png", "label": "unknown_class"}]
    ).to_csv(csv_path, index=False)

    from src import config as cfg

    monkeypatch.setattr(cfg, "LABELS_CSV", csv_path)
    import importlib

    from src import dataset as ds_module

    importlib.reload(ds_module)

    with pytest.raises(DataError, match="未知のラベル"):
        ds_module.discover_labels()

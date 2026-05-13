"""augment.py のテスト。実画像不要、NumPy 配列で完結。"""
from __future__ import annotations

import numpy as np
import pytest

from src.augment import eval_transform, train_transform
from src.config import IMAGE_SIZE


def _dummy_image(h: int = 100, w: int = 200) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_eval_transform_output_shape() -> None:
    img = _dummy_image()
    out = eval_transform()(image=img)["image"]
    # albumentations + ToTensorV2 → torch.Tensor (C, H, W)
    assert out.ndim == 3
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_train_transform_output_shape() -> None:
    img = _dummy_image()
    out = train_transform()(image=img)["image"]
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_train_transform_is_stochastic() -> None:
    """訓練変換は確率的なので、2回呼ぶと（高確率で）異なる結果になる。"""
    img = _dummy_image()
    tf = train_transform()
    a = tf(image=img)["image"]
    b = tf(image=img)["image"]
    # まったく同じになることは確率的に極めて低い
    assert not (a == b).all()


def test_eval_transform_is_deterministic() -> None:
    img = _dummy_image()
    tf = eval_transform()
    a = tf(image=img)["image"]
    b = tf(image=img)["image"]
    assert (a == b).all()


@pytest.mark.parametrize("h,w", [(50, 50), (300, 150), (100, 400)])
def test_handles_various_aspect_ratios(h: int, w: int) -> None:
    img = _dummy_image(h, w)
    out = eval_transform()(image=img)["image"]
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

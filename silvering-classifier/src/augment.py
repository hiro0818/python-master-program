"""データ拡張パイプライン。少データ向けに「強め」に振っている。

設計思想:
    - 銀化判定は「体色・形状」が手がかり。色相変化は控えめ、明度・コントラストは強め。
    - 水平反転は OK（魚の向きは左右どちらでも銀化判定に影響しない）。
    - 上下反転は NG（背側 vs 腹側の銀化グラデーションを破壊する）。
    - 回転は ±15° まで（撮影傾きの吸収）。
    - 訓練時のみ強拡張、推論時はリサイズと正規化のみ。
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import IMAGE_SIZE

# ImageNet 統計（バックボーンが ImageNet 事前学習なので必須）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform() -> A.Compose:
    """訓練時の変換（強拡張あり）。"""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_SIZE * 2),
            A.PadIfNeeded(
                min_height=IMAGE_SIZE * 2,
                min_width=IMAGE_SIZE * 2,
                border_mode=0,
                fill=0,
            ),
            A.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0), ratio=(0.75, 1.33)
            ),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform() -> A.Compose:
    """推論／検証時の変換（拡張なし、リサイズ + 正規化のみ）。"""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_SIZE,
                min_width=IMAGE_SIZE,
                border_mode=0,
                fill=0,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

"""モデルファクトリ。timm 経由で EfficientNet-B0 を ImageNet 重みでロード。

設計:
    - バックボーンを差し替え可能（config.BACKBONE で任意のtimmモデルに切替）
    - 出力層は 2 クラス（parr/smolt）
    - Grad-CAM 用にターゲット層への参照を取り出すヘルパーも用意
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import BACKBONE, CLASSES
from .exceptions import ModelError


def build_model(pretrained: bool = True, num_classes: int = len(CLASSES)) -> nn.Module:
    """事前学習済みバックボーンに分類ヘッドを付けたモデルを返す。"""
    try:
        import timm
    except ImportError as e:
        raise ModelError("timm がインストールされていません。pip install timm") from e

    try:
        model = timm.create_model(
            BACKBONE,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception as e:
        raise ModelError(
            f"バックボーン '{BACKBONE}' の構築に失敗: {type(e).__name__}: {e}"
        ) from e
    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM のターゲット層を返す。

    EfficientNet 系は最終 ConvBlock を使うのが定石。
    timm の EfficientNet は `conv_head` の前の `blocks[-1]` がターゲットに最適。
    """
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        return model.blocks[-1]
    if hasattr(model, "layer4"):
        return model.layer4  # ResNet系
    raise ModelError(
        f"Grad-CAM ターゲット層を自動推定できません: model class = {type(model).__name__}"
    )


def save_checkpoint(model: nn.Module, path) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path, map_location: str = "cpu") -> nn.Module:
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

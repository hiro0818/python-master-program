"""銀化分類器の設定。パラメータ変更はここで完結する。

環境変数で上書き可能（クラウド・GPU環境向け）:
    BACKBONE, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR, N_FOLDS, SEED
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# データの置き場所
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LABELS_CSV = DATA_DIR / "labels.csv"
ATTRIBUTION_CSV = DATA_DIR / "attribution.csv"

# 学習済み重み・出力
MODELS_DIR = BASE_DIR / "models"
GRADCAM_DIR = MODELS_DIR / "gradcam"

# クラス定義（順序が確定的、CSV や予測出力で使う）
CLASSES = ["parr", "smolt"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# バックボーン（timm の名前）
# EfficientNet-B0: 5.3M params, ImageNet 1k 事前学習済み、軽量・速い
BACKBONE = os.getenv("BACKBONE", "tf_efficientnet_b0.in1k")

# 入力画像サイズ（B0 のデフォルト 224）
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))

# 訓練ハイパーパラメータ
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "30"))
LR = float(os.getenv("LR", "3e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))

# 交差検証（少データで信頼度を上げる）
N_FOLDS = int(os.getenv("N_FOLDS", "5"))

# 再現性
SEED = int(os.getenv("SEED", "42"))

# デバイス（CPU/CUDA/MPS 自動選択）
def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

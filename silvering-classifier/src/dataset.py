"""PyTorch Dataset: ディスク上の画像ファイルから (image_tensor, label) を返す。

ラベルソース:
    1. data/labels.csv があればそれを使う（image_path, label の2列）
    2. 無ければ data/raw/{parr,smolt}/*.{jpg,jpeg,png,webp} を自動スキャン

設計:
    - 画像読込は Pillow（PNG/JPEG/WebP対応）
    - 変換は albumentations（NumPy配列前提なので Pillow → NumPy 経由）
    - 不整合は DataError で日本語メッセージにラップ
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .config import CLASS_TO_IDX, CLASSES, LABELS_CSV, RAW_DIR
from .exceptions import DataError

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def discover_labels() -> pd.DataFrame:
    """ラベルCSV か data/raw/{class}/ ディレクトリから (path, label) を発見。

    戻り値: 列 = ['image_path' (str, 絶対パス), 'label' (str, クラス名)]
    """
    if LABELS_CSV.exists():
        df = pd.read_csv(LABELS_CSV)
        required = {"image_path", "label"}
        if not required.issubset(df.columns):
            raise DataError(
                f"labels.csv の列が不足しています。必要: {required}, 実際: {set(df.columns)}"
            )
        df["label"] = df["label"].astype(str).str.strip()
        unknown = set(df["label"]) - set(CLASSES)
        if unknown:
            raise DataError(
                f"未知のラベルが含まれます: {unknown}。CLASSES={CLASSES} に統一してください。"
            )
        df["image_path"] = df["image_path"].astype(str)
        return df

    rows: list[dict] = []
    for cls in CLASSES:
        d = RAW_DIR / cls
        if not d.exists():
            continue
        for ext in IMAGE_EXTS:
            for p in sorted(d.glob(f"*{ext}")):
                rows.append({"image_path": str(p.resolve()), "label": cls})
    if not rows:
        raise DataError(
            f"画像が見つかりません。{RAW_DIR}/{{parr,smolt}}/ に画像を置くか、"
            f"{LABELS_CSV} を用意してください。"
        )
    return pd.DataFrame(rows)


def load_image_rgb(path: str | Path) -> np.ndarray:
    """画像を RGB の uint8 NumPy 配列で読み込む。"""
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        raise DataError(f"画像の読み込みに失敗しました ({path}): {e}") from e
    return np.array(img, dtype=np.uint8)


class FishImageDataset(Dataset):
    """魚の写真と銀化ラベルのデータセット。

    Args:
        df: 列 ['image_path', 'label'] を持つ DataFrame
        transform: albumentations の Compose（image=np_array で受け取り、'image' を返す）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
    ) -> None:
        if df.empty:
            raise DataError("空のDataFrameが渡されました。")
        self._df = df.reset_index(drop=True)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        image = load_image_rgb(row["image_path"])
        label = CLASS_TO_IDX[row["label"]]

        if self._transform is not None:
            image = self._transform(image=image)["image"]
        return image, label

    @property
    def labels(self) -> list[int]:
        return [CLASS_TO_IDX[l] for l in self._df["label"]]

    @property
    def df(self) -> pd.DataFrame:
        return self._df

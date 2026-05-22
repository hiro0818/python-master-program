"""YOLOv8 分類モデル wrapper。

EfficientNet-B0 (src/model.py) と並列のベースライン。同じ FoldPredictions
スキーマを返すので、src/evaluate.py の compute_metrics 等がそのまま使える。

設計判断:
    - Ultralytics は dataset を ImageFolder 形式（train/<class>/*.jpg, val/<class>/*.jpg）で
      期待するため、fold ごとに tempdir + symlink で構造を作る。
    - YOLO はクラスをアルファベット順に並べ替えるので、明示的に smolt の index を
      `model.names` から引いて P(smolt) を取り出す。
    - 既存の MODELS_DIR/yolo_fold{k}.pt として best weights を保存（推論用）。
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import CLASSES


def build_yolo_split(
    train_paths: Sequence[str | Path],
    val_paths: Sequence[str | Path],
    train_labels: Sequence[str],
    val_labels: Sequence[str],
    root: Path,
) -> Path:
    """YOLOv8-cls が期待する ImageFolder 構造を root 配下に作る。

    train/<class>/*.jpg, val/<class>/*.jpg。
    シンボリックリンクを優先し、不可なら copy フォールバック。

    Args:
        train_paths, val_paths: 画像の絶対パス
        train_labels, val_labels: 各画像のクラス名（"parr" or "smolt"）
        root: 出力先ディレクトリ（存在しない場合は作成）

    Returns:
        root（YOLO の data 引数にそのまま渡せる）
    """
    if len(train_paths) != len(train_labels):
        raise ValueError("train_paths と train_labels の長さが一致しません。")
    if len(val_paths) != len(val_labels):
        raise ValueError("val_paths と val_labels の長さが一致しません。")

    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for cls_name in CLASSES:
            (root / split / cls_name).mkdir(parents=True, exist_ok=True)

    for split, paths, labels in (
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
    ):
        for path, label in zip(paths, labels):
            src = Path(path).resolve()
            if not src.exists():
                raise FileNotFoundError(f"画像が見つかりません: {src}")
            dest = root / split / str(label) / src.name
            if dest.exists() or dest.is_symlink():
                continue
            try:
                dest.symlink_to(src)
            except (OSError, NotImplementedError):
                shutil.copy(src, dest)
    return root


def _smolt_index(class_names: dict[int, str]) -> int:
    """model.names の {idx: name} 辞書から smolt クラスの index を返す。"""
    for idx, name in class_names.items():
        if name == "smolt":
            return int(idx)
    raise KeyError(f"smolt クラスが見つかりません: {class_names}")


def train_yolo_fold(
    fold_idx: int,
    train_paths: Sequence[str | Path],
    val_paths: Sequence[str | Path],
    train_labels: Sequence[str],
    val_labels: Sequence[str],
    *,
    epochs: int = 20,
    imgsz: int = 224,
    model_name: str = "yolov8n-cls.pt",
    save_weights_to: Path | None = None,
) -> np.ndarray:
    """1 fold だけ YOLOv8-cls を訓練し、val の smolt 確率を返す。

    Returns:
        val_paths と同じ順序の P(smolt) 配列、shape=(len(val_paths),)
    """
    from ultralytics import YOLO

    workdir = Path(tempfile.mkdtemp(prefix=f"yolo_fold{fold_idx}_"))
    try:
        dataset_root = workdir / "data"
        build_yolo_split(
            train_paths, val_paths, train_labels, val_labels, dataset_root
        )

        model = YOLO(model_name)
        model.train(
            data=str(dataset_root),
            epochs=epochs,
            imgsz=imgsz,
            verbose=False,
            plots=False,
            project=str(workdir / "runs"),
            name=f"fold{fold_idx}",
            exist_ok=True,
        )

        smolt_idx = _smolt_index(model.names)
        probs: list[float] = []
        for path in val_paths:
            result = model.predict(source=str(path), verbose=False)
            p = result[0].probs.data.cpu().numpy()
            probs.append(float(p[smolt_idx]))

        if save_weights_to is not None:
            run_dir = workdir / "runs" / f"fold{fold_idx}" / "weights"
            best = run_dir / "best.pt"
            if best.exists():
                save_weights_to.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(best, save_weights_to)

        return np.array(probs)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

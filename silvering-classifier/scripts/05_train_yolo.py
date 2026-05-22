"""YOLOv8-cls 5-fold ベースライン。

EfficientNet-B0 (03_run_baseline.py) と同じ評価スキーマで
models/evaluation_report_yolo.json を生成する。

使い方:
    pip install -r requirements-train.txt   # ultralytics が必要
    python scripts/05_train_yolo.py         # 5-fold 訓練 → metrics → JSON

出力:
    - models/yolo_fold{0..4}.pt  各 fold の best weights
    - models/evaluation_report_yolo.json  完全な評価レポート（EfficientNet と同形式）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASSES, MODELS_DIR, N_FOLDS, SEED  # noqa: E402
from src.dataset import discover_labels  # noqa: E402
from src.evaluate import (  # noqa: E402
    FoldPredictions,
    compute_metrics,
    cost_aware_threshold,
    cost_sensitivity_analysis,
    evaluate_distribution_shift,
)


def _require_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "ultralytics が見つかりません。\n"
            "  pip install -r requirements-train.txt\n"
            "を実行してください。"
        ) from e


def main() -> None:
    _require_ultralytics()
    from src.model_yolo import train_yolo_fold  # 遅延import: torch経由

    df = discover_labels()
    paths = df["image_path"].tolist()
    labels = df["label"].tolist()
    y_numeric = np.array([CLASSES.index(c) for c in labels])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    y_true_all: list[int] = []
    y_proba_all: list[float] = []

    for fold_idx, (_, val_idx) in enumerate(skf.split(paths, y_numeric)):
        print(f"\n=== Fold {fold_idx + 1}/{N_FOLDS} ===")
        train_idx = [i for i in range(len(paths)) if i not in set(val_idx)]
        train_paths = [paths[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        save_to = MODELS_DIR / f"yolo_fold{fold_idx}.pt"
        proba = train_yolo_fold(
            fold_idx,
            train_paths,
            val_paths,
            train_labels,
            val_labels,
            epochs=20,
            imgsz=224,
            model_name="yolov8n-cls.pt",
            save_weights_to=save_to,
        )

        y_true_all.extend([CLASSES.index(c) for c in val_labels])
        y_proba_all.extend(proba.tolist())

    y_true = np.array(y_true_all)
    y_proba = np.array(y_proba_all)
    y_pred = (y_proba >= 0.5).astype(int)

    preds = FoldPredictions(y_true=y_true, y_pred=y_pred, y_proba=y_proba)

    print("\n=== 集計 ===")
    metrics = compute_metrics(preds)
    cost = cost_aware_threshold(preds, cost_fn=10.0, cost_fp=1.0)
    shift = evaluate_distribution_shift(preds)
    sensitivity = cost_sensitivity_analysis(preds)

    report = {
        "model": "YOLOv8n-cls",
        "n_folds": N_FOLDS,
        "predictions": {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist(),
        },
        "metrics_at_0.5": metrics,
        "cost_optimization": cost,
        "distribution_shift": shift,
        "cost_sensitivity": sensitivity,
    }

    out_path = MODELS_DIR / "evaluation_report_yolo.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(
        f"YOLOv8n-cls: MCC={metrics['mcc']:.3f} "
        f"PR-AUC={metrics['pr_auc']:.3f} "
        f"F1={metrics['f1']:.3f} "
        f"(n={metrics['n']})"
    )


if __name__ == "__main__":
    main()

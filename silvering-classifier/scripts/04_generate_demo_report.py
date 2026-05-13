"""デモ用の合成評価レポートを生成。

本物のモデルが無くてもダッシュボード全タブを動かせるよう、現実的な数値の
predictions / metrics / shift / sensitivity を生成して models/demo_evaluation_report.json
に書く。Streamlit Cloud にデプロイした時、見た人が機能を把握できるようにするためのもの。

使い方:
    python scripts/04_generate_demo_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    FoldPredictions,
    compute_metrics,
    cost_aware_threshold,
    cost_sensitivity_analysis,
    evaluate_distribution_shift,
)


def synthesize_predictions(
    n_per_class: int = 60, seed: int = 42, separation: float = 0.25
) -> FoldPredictions:
    """二値分類の現実的な予測値を生成。

    parr (0) の確率は 0.5-separation 中心、smolt (1) は 0.5+separation 中心、
    どちらもガウシアン分散でオーバーラップさせる。よくある「accuracy 0.85 くらいの
    そこそこ性能の分類器」が再現できる。
    """
    rng = np.random.default_rng(seed)
    y_true = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)

    proba_neg = rng.normal(0.5 - separation, 0.15, n_per_class)
    proba_pos = rng.normal(0.5 + separation, 0.15, n_per_class)
    y_proba = np.concatenate([proba_neg, proba_pos]).clip(0.01, 0.99)

    y_pred = (y_proba >= 0.5).astype(int)
    return FoldPredictions(y_true=y_true, y_pred=y_pred, y_proba=y_proba)


def main() -> int:
    preds = synthesize_predictions(n_per_class=60, separation=0.25)
    metrics = compute_metrics(preds)
    cost = cost_aware_threshold(preds, cost_fn=10.0, cost_fp=1.0)
    sensitivity = cost_sensitivity_analysis(preds)
    shift = evaluate_distribution_shift(preds)

    out = ROOT / "models" / "demo_evaluation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "_demo": True,
                "_note": "Synthetic data for dashboard demonstration. Replace with output of scripts/03_run_baseline.py once real training is done.",
                "training_summary": {
                    "mean": {
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "roc_auc": metrics["roc_auc"],
                    },
                    "n_folds": 5,
                },
                "metrics_at_0.5": metrics,
                "cost_optimization": cost,
                "cost_sensitivity": sensitivity,
                "distribution_shift": shift,
                "confusion_matrix": {
                    "confusion_matrix": [
                        [metrics["tn"], metrics["fp"]],
                        [metrics["fn"], metrics["tp"]],
                    ],
                    "n": metrics["n"],
                },
                "predictions": {
                    "y_true": preds.y_true.tolist(),
                    "y_pred": preds.y_pred.tolist(),
                    "y_proba": preds.y_proba.tolist(),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"✅ デモレポート生成: {out}")
    print(f"   accuracy={metrics['accuracy']:.3f}  MCC={metrics['mcc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

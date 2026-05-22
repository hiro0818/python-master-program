"""EfficientNet-B0 vs YOLOv8-cls の指標を並べた比較表を生成する。

修論の「モデル選定の根拠」章に使う。両方の評価レポートを読み込み:
    - models/evaluation_report.json       (EfficientNet, 03_run_baseline.py の出力)
    - models/evaluation_report_yolo.json  (YOLOv8-cls, 05_train_yolo.py の出力)

出力:
    - models/model_comparison.json   機械可読
    - 標準出力に Markdown テーブル
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR  # noqa: E402

EFFICIENTNET_REPORT = MODELS_DIR / "evaluation_report.json"
YOLO_REPORT = MODELS_DIR / "evaluation_report_yolo.json"

METRIC_KEYS = ["accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "pr_auc"]


def _load(path: Path, label: str) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{label} のレポートが見つかりません: {path}\n"
            f"先に対応する訓練スクリプトを実行してください。"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _row(name: str, metrics: dict) -> dict:
    out = {"model": name}
    for k in METRIC_KEYS:
        out[k] = round(float(metrics[k]), 4)
    return out


def main() -> None:
    eff = _load(EFFICIENTNET_REPORT, "EfficientNet-B0")
    yolo = _load(YOLO_REPORT, "YOLOv8n-cls")

    rows = [
        _row(eff.get("model", "EfficientNet-B0"), eff["metrics_at_0.5"]),
        _row(yolo.get("model", "YOLOv8n-cls"), yolo["metrics_at_0.5"]),
    ]

    out = {
        "models": rows,
        "best_by_metric": {
            k: max(rows, key=lambda r: r[k])["model"] for k in METRIC_KEYS
        },
    }
    out_path = MODELS_DIR / "model_comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}\n")

    headers = ["model"] + METRIC_KEYS
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [row["model"]] + [f"{row[k]:.4f}" for k in METRIC_KEYS]
        print("| " + " | ".join(cells) + " |")

    print("\nBest model by metric:")
    for k, name in out["best_by_metric"].items():
        print(f"  {k:12s} -> {name}")


if __name__ == "__main__":
    main()

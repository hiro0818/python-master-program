"""ベースラインを1コマンドで実行: 訓練 → 評価 → Grad-CAM。

使い方:
    python scripts/03_run_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import GRADCAM_DIR, MODELS_DIR  # noqa: E402
from src.evaluate import plot_confusion_matrix, save_gradcam_examples  # noqa: E402
from src.exceptions import ClassifierError  # noqa: E402
from src.train import train_cv  # noqa: E402


def main() -> int:
    print("=" * 60)
    print("🐟 銀化分類器 ベースライン実行")
    print("=" * 60)

    try:
        print("\n[1/3] 訓練 (Stratified K-Fold CV)")
        summary = train_cv()
        print("\n✅ 訓練完了")
        print(json.dumps(summary["mean"], indent=2, ensure_ascii=False))

        print("\n[2/3] 混同行列を描画")
        cm_path = MODELS_DIR / "confusion_matrix.png"
        cm_result = plot_confusion_matrix(cm_path)
        print(f"✅ {cm_path}")
        print(f"   集計サンプル数: {cm_result['n']}")

        print("\n[3/3] Grad-CAM を保存（fold 0）")
        save_gradcam_examples(fold=0, n_per_class=4)
        print(f"✅ {GRADCAM_DIR}")

        print("\n" + "=" * 60)
        print(f"📊 metrics.json: {MODELS_DIR / 'metrics.json'}")
        print(f"📈 confusion_matrix.png: {cm_path}")
        print(f"🔍 Grad-CAM: {GRADCAM_DIR}")
        print("=" * 60)
        return 0
    except ClassifierError as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

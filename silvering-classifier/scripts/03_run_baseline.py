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
from src.evaluate import (  # noqa: E402
    collect_cv_predictions,
    compute_metrics,
    cost_aware_threshold,
    cost_sensitivity_analysis,
    evaluate_distribution_shift,
    plot_confusion_matrix,
    save_gradcam_examples,
)
from src.exceptions import ClassifierError  # noqa: E402
from src.train import train_cv  # noqa: E402
from src.visualize import save_all_figures  # noqa: E402


def main() -> int:
    print("=" * 60)
    print("🐟 銀化分類器 ベースライン実行")
    print("=" * 60)

    try:
        print("\n[1/5] 訓練 (Stratified K-Fold CV)")
        summary = train_cv()
        print("\n✅ 訓練完了")
        print(json.dumps(summary["mean"], indent=2, ensure_ascii=False))

        print("\n[2/5] 予測を集約して指標を計算")
        preds = collect_cv_predictions()
        metrics = compute_metrics(preds)
        print("✅ 主要指標 (threshold=0.5):")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

        print("\n[3/5] コスト考慮型閾値最適化 (FN:FP = 10:1)")
        cost = cost_aware_threshold(preds)
        print(f"✅ 最適閾値: {cost['best_threshold']:.3f}, 総コスト: {cost['best_cost']:.1f}")
        print("   default vs optimal:")
        print(
            json.dumps(
                {
                    "default": cost["default_metrics"],
                    "optimal": cost["optimal_metrics"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n[4/5] 分布シフト評価 (正例比率を 0.1〜0.9 で変動)")
        shift = evaluate_distribution_shift(preds)
        print(json.dumps(shift, indent=2, ensure_ascii=False))

        print("\n[4b/5] コスト比 sensitivity analysis")
        sensitivity = cost_sensitivity_analysis(preds)
        print(json.dumps(sensitivity, indent=2, ensure_ascii=False))

        print("\n[5/5] 混同行列・プレゼン用Figure・Grad-CAM を保存")
        cm_path = MODELS_DIR / "confusion_matrix.png"
        cm_result = plot_confusion_matrix(cm_path, preds=preds)
        print(f"✅ {cm_path}")
        print(f"   集計サンプル数: {cm_result['n']}")

        figures_dir = MODELS_DIR / "figures"
        fig_paths = save_all_figures(
            preds=preds,
            cost_result=cost,
            shift_result=shift,
            metrics=metrics,
            out_dir=figures_dir,
            sensitivity_result=sensitivity,
        )
        for name, path in fig_paths.items():
            print(f"   📈 {name}: {path}")

        save_gradcam_examples(fold=0, n_per_class=4)
        print(f"✅ Grad-CAM: {GRADCAM_DIR}")

        # 全指標と予測値を JSON に保存（Notebookで再可視化できるよう）
        report_path = MODELS_DIR / "evaluation_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "training_summary": summary,
                    "metrics_at_0.5": metrics,
                    "cost_optimization": cost,
                    "cost_sensitivity": sensitivity,
                    "distribution_shift": shift,
                    "confusion_matrix": cm_result,
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

        print("\n" + "=" * 60)
        print(f"📊 metrics.json: {MODELS_DIR / 'metrics.json'}")
        print(f"📋 evaluation_report.json: {report_path}")
        print(f"📈 confusion_matrix.png: {cm_path}")
        print(f"🎨 figures/: {figures_dir}")
        print(f"🔍 Grad-CAM: {GRADCAM_DIR}")
        print("=" * 60)
        return 0
    except ClassifierError as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

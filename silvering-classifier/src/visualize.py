"""修論・学会プレゼン用の可視化。

各関数は matplotlib の Figure を返すので、呼び出し側で `fig.savefig()` できる。
パスを渡せば PNG として保存もする。

参照: ghmagazine/evaluation_book (Apache-2.0) 3.10/3.11 のグラフ作成手法。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)

from .evaluate import FoldPredictions


def _save_and_return(fig, out_path: Optional[Path]):
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pr_curve(preds: FoldPredictions, out_path: Optional[Path] = None):
    """Precision-Recall 曲線。不均衡データではこちらが本命。"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score

    precision, recall, _ = precision_recall_curve(preds.y_true, preds.y_proba)
    ap = average_precision_score(preds.y_true, preds.y_proba)
    baseline = float(preds.y_true.mean())

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.axhline(baseline, ls="--", color="gray", label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_and_return(fig, out_path)


def plot_roc_curve(preds: FoldPredictions, out_path: Optional[Path] = None):
    """ROC 曲線。慣習として必ず添える。"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    fpr, tpr, _ = roc_curve(preds.y_true, preds.y_proba)
    auc = roc_auc_score(preds.y_true, preds.y_proba)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_and_return(fig, out_path)


def plot_cost_threshold_curve(cost_result: dict, out_path: Optional[Path] = None):
    """コスト-閾値曲線。最適閾値が一目でわかる。

    Args:
        cost_result: evaluate.cost_aware_threshold() の戻り値。
    """
    import matplotlib.pyplot as plt

    history = cost_result["history"]
    thresholds = [h["threshold"] for h in history]
    costs = [h["cost"] for h in history]
    fns = [h["fn"] for h in history]
    fps = [h["fp"] for h in history]

    best_t = cost_result["best_threshold"]
    cost_fn = cost_result["cost_fn"]
    cost_fp = cost_result["cost_fp"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左: 総コスト
    ax1.plot(thresholds, costs, lw=2, color="C3")
    ax1.axvline(best_t, ls="--", color="black", label=f"Optimal t = {best_t:.2f}")
    ax1.axvline(0.5, ls=":", color="gray", label="Default t = 0.5")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel(f"Total cost  ({cost_fn:.0f}·FN + {cost_fp:.0f}·FP)")
    ax1.set_title("Cost vs Threshold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 右: FN/FP 内訳
    ax2.plot(thresholds, fns, lw=2, label="False Negatives", color="C0")
    ax2.plot(thresholds, fps, lw=2, label="False Positives", color="C1")
    ax2.axvline(best_t, ls="--", color="black")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Composition")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Cost-aware threshold optimization (FN:FP = {cost_fn:.0f}:{cost_fp:.0f})",
        fontsize=11,
    )
    fig.tight_layout()
    return _save_and_return(fig, out_path)


def plot_distribution_shift(shift_result: dict, out_path: Optional[Path] = None):
    """正例比率を変動させた時の指標推移。デプロイ時の頑健性を示す。

    Args:
        shift_result: evaluate.evaluate_distribution_shift() の戻り値。
    """
    import matplotlib.pyplot as plt

    rows = shift_result.get("shift_evaluation")
    if not rows:
        raise ValueError("shift_result に 'shift_evaluation' が含まれていない")

    ratios = [r["positive_ratio"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    f1s = [r["f1"] for r in rows]
    mccs = [r["mcc"] for r in rows]
    rocs = [r["roc_auc"] for r in rows]
    prs = [r["pr_auc"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ratios, accs, marker="o", lw=2, label="Accuracy")
    ax.plot(ratios, f1s, marker="s", lw=2, label="F1")
    ax.plot(ratios, mccs, marker="^", lw=2, label="MCC")
    ax.plot(ratios, rocs, marker="D", lw=2, label="ROC-AUC")
    ax.plot(ratios, prs, marker="v", lw=2, label="PR-AUC")
    ax.set_xlabel("Positive (smolt) ratio in test set")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Metric stability under class-distribution shift")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_and_return(fig, out_path)


def plot_metric_summary_bar(metrics: dict, out_path: Optional[Path] = None):
    """主要指標を1枚の棒グラフで俯瞰。スライド冒頭に使える。

    Args:
        metrics: evaluate.compute_metrics() の戻り値。
    """
    import matplotlib.pyplot as plt

    keys = ["accuracy", "precision", "recall", "f1", "g_mean", "mcc", "roc_auc", "pr_auc"]
    values = [metrics[k] for k in keys]
    colors = ["#9aa0a6"] * 4 + ["#5b8def"] * 2 + ["#34a853"] * 2

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Summary metrics (n = {metrics['n']}, threshold = {metrics['threshold']:.2f})")
    ax.grid(axis="y", alpha=0.3)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.3f}",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    return _save_and_return(fig, out_path)


def save_all_figures(
    preds: FoldPredictions,
    cost_result: dict,
    shift_result: dict,
    metrics: dict,
    out_dir: Path,
) -> dict[str, Path]:
    """5枚をまとめて保存。修論Figure用の一括出力。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pr_curve": out_dir / "pr_curve.png",
        "roc_curve": out_dir / "roc_curve.png",
        "cost_threshold": out_dir / "cost_threshold.png",
        "distribution_shift": out_dir / "distribution_shift.png",
        "metric_summary": out_dir / "metric_summary.png",
    }
    plot_pr_curve(preds, paths["pr_curve"])
    plot_roc_curve(preds, paths["roc_curve"])
    plot_cost_threshold_curve(cost_result, paths["cost_threshold"])
    plot_distribution_shift(shift_result, paths["distribution_shift"])
    plot_metric_summary_bar(metrics, paths["metric_summary"])
    return paths

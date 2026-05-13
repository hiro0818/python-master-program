"""visualize.py のスモークテスト。

matplotlib が無い環境では skip。あれば各 plotter が Figure を返すことを確認。
画像の中身は検証しない（修論の図は人間が見て品質チェックする）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # ヘッドレス環境用

from src.evaluate import (  # noqa: E402
    FoldPredictions,
    cost_aware_threshold,
    evaluate_distribution_shift,
    compute_metrics,
)
from src.visualize import (  # noqa: E402
    plot_cost_threshold_curve,
    plot_distribution_shift,
    plot_metric_summary_bar,
    plot_pr_curve,
    plot_roc_curve,
    save_all_figures,
)


@pytest.fixture
def preds() -> FoldPredictions:
    rng = np.random.default_rng(7)
    n = 100
    y_true = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    y_proba = np.where(
        y_true == 1, rng.uniform(0.55, 0.95, n), rng.uniform(0.05, 0.45, n)
    )
    return FoldPredictions(
        y_true=y_true,
        y_pred=(y_proba >= 0.5).astype(int),
        y_proba=y_proba,
    )


def test_plot_pr_curve_returns_figure(preds: FoldPredictions, tmp_path: Path) -> None:
    fig = plot_pr_curve(preds, tmp_path / "pr.png")
    assert fig is not None
    assert (tmp_path / "pr.png").exists()


def test_plot_roc_curve_returns_figure(preds: FoldPredictions, tmp_path: Path) -> None:
    fig = plot_roc_curve(preds, tmp_path / "roc.png")
    assert fig is not None
    assert (tmp_path / "roc.png").exists()


def test_plot_cost_threshold_curve(preds: FoldPredictions, tmp_path: Path) -> None:
    cost = cost_aware_threshold(preds)
    fig = plot_cost_threshold_curve(cost, tmp_path / "cost.png")
    assert fig is not None
    assert (tmp_path / "cost.png").exists()


def test_plot_distribution_shift(preds: FoldPredictions, tmp_path: Path) -> None:
    shift = evaluate_distribution_shift(preds)
    fig = plot_distribution_shift(shift, tmp_path / "shift.png")
    assert fig is not None
    assert (tmp_path / "shift.png").exists()


def test_plot_metric_summary_bar(preds: FoldPredictions, tmp_path: Path) -> None:
    m = compute_metrics(preds)
    fig = plot_metric_summary_bar(m, tmp_path / "summary.png")
    assert fig is not None
    assert (tmp_path / "summary.png").exists()


def test_save_all_figures(preds: FoldPredictions, tmp_path: Path) -> None:
    cost = cost_aware_threshold(preds)
    shift = evaluate_distribution_shift(preds)
    metrics = compute_metrics(preds)
    paths = save_all_figures(
        preds=preds,
        cost_result=cost,
        shift_result=shift,
        metrics=metrics,
        out_dir=tmp_path / "figures",
    )
    for name, path in paths.items():
        assert path.exists(), f"{name} should exist at {path}"


def test_plot_distribution_shift_error_on_missing_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shift_evaluation"):
        plot_distribution_shift({}, tmp_path / "x.png")

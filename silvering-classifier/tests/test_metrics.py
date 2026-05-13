"""compute_metrics / cost_aware_threshold / evaluate_distribution_shift のテスト。

evaluate_book 第3章ベースの実装。torch 依存なしで動くよう、
FoldPredictions を直接組み立ててテストする。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    FoldPredictions,
    compute_metrics,
    cost_aware_threshold,
    evaluate_distribution_shift,
)


def _make_preds(
    y_true: list[int], y_proba: list[float]
) -> FoldPredictions:
    arr_true = np.array(y_true, dtype=int)
    arr_proba = np.array(y_proba, dtype=float)
    arr_pred = (arr_proba >= 0.5).astype(int)
    return FoldPredictions(y_true=arr_true, y_pred=arr_pred, y_proba=arr_proba)


# ----------------------------- compute_metrics -----------------------------


def test_compute_metrics_perfect_classifier() -> None:
    preds = _make_preds(
        y_true=[0, 0, 0, 1, 1, 1],
        y_proba=[0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    )
    m = compute_metrics(preds)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["tp"] == 3 and m["tn"] == 3
    assert m["fp"] == 0 and m["fn"] == 0


def test_compute_metrics_random_classifier_has_low_mcc() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(size=n)
    preds = FoldPredictions(
        y_true=y_true.astype(int),
        y_pred=(y_proba >= 0.5).astype(int),
        y_proba=y_proba,
    )
    m = compute_metrics(preds)
    assert abs(m["mcc"]) < 0.3
    assert 0.3 < m["roc_auc"] < 0.7


def test_compute_metrics_keys_present() -> None:
    preds = _make_preds([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.6])
    m = compute_metrics(preds)
    expected = {
        "threshold", "accuracy", "precision", "recall", "specificity",
        "f1", "g_mean", "mcc", "roc_auc", "pr_auc",
        "tp", "fp", "fn", "tn", "n",
    }
    assert expected.issubset(m.keys())


def test_compute_metrics_custom_threshold_changes_pred() -> None:
    preds = _make_preds(
        y_true=[0, 0, 1, 1],
        y_proba=[0.1, 0.4, 0.55, 0.9],
    )
    # threshold=0.5 では 0.4 は 0 と判定 → 全て正解
    m_default = compute_metrics(preds, threshold=0.5)
    # threshold=0.35 にすると 0.4 を 1 と誤判定 → FP 発生
    m_low = compute_metrics(preds, threshold=0.35)
    assert m_low["fp"] >= m_default["fp"]


# -------------------------- cost_aware_threshold --------------------------


def test_cost_aware_threshold_lowers_for_high_fn_cost() -> None:
    """FN コストが高い時、閾値は 0.5 より下に動く（=正例として拾いやすくなる）。"""
    rng = np.random.default_rng(1)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    # 正例の確率を 0.55±0.2, 負例を 0.45±0.2 にしてオーバーラップさせる
    y_proba = np.where(
        y_true == 1,
        rng.normal(0.55, 0.2, n),
        rng.normal(0.45, 0.2, n),
    ).clip(0.01, 0.99)
    preds = FoldPredictions(
        y_true=y_true.astype(int),
        y_pred=(y_proba >= 0.5).astype(int),
        y_proba=y_proba,
    )
    res = cost_aware_threshold(preds, cost_fn=10.0, cost_fp=1.0)
    assert res["best_threshold"] <= 0.5
    assert res["best_cost"] < float("inf")


def test_cost_aware_threshold_returns_metrics_at_both_thresholds() -> None:
    preds = _make_preds(
        y_true=[0, 0, 0, 1, 1, 1],
        y_proba=[0.1, 0.4, 0.6, 0.55, 0.8, 0.9],
    )
    res = cost_aware_threshold(preds)
    assert "default_metrics" in res
    assert "optimal_metrics" in res
    assert res["default_metrics"]["threshold"] == 0.5
    assert res["optimal_metrics"]["threshold"] == res["best_threshold"]


def test_cost_aware_threshold_symmetric_cost_keeps_default() -> None:
    """コスト対称 (FN=FP) なら、最適閾値は 0.5 近傍に来る。"""
    rng = np.random.default_rng(2)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_proba = np.where(
        y_true == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n)
    )
    preds = FoldPredictions(
        y_true=y_true.astype(int),
        y_pred=(y_proba >= 0.5).astype(int),
        y_proba=y_proba,
    )
    res = cost_aware_threshold(preds, cost_fn=1.0, cost_fp=1.0)
    assert 0.3 <= res["best_threshold"] <= 0.7


# ----------------------- evaluate_distribution_shift -----------------------


def test_evaluate_distribution_shift_returns_all_ratios() -> None:
    rng = np.random.default_rng(3)
    n = 60
    y_true = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    y_proba = np.where(
        y_true == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n)
    )
    preds = FoldPredictions(
        y_true=y_true,
        y_pred=(y_proba >= 0.5).astype(int),
        y_proba=y_proba,
    )
    res = evaluate_distribution_shift(preds, positive_ratios=[0.1, 0.5, 0.9])
    assert len(res["shift_evaluation"]) == 3
    ratios = [r["positive_ratio"] for r in res["shift_evaluation"]]
    assert ratios == [0.1, 0.5, 0.9]


def test_evaluate_distribution_shift_accuracy_swings_but_auc_stable() -> None:
    """ROC-AUC は分布変化に頑健、accuracy は揺れる、という性質を確認。

    evaluation_book 3.13 の「クラス分布の変化による評価指標への影響」の核心。
    """
    rng = np.random.default_rng(4)
    n = 200
    y_true = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    # 良いが完璧でない分類器
    y_proba = np.where(
        y_true == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n)
    )
    preds = FoldPredictions(
        y_true=y_true,
        y_pred=(y_proba >= 0.5).astype(int),
        y_proba=y_proba,
    )
    res = evaluate_distribution_shift(preds, positive_ratios=[0.1, 0.5, 0.9])
    aucs = [r["roc_auc"] for r in res["shift_evaluation"]]
    # ROC-AUC は分布変化に対して比較的安定
    assert max(aucs) - min(aucs) < 0.15


def test_evaluate_distribution_shift_handles_single_class() -> None:
    preds = _make_preds(
        y_true=[1, 1, 1, 1],
        y_proba=[0.6, 0.7, 0.8, 0.9],
    )
    res = evaluate_distribution_shift(preds)
    assert "error" in res

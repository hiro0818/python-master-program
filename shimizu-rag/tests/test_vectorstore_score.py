"""distance_to_score の数学的性質をテスト。

正規化済みベクターでは ||a-b||² = 2(1 - cos_sim) なので
score = 1 - distance/2 は cos_sim に一致するはず。
"""
from __future__ import annotations

import math

import pytest

from rag.vectorstore import distance_to_score


def test_distance_zero_means_perfect_match() -> None:
    assert distance_to_score(0.0) == 1.0


def test_distance_two_means_opposite() -> None:
    # 正規化済みベクター同士の最大L2²距離は2 (cos_sim=-1)
    assert distance_to_score(2.0) == 0.0


def test_distance_one_means_orthogonal() -> None:
    # cos_sim=0.5 のはずだが、L2² = 2(1 - 0.5) = 1.0
    assert distance_to_score(1.0) == pytest.approx(0.5)


@pytest.mark.parametrize("distance", [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
def test_score_is_clamped_to_unit_interval(distance: float) -> None:
    score = distance_to_score(distance)
    assert 0.0 <= score <= 1.0


def test_score_is_monotonic_decreasing() -> None:
    distances = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
    scores = [distance_to_score(d) for d in distances]
    for prev, nxt in zip(scores, scores[1:]):
        assert prev >= nxt


def test_score_matches_cosine_similarity_identity() -> None:
    # 任意の cos_sim ∈ [-1, 1] について、対応する L2² から復元できる
    for cos_sim in [1.0, 0.7, 0.0, -0.3, -1.0]:
        l2_sq = 2 * (1 - cos_sim)
        assert distance_to_score(l2_sq) == pytest.approx(max(0.0, min(1.0, cos_sim)))

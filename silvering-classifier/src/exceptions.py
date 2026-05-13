"""銀化スコアラーの例外階層。UI/CLIが `except ClassifierError` でまとめて捕捉できる。"""
from __future__ import annotations


class ClassifierError(Exception):
    """銀化分類器が投げる全例外の基底。"""


class DataError(ClassifierError):
    """画像データ・ラベルCSVの不整合。"""


class ModelError(ClassifierError):
    """モデルロード／構築の失敗。"""


class TrainingError(ClassifierError):
    """訓練ループの失敗。"""

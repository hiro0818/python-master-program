"""例外階層が想定通りであることを確認。"""
from __future__ import annotations

from src.exceptions import (
    ClassifierError,
    DataError,
    ModelError,
    TrainingError,
)


def test_all_subclasses_inherit_from_classifier_error() -> None:
    for cls in (DataError, ModelError, TrainingError):
        assert issubclass(cls, ClassifierError)


def test_classifier_error_catches_all_descendants() -> None:
    for exc in (DataError("a"), ModelError("b"), TrainingError("c")):
        try:
            raise exc
        except ClassifierError as caught:
            assert isinstance(caught, type(exc))

"""単画像推論。CLIから `python -m src.predict path/to/image.jpg` で使う。

全 fold モデルの確率を平均（アンサンブル）して安定した予測を返す。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .augment import eval_transform
from .config import CLASSES, MODELS_DIR, N_FOLDS, device
from .dataset import load_image_rgb
from .exceptions import ModelError
from .model import build_model, load_checkpoint


def _available_folds() -> list[int]:
    folds = []
    for k in range(N_FOLDS):
        if (MODELS_DIR / f"baseline_fold{k}.pt").exists():
            folds.append(k)
    return folds


def predict_single(image_path: str | Path) -> dict:
    """1枚の画像に対し、parr/smolt の確率を返す。

    戻り値:
        {
            "image": "...",
            "predicted_class": "smolt",
            "probabilities": {"parr": 0.12, "smolt": 0.88},
            "silvering_score": 0.88,  # smolt 確率を 0..1 の連続スコアとして提供
            "n_models": 5,
        }
    """
    folds = _available_folds()
    if not folds:
        raise ModelError(
            f"学習済みモデルが見つかりません。{MODELS_DIR} に baseline_fold*.pt を置いてください。"
        )

    dev = device()
    rgb = load_image_rgb(image_path)
    tensor = eval_transform()(image=rgb)["image"].unsqueeze(0).to(dev)

    probs_acc = np.zeros(len(CLASSES), dtype=np.float64)
    for k in folds:
        model = build_model(pretrained=False)
        model = load_checkpoint(model, MODELS_DIR / f"baseline_fold{k}.pt", map_location=dev)
        model.to(dev).eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probs_acc += probs

    probs_mean = probs_acc / len(folds)
    pred_idx = int(np.argmax(probs_mean))
    return {
        "image": str(image_path),
        "predicted_class": CLASSES[pred_idx],
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs_mean)},
        "silvering_score": float(probs_mean[CLASSES.index("smolt")]),
        "n_models": len(folds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="銀化スコアを1枚の画像から推論")
    parser.add_argument("image", type=Path, help="推論する画像へのパス")
    args = parser.parse_args()
    if not args.image.exists():
        print(f"❌ ファイルが見つかりません: {args.image}", file=sys.stderr)
        return 2
    result = predict_single(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

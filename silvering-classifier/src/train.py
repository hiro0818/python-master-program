"""少データ向けの訓練パイプライン（Stratified K-Fold CV）。

なぜ k-fold:
    50〜100枚規模で train/val を1回だけ切ると、検証スコアの分散が極端に大きい。
    5-fold で平均すれば「このモデルは本当に銀化を判定できているか」が信頼できる数字になる。

設計:
    - クラス不均衡に対応: WeightedRandomSampler で訓練を、評価は素のループで。
    - 早期停止: 検証 loss が `patience` epoch 改善しなければ打ち切り。
    - 重みは models/baseline_fold{k}.pt として保存。
    - 各 fold の検証メトリクスは metrics.json に統合。
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .augment import eval_transform, train_transform
from .config import (
    BATCH_SIZE,
    CLASSES,
    EPOCHS,
    LR,
    MODELS_DIR,
    N_FOLDS,
    SEED,
    WEIGHT_DECAY,
    device,
)
from .dataset import FishImageDataset, discover_labels
from .exceptions import TrainingError
from .model import build_model, save_checkpoint


@dataclass
class FoldMetrics:
    fold: int
    n_train: int
    n_val: int
    val_loss: float
    val_acc: float
    val_f1: float
    val_auc: float
    best_epoch: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """クラス頻度の逆数でサンプル重みを与える（少数クラスを多めに引く）。"""
    counts = np.bincount(labels)
    weights_per_class = 1.0 / np.maximum(counts, 1)
    sample_weights = [weights_per_class[lbl] for lbl in labels]
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def _train_one_fold(
    train_ds: FishImageDataset,
    val_ds: FishImageDataset,
    fold: int,
    epochs: int,
    lr: float,
    patience: int = 8,
) -> FoldMetrics:
    dev = device()
    model = build_model(pretrained=True).to(dev)

    sampler = _make_weighted_sampler(train_ds.labels)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best = {"loss": float("inf"), "epoch": 0, "state": None}
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"fold{fold} train e{epoch+1}"):
            images = images.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_loss, val_metrics = _evaluate(model, val_loader, dev, criterion)
        tqdm.write(
            f"  fold{fold} e{epoch+1}: val_loss={val_loss:.4f} "
            f"acc={val_metrics['acc']:.3f} f1={val_metrics['f1']:.3f}"
        )

        if val_loss < best["loss"] - 1e-4:
            best.update(
                {"loss": val_loss, "epoch": epoch, "state": _cpu_state(model)}
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"  早期停止 (epoch {epoch+1})")
                break

    if best["state"] is None:
        raise TrainingError(f"fold {fold} で1度も改善しなかった（データを見直して）。")
    model.load_state_dict(best["state"])
    save_checkpoint(model, MODELS_DIR / f"baseline_fold{fold}.pt")

    final_loss, final_metrics = _evaluate(model, val_loader, dev, criterion)
    return FoldMetrics(
        fold=fold,
        n_train=len(train_ds),
        n_val=len(val_ds),
        val_loss=final_loss,
        val_acc=final_metrics["acc"],
        val_f1=final_metrics["f1"],
        val_auc=final_metrics["auc"],
        best_epoch=best["epoch"],
    )


def _cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _evaluate(
    model: nn.Module, loader: DataLoader, dev: str, criterion: nn.Module
) -> tuple[float, dict]:
    model.eval()
    all_loss = 0.0
    n = 0
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(dev)
            labels = labels.to(dev)
            logits = model(images)
            loss = criterion(logits, labels)
            all_loss += loss.item() * images.size(0)
            n += images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(probs.tolist())

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        # 片方のクラスしか val に出なかった等
        metrics["auc"] = float("nan")
    return all_loss / max(n, 1), metrics


def train_cv(epochs: int = EPOCHS, lr: float = LR) -> dict:
    """Stratified K-Fold で訓練を実行し、全 fold のメトリクスを返す。"""
    _set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = discover_labels()
    if len(df) < N_FOLDS * 2:
        raise TrainingError(
            f"画像数が少なすぎます (n={len(df)})。最低 {N_FOLDS * 2} 枚必要。"
        )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    labels = df["label"].map(lambda c: CLASSES.index(c)).values

    fold_metrics: list[FoldMetrics] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, labels)):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        train_ds = FishImageDataset(train_df, transform=train_transform())
        val_ds = FishImageDataset(val_df, transform=eval_transform())

        fm = _train_one_fold(train_ds, val_ds, fold=fold, epochs=epochs, lr=lr)
        fold_metrics.append(fm)

    summary = {
        "n_total": len(df),
        "n_folds": N_FOLDS,
        "folds": [asdict(fm) for fm in fold_metrics],
        "mean": {
            "val_acc": float(np.mean([fm.val_acc for fm in fold_metrics])),
            "val_f1": float(np.mean([fm.val_f1 for fm in fold_metrics])),
            "val_auc": float(np.nanmean([fm.val_auc for fm in fold_metrics])),
        },
    }
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    out = train_cv()
    print(json.dumps(out["mean"], indent=2, ensure_ascii=False))

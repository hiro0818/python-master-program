"""評価: 全 fold の予測を集めて混同行列・指標を計算し、Grad-CAM を描く。

主要指標は ghmagazine/evaluation_book (Apache-2.0) 第3章を参考に整備した。
銀化分類は不均衡データになりやすいため、accuracy 単独では不十分:
  - MCC: 不均衡時のロバストな総合指標
  - PR-AUC: 正例(smolt)が少ない時の本命
  - コスト考慮型閾値: 銀化見落としと誤検出の非対称性を反映

なぜ Grad-CAM:
    「モデルが本当に体表の銀化を見ているのか、それとも撮影背景を覚えただけか」を
    確認できる。配属面談で「説明可能性も付けました」と言える要素。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from .config import (
    CLASSES,
    GRADCAM_DIR,
    MODELS_DIR,
    N_FOLDS,
    SEED,
)


@dataclass
class FoldPredictions:
    """各 fold の検証セットに対する予測結果。"""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray  # smolt クラス(=index 1)の確率


def collect_cv_predictions() -> FoldPredictions:
    """全 fold の検証予測を集約。確率（softmax の smolt 側）も保持する。"""
    import torch
    from sklearn.model_selection import StratifiedKFold

    from .augment import eval_transform
    from .config import device
    from .dataset import FishImageDataset, discover_labels
    from .model import build_model, load_checkpoint

    df = discover_labels()
    labels = df["label"].map(lambda c: CLASSES.index(c)).values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    dev = device()
    transform = eval_transform()

    y_true_all, y_pred_all, y_proba_all = [], [], []
    for fold, (_, va_idx) in enumerate(skf.split(df, labels)):
        ckpt = MODELS_DIR / f"baseline_fold{fold}.pt"
        if not ckpt.exists():
            print(f"⚠️  {ckpt} が無い、スキップ")
            continue
        model = build_model(pretrained=False)
        model = load_checkpoint(model, ckpt, map_location=dev)
        model.to(dev).eval()

        val_df = df.iloc[va_idx].reset_index(drop=True)
        val_ds = FishImageDataset(val_df, transform=transform)
        for image, label in val_ds:
            with torch.no_grad():
                logits = model(image.unsqueeze(0).to(dev))
                proba = torch.softmax(logits, dim=1)[0, 1].item()
                pred = logits.argmax(dim=1).item()
            y_true_all.append(int(label))
            y_pred_all.append(pred)
            y_proba_all.append(proba)

    return FoldPredictions(
        y_true=np.array(y_true_all, dtype=int),
        y_pred=np.array(y_pred_all, dtype=int),
        y_proba=np.array(y_proba_all, dtype=float),
    )


def plot_confusion_matrix(out_path: Path, preds: Optional[FoldPredictions] = None) -> dict:
    """全 fold の検証予測を集約して混同行列を描く。"""
    import matplotlib.pyplot as plt

    preds = preds or collect_cv_predictions()
    cm = confusion_matrix(preds.y_true, preds.y_pred, labels=list(range(len(CLASSES))))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (CV aggregated)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {"confusion_matrix": cm.tolist(), "n": len(preds.y_true)}


def compute_metrics(preds: FoldPredictions, threshold: float = 0.5) -> dict:
    """二値分類の主要指標を一括計算。

    指標の選定は ghmagazine/evaluation_book 3.5/3.8/3.10/3.11 に準拠。

    Args:
        preds: CV で集めた予測結果。
        threshold: y_proba を二値化する閾値。デフォルト 0.5。
    """
    y_true = preds.y_true
    y_proba = preds.y_proba
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    # G-Mean: smolt(正)/parr(負) どちらも均等に評価
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = float(np.sqrt(sensitivity * specificity))

    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "g_mean": g_mean,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n": int(len(y_true)),
    }


def cost_aware_threshold(
    preds: FoldPredictions,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    n_thresholds: int = 91,
) -> dict:
    """非対称コストの下で総コストを最小化する閾値を探索。

    銀化（smolt=正例）の見落としと誤検出はコストが違う:
      - 見落とし (FN): 養殖現場で出荷タイミングを逃す → 損失大
      - 誤検出 (FP): 早期出荷で品質低下 → 損失小〜中

    cost_fn=10, cost_fp=1 はデフォルト仮説（修論で sensitivity analysis 推奨）。
    考え方は ghmagazine/evaluation_book 3.16.4 に準拠。
    """
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_cost = float("inf")
    best_t = 0.5
    history = []

    for t in thresholds:
        y_pred = (preds.y_proba >= t).astype(int)
        fn = int(((preds.y_true == 1) & (y_pred == 0)).sum())
        fp = int(((preds.y_true == 0) & (y_pred == 1)).sum())
        cost = cost_fn * fn + cost_fp * fp
        history.append({"threshold": float(t), "cost": float(cost), "fn": fn, "fp": fp})
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)

    return {
        "best_threshold": best_t,
        "best_cost": float(best_cost),
        "cost_fn": float(cost_fn),
        "cost_fp": float(cost_fp),
        "default_metrics": compute_metrics(preds, threshold=0.5),
        "optimal_metrics": compute_metrics(preds, threshold=best_t),
        "history": history,
    }


def cost_sensitivity_analysis(
    preds: FoldPredictions,
    cost_ratios: Optional[list[float]] = None,
) -> dict:
    """コスト比 (FN/FP) を変えた時の最適閾値と Recall の推移を集計。

    修論で「FN:FP=10:1 はあくまで仮定」という指摘に備えるためのもの。
    複数のコスト比で最適閾値・Recall・FN を並べて、結論の頑健性を示す。
    """
    cost_ratios = cost_ratios or [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    rows = []
    for ratio in cost_ratios:
        res = cost_aware_threshold(preds, cost_fn=ratio, cost_fp=1.0)
        opt = res["optimal_metrics"]
        rows.append(
            {
                "cost_ratio_fn_over_fp": float(ratio),
                "best_threshold": res["best_threshold"],
                "recall": opt["recall"],
                "precision": opt["precision"],
                "f1": opt["f1"],
                "mcc": opt["mcc"],
                "fn": opt["fn"],
                "fp": opt["fp"],
            }
        )
    return {"sensitivity": rows}


def evaluate_distribution_shift(
    preds: FoldPredictions,
    positive_ratios: Optional[list[float]] = None,
    rng_seed: int = SEED,
) -> dict:
    """正例(smolt)比率を変えて指標がどう動くかを検証。

    養殖場ごとに parr/smolt 比率は異なる。本番デプロイ時の頑健性を示すため、
    bootstrap で複数の比率を再現して各指標を計算する。
    視点は ghmagazine/evaluation_book 3.13 に準拠。
    """
    positive_ratios = positive_ratios or [0.1, 0.3, 0.5, 0.7, 0.9]
    rng = np.random.default_rng(rng_seed)

    pos_idx = np.where(preds.y_true == 1)[0]
    neg_idx = np.where(preds.y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {"error": "両クラスのサンプルが必要"}

    # 各比率について、母数を 200 に固定して再サンプル
    target_total = min(200, 2 * min(len(pos_idx), len(neg_idx)) // 1 + len(pos_idx) + len(neg_idx))
    results = []
    for ratio in positive_ratios:
        n_pos = max(int(round(target_total * ratio)), 1)
        n_neg = max(target_total - n_pos, 1)
        sampled_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        sampled_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([sampled_pos, sampled_neg])

        sub = FoldPredictions(
            y_true=preds.y_true[idx],
            y_pred=preds.y_pred[idx],
            y_proba=preds.y_proba[idx],
        )
        m = compute_metrics(sub)
        results.append(
            {
                "positive_ratio": float(ratio),
                "n_total": int(len(idx)),
                "n_positive": int(n_pos),
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "mcc": m["mcc"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
            }
        )

    return {"shift_evaluation": results}


def save_gradcam_examples(
    fold: int = 0, n_per_class: int = 4, out_dir: Optional[Path] = None
) -> None:
    """指定 fold のモデルで、各クラスから n_per_class 枚の Grad-CAM を保存。"""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold

    from .augment import eval_transform
    from .config import device
    from .dataset import discover_labels, load_image_rgb
    from .model import build_model, gradcam_target_layer, load_checkpoint

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("pytorch-grad-cam が無いのでスキップ。pip install pytorch-grad-cam")
        return

    out_dir = out_dir or GRADCAM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device()

    df = discover_labels()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    labels = df["label"].map(lambda c: CLASSES.index(c)).values
    splits = list(skf.split(df, labels))
    _, va_idx = splits[fold]
    val_df = df.iloc[va_idx].reset_index(drop=True)

    ckpt = MODELS_DIR / f"baseline_fold{fold}.pt"
    model = build_model(pretrained=False)
    model = load_checkpoint(model, ckpt, map_location=dev)
    model.to(dev).eval()

    target_layer = gradcam_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])

    transform = eval_transform()
    saved = {cls: 0 for cls in CLASSES}
    for _, row in val_df.iterrows():
        cls = row["label"]
        if saved[cls] >= n_per_class:
            continue

        rgb = load_image_rgb(row["image_path"])
        # eval_transform は Normalize 込み → CAM 描画用には元画像を 0..1 で別途準備
        resized = _resize_for_display(rgb, max_size=448)
        input_tensor = transform(image=rgb)["image"].unsqueeze(0).to(dev)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        # CAM サイズを表示用画像に合わせる
        grayscale_cam_resized = _resize_array_to(grayscale_cam, resized.shape[:2])
        rgb01 = resized.astype(np.float32) / 255.0
        viz = show_cam_on_image(rgb01, grayscale_cam_resized, use_rgb=True)

        out = out_dir / f"fold{fold}_{cls}_{saved[cls]:02d}.png"
        plt.imsave(out, viz)
        saved[cls] += 1
        if all(v >= n_per_class for v in saved.values()):
            break


def _resize_for_display(rgb: np.ndarray, max_size: int) -> np.ndarray:
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return np.array(Image.fromarray(rgb).resize((new_w, new_h)))
    return rgb


def _resize_array_to(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray((arr * 255).astype(np.uint8))
    h, w = target_hw
    img = img.resize((w, h))
    return np.array(img).astype(np.float32) / 255.0


if __name__ == "__main__":
    preds = collect_cv_predictions()

    cm_path = MODELS_DIR / "confusion_matrix.png"
    cm_res = plot_confusion_matrix(cm_path, preds=preds)
    print("=== Confusion matrix ===")
    print(json.dumps(cm_res, indent=2, ensure_ascii=False))

    print("\n=== Metrics @ threshold=0.5 ===")
    print(json.dumps(compute_metrics(preds), indent=2, ensure_ascii=False))

    print("\n=== Cost-aware threshold (FN:FP = 10:1) ===")
    print(json.dumps(cost_aware_threshold(preds), indent=2, ensure_ascii=False))

    print("\n=== Distribution shift evaluation ===")
    print(json.dumps(evaluate_distribution_shift(preds), indent=2, ensure_ascii=False))

    save_gradcam_examples(fold=0)
    print(f"\nGrad-CAM 出力: {GRADCAM_DIR}")

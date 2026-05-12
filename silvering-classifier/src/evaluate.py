"""評価: 全 fold の予測を集めて混同行列を描き、Grad-CAM 可視化を吐く。

なぜ Grad-CAM:
    「モデルが本当に体表の銀化を見ているのか、それとも撮影背景を覚えただけか」を
    確認できる。配属面談で「説明可能性も付けました」と言える要素。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from .augment import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from .config import (
    CLASSES,
    GRADCAM_DIR,
    MODELS_DIR,
    N_FOLDS,
    SEED,
    device,
)
from .dataset import FishImageDataset, discover_labels, load_image_rgb
from .model import build_model, gradcam_target_layer, load_checkpoint


def plot_confusion_matrix(out_path: Path) -> dict:
    """全 fold の検証予測を集約して混同行列を描く。"""
    df = discover_labels()
    labels = df["label"].map(lambda c: CLASSES.index(c)).values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    dev = device()
    transform = eval_transform()

    y_true_all, y_pred_all = [], []
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
                pred = logits.argmax(dim=1).item()
            y_true_all.append(int(label))
            y_pred_all.append(pred)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(CLASSES))))

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

    return {"confusion_matrix": cm.tolist(), "n": len(y_true_all)}


def save_gradcam_examples(
    fold: int = 0, n_per_class: int = 4, out_dir: Optional[Path] = None
) -> None:
    """指定 fold のモデルで、各クラスから n_per_class 枚の Grad-CAM を保存。"""
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
    cm_path = MODELS_DIR / "confusion_matrix.png"
    res = plot_confusion_matrix(cm_path)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    save_gradcam_examples(fold=0)
    print(f"Grad-CAM 出力: {GRADCAM_DIR}")

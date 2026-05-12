"""対話的ラベリングCLI。

data/raw/ 配下に置いた未分類の画像を、1枚ずつターミナルに表示できる場所に
（実際にはファイル名と要約だけ）表示し、ユーザに parr/smolt/skip を選ばせて
data/labels.csv に追記する。

使い方:
    python scripts/02_label_cli.py [画像ディレクトリ]

何もディレクトリ指定がなければ data/raw/unsorted/ を見にいく。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# プロジェクトルートを sys.path に
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASSES, LABELS_CSV, RAW_DIR  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _existing_labeled() -> set[str]:
    if not LABELS_CSV.exists():
        return set()
    out: set[str] = set()
    with LABELS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add(row.get("image_path", ""))
    return out


def _ensure_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "scorer", "notes"])


def _append_label(path: Path, image_path: str, label: str, scorer: str, notes: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([image_path, label, scorer, notes])


def main() -> int:
    parser = argparse.ArgumentParser(description="銀化画像の対話的ラベリング")
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=RAW_DIR / "unsorted",
        help="ラベル付け対象の画像ディレクトリ",
    )
    parser.add_argument("--scorer", default="self", help="採点者ID（自分用は self でOK）")
    args = parser.parse_args()

    if not args.directory.exists():
        print(f"❌ ディレクトリが存在しません: {args.directory}", file=sys.stderr)
        return 2

    _ensure_header(LABELS_CSV)
    labeled = _existing_labeled()

    candidates: list[Path] = []
    for ext in IMAGE_EXTS:
        candidates.extend(sorted(args.directory.glob(f"*{ext}")))
    candidates = [p for p in candidates if str(p.resolve()) not in labeled]

    if not candidates:
        print(f"✅ 全部ラベル済みか、画像がありません ({args.directory})")
        return 0

    print(f"📸 {len(candidates)} 枚をラベリングします。")
    print("操作: 1=parr / 2=smolt / s=skip / q=quit")
    print()

    for i, path in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {path.name}")
        while True:
            ans = input("  ラベル (1/2/s/q): ").strip().lower()
            if ans == "q":
                print("中断します。")
                return 0
            if ans == "s":
                break
            if ans in ("1", "2"):
                label = CLASSES[int(ans) - 1]
                notes = input("  メモ（出典・気付き、任意）: ").strip()
                _append_label(LABELS_CSV, str(path.resolve()), label, args.scorer, notes)
                print(f"  ✅ {label} で記録")
                break
            print("  → 1, 2, s, q のいずれかを入力してください")
    print(f"完了。{LABELS_CSV} を確認してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

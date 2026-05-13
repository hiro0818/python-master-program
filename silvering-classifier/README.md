# 🐟 銀化スコアラー — Salmonid Silvering Classifier

サクラマス／ヤマメ等の銀化（スモルト化）を画像から自動評価するプロトタイプ。
修論「マルチモーダル深層学習によるスモルト化準備度評価システム」の Year 0（配属前）プロトタイプ。

## 背景

清水研RAGとNotebookLMで16論文を解析した結果：

- ❌ **銀化スコアの標準化された定量手法は英語論文に存在しない**
- ❌ **清水研も非侵襲評価を主軸にしていない**
- ✅ **IGFBP-1 と肥満度の負の相関は R² = 0.68 まで出る**（=ML で学習可能）

→ **画像から銀化を定量する**システムは世界的に空白地。修論で攻める価値あり。

## 技術スタック（5基準で評価済）

| レイヤー | 技術 | ライセンス | 役割 |
|---|---|---|---|
| 画像処理 | Pillow + albumentations | MIT/MIT | 読み込み・拡張 |
| モデル | timm (EfficientNet-B0) | Apache 2.0 | ImageNet 転移学習 |
| 学習 | PyTorch | BSD 3-Clause | 訓練ループ |
| 評価 | scikit-learn | BSD 3-Clause | metrics, CV |
| 説明 | Grad-CAM (pytorch-grad-cam) | MIT | 可視化 |

清水研RAGと同じ哲学: **MIT/Apache 2.0 のみ、コア依存はすべて代替可能**。

## ディレクトリ構成

```
silvering-classifier/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── data/
│   ├── raw/                   # 集めた画像（gitignored）
│   │   ├── parr/              # パー型（川型）
│   │   └── smolt/             # スモルト型（銀化降海型）
│   ├── labels.csv             # 統一ラベル CSV
│   └── attribution.csv        # ライセンス・出典記録
├── src/
│   ├── config.py              # パラメータ集約
│   ├── exceptions.py
│   ├── dataset.py             # PyTorch Dataset
│   ├── augment.py             # albumentations 拡張
│   ├── model.py               # timm モデルファクトリ
│   ├── train.py               # 訓練（CV対応）
│   ├── evaluate.py            # 評価 + Grad-CAM
│   └── predict.py             # 単画像推論
├── scripts/
│   ├── 01_collect_guide.md    # 画像収集手順
│   ├── 02_label_cli.py        # 対話的ラベリング
│   └── 03_run_baseline.py     # ベースライン実行
├── tests/                     # pytest
├── models/                    # 学習済み重み（gitignored）
└── notebooks/
```

## クイックスタート

### 1. 依存関係

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 画像を集める（scripts/01_collect_guide.md 参照）

各クラス最低 25 枚以上、合計 50 枚あれば動く。

```bash
data/raw/parr/    # 暗色・斑紋・河川型 の写真
data/raw/smolt/   # 銀白色・流線型 の写真
```

### 3. ラベリング（オプション、自分で目視確認したいとき）

```bash
python scripts/02_label_cli.py
```

### 4. 訓練 + 評価（一発）

```bash
python scripts/03_run_baseline.py
```

出力:
- `models/baseline_fold{0..4}.pt` — k-fold モデル
- `models/confusion_matrix.png` — 混同行列
- `models/gradcam/*.png` — Grad-CAM 可視化（モデルがどこを見ているか）
- `models/metrics.json` — accuracy, F1, AUC
- `models/evaluation_report.json` — 全指標 + 予測値（Notebookで再可視化用）
- `models/figures/*.png` — PR/ROC/コスト/分布シフト/sensitivity 図

### 5. 単画像推論

```bash
python src/predict.py path/to/image.jpg
```

### 6. ダッシュボード（修論プレゼン・配属面談用）

```bash
streamlit run app.py
```

タブ構成:
- **Overview** — 研究概要 + パイプライン図
- **Evaluation** — PR/ROC/コスト/分布シフト/sensitivity の各図
- **Cost Playground** — FN/FP コスト比をスライダーで操作、最適閾値が即更新
- **Try It** — 画像をアップロード → silvering_score + Grad-CAM
- **About** — 設計思想 + 参照文献

Streamlit Cloud にもデプロイ可能（shimizu-rag と同じ構成）。

### 7. 修論Figureだけ別途生成

```bash
jupyter notebook notebooks/evaluation_report.ipynb
```

PNG (200dpi スライド用) と PDF (LaTeX `\includegraphics` 用) を一括出力。
さらに修論本文用LaTeX summary table も自動生成される。

## 評価指標の defendsability

| 指標 | 目的 | 参照 |
|---|---|---|
| MCC | 不均衡データでロバストな総合指標 | 評価指標入門 3.5 |
| PR-AUC | 正例(smolt)が少ない時の本命 | 同 3.11 |
| Cost-aware threshold | 銀化見落とし vs 誤検出の非対称コストを反映 | 同 3.16.4 |
| Distribution shift | 養殖場ごとに parr/smolt 比率が違う想定 | 同 3.13 |
| Cost sensitivity | コスト比仮定の頑健性 | 同上 |

実装は `references/evaluation_book/` (Apache-2.0, gitignored) を参考にしている。

## 配属時の見せ方

1. **「16論文に銀化スコアの標準法が無いと確認しました」** (NotebookLM スクショ)
2. **「画像から自動評価するプロトタイプを作りました」** (混同行列 + Grad-CAM)
3. **「次は実魚で検証してATPase活性と比較したい」** (修論プラン)

これで M1 配属初日に「修論テーマ確定」状態に持っていける。

## 制限と次のステップ

### 現バージョンの限界

- バイナリ分類（parr / smolt）のみ。連続スコアは Phase 2 で。
- 画像はWeb収集のため、撮影条件がバラバラ。本実験では統一撮影プロトコルが必要。
- ATPase 活性・塩水耐性試験との対応データなし（配属後に取得）。

### Phase 2 (配属後)

- [ ] 連続スコア（0-1 or 1-5 段階）への拡張
- [ ] ATPase 活性とのキャリブレーション
- [ ] マルチモーダル統合（血中 IGF-I/IGFBP との結合）
- [ ] 養殖会社での実証

## ライセンス

コード: MIT
画像: 各画像の出典・ライセンスを `data/attribution.csv` に記録。
学習済み重み: 公開時は CC-BY 4.0 予定。

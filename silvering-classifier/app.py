"""銀化分類器 Streamlit ダッシュボード。

修論プレゼン・配属面談用のライブデモ。Streamlit Cloud にもデプロイ可能。

タブ構成:
  1. Overview          - 研究概要、パイプライン図
  2. Evaluation        - 評価レポート（PR/ROC/コスト/分布シフト）
  3. Cost Playground   - コスト比をスライダーで操作して最適閾値を更新
  4. Try It            - 画像をアップロード → silvering_score + Grad-CAM
  5. About             - 設計思想・参照文献

起動:
  streamlit run silvering-classifier/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    FoldPredictions,
    compute_metrics,
    cost_aware_threshold,
)
from src.visualize import (  # noqa: E402
    plot_cost_sensitivity,
    plot_cost_threshold_curve,
    plot_distribution_shift,
    plot_metric_summary_bar,
    plot_pr_curve,
    plot_roc_curve,
)

st.set_page_config(
    page_title="Silvering Classifier",
    page_icon="🐟",
    layout="wide",
)


def _expected_password() -> str | None:
    """環境変数 or Streamlit secrets からパスワードを取得。
    どちらも未設定なら認証スキップ（ローカル開発用）。
    """
    pw = os.environ.get("APP_PASSWORD")
    if pw:
        return pw
    try:
        return st.secrets.get("APP_PASSWORD")  # type: ignore[no-any-return]
    except (FileNotFoundError, KeyError, AttributeError):
        return None


def require_password() -> None:
    """パスワード認証ゲート。未認証なら入力フォームを出して st.stop() する。"""
    expected = _expected_password()
    if expected is None:
        return  # パスワード未設定 = ローカル/オープン環境
    if st.session_state.get("authenticated"):
        return

    st.title("🔒 Silvering Classifier")
    st.markdown("研究室メンバー限定のダッシュボードです。パスワードを入力してください。")
    with st.form("auth", clear_on_submit=True):
        pw = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("入る")
    if submitted:
        if pw == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    st.stop()


require_password()


REPORT_PATH = ROOT / "models" / "evaluation_report.json"
DEMO_REPORT_PATH = ROOT / "models" / "demo_evaluation_report.json"


@st.cache_data
def load_report() -> tuple[dict | None, bool]:
    """評価レポートを読む。本物が無ければデモにフォールバック。

    Returns:
        (report_dict, is_demo)
    """
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text(encoding="utf-8")), False
    if DEMO_REPORT_PATH.exists():
        return json.loads(DEMO_REPORT_PATH.read_text(encoding="utf-8")), True
    return None, False


def preds_from_report(report: dict) -> FoldPredictions:
    p = report["predictions"]
    return FoldPredictions(
        y_true=np.array(p["y_true"]),
        y_pred=np.array(p["y_pred"]),
        y_proba=np.array(p["y_proba"]),
    )


# ------------------------------- Sidebar -------------------------------

st.sidebar.title("🐟 Silvering Classifier")
st.sidebar.markdown(
    "サクラマス・ヤマメの**スモルト化準備度**を画像から推定する深層学習プロトタイプ。"
)
tab = st.sidebar.radio(
    "ページ",
    ["Overview", "Evaluation", "Cost Playground", "Try It", "About"],
)

report, is_demo = load_report()
if report is None:
    st.sidebar.warning(
        "`models/evaluation_report.json` が見つかりません。"
        "`python scripts/03_run_baseline.py` か `scripts/04_generate_demo_report.py` を実行してください。"
    )
elif is_demo:
    st.sidebar.info("📊 デモデータ表示中（合成予測値）。実データ訓練後に自動で切り替わります。")

# ------------------------------- Overview -------------------------------

if tab == "Overview":
    st.title("🐟 銀化（スモルト化）分類器")
    st.markdown(
        """
**動機:**
NotebookLM による先行研究分析の結果、銀化（パー→スモルト変態）の定量評価には
英文文献全16本でも標準が存在しない。視覚評価に依存している現状を、
画像から客観的な `silvering_score ∈ [0, 1]` を返す深層学習で代替する。

**パイプライン:**
"""
    )
    st.code(
        """
images (raw)
   │ albumentations (HFlip/Rotate±15°/HSV/etc.  ※VFlipは禁止)
   ▼
EfficientNet-B0 (ImageNet pretrained, timm)
   │ Stratified 5-fold CV + WeightedRandomSampler
   ▼
ensemble across folds → silvering_score
   │ + Grad-CAM (explainability)
   │ + MCC / PR-AUC / cost-aware threshold
   ▼
clinical report
""",
        language="text",
    )

    st.markdown("**主要な技術選択:**")
    st.table(
        {
            "項目": [
                "バックボーン",
                "augmentation",
                "クラス不均衡",
                "主指標",
                "閾値",
                "説明可能性",
            ],
            "選択": [
                "EfficientNet-B0 (timm)",
                "albumentations / VFlipなし",
                "WeightedRandomSampler",
                "MCC + PR-AUC",
                "cost-aware (FN:FP=10:1)",
                "Grad-CAM",
            ],
            "根拠": [
                "5.3M params で小データに適合",
                "背腹の銀化勾配を保持",
                "正例(smolt)が少ない想定",
                "不均衡データでロバスト",
                "養殖場での見落としコストを反映",
                "「体表を見ている」ことの検証",
            ],
        }
    )

# ------------------------------- Evaluation -------------------------------

elif tab == "Evaluation":
    st.title("📊 評価レポート")
    if report is None:
        st.error("評価レポートが見つかりません。先にベースラインを実行してください。")
        st.stop()
    if is_demo:
        st.info("📊 **デモデータ表示中** — 合成予測値での見た目確認用。実データ訓練後は本物の評価値に切り替わります。")

    preds = preds_from_report(report)
    metrics = report["metrics_at_0.5"]
    cost = report["cost_optimization"]
    shift = report["distribution_shift"]
    sensitivity = report.get("cost_sensitivity")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MCC", f"{metrics['mcc']:.3f}")
    col2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    col3.metric("F1", f"{metrics['f1']:.3f}")
    col4.metric("n samples", f"{metrics['n']}")

    st.subheader("主要指標サマリー")
    st.pyplot(plot_metric_summary_bar(metrics))

    col_pr, col_roc = st.columns(2)
    with col_pr:
        st.subheader("Precision-Recall 曲線")
        st.pyplot(plot_pr_curve(preds))
    with col_roc:
        st.subheader("ROC 曲線")
        st.pyplot(plot_roc_curve(preds))

    st.subheader("コスト考慮型閾値最適化（FN:FP = 10:1）")
    st.pyplot(plot_cost_threshold_curve(cost))
    st.markdown(
        f"- デフォルト 0.5 → Recall **{cost['default_metrics']['recall']:.3f}**, FN={cost['default_metrics']['fn']}"
        f"\n- 最適 {cost['best_threshold']:.3f} → Recall **{cost['optimal_metrics']['recall']:.3f}**, FN={cost['optimal_metrics']['fn']}"
    )

    if sensitivity:
        st.subheader("コスト比 sensitivity analysis")
        st.pyplot(plot_cost_sensitivity(sensitivity))

    st.subheader("分布シフト評価（正例比率を 0.1〜0.9 に変動）")
    st.pyplot(plot_distribution_shift(shift))

    with st.expander("Raw JSON"):
        st.json(report)

# ------------------------------- Cost Playground -------------------------------

elif tab == "Cost Playground":
    st.title("💰 コスト比 Playground")
    st.markdown(
        "**FN/FP のコスト比をスライダーで操作**し、最適閾値・Recall がどう変動するかを試す。\n"
        "修論の sensitivity analysis 表の根拠を、その場で生成できる。"
    )
    if report is None:
        st.error("評価レポートが見つかりません。")
        st.stop()
    if is_demo:
        st.info("📊 デモデータ表示中。")

    preds = preds_from_report(report)

    col_a, col_b = st.columns(2)
    with col_a:
        cost_fn = st.slider("Cost of False Negative (見落とし)", 1.0, 50.0, 10.0, step=1.0)
    with col_b:
        cost_fp = st.slider("Cost of False Positive (誤検出)", 1.0, 50.0, 1.0, step=1.0)

    res = cost_aware_threshold(preds, cost_fn=cost_fn, cost_fp=cost_fp)

    c1, c2, c3 = st.columns(3)
    c1.metric("最適閾値", f"{res['best_threshold']:.3f}")
    c2.metric("Recall (最適)", f"{res['optimal_metrics']['recall']:.3f}")
    c3.metric("FN (最適)", res["optimal_metrics"]["fn"])

    st.pyplot(plot_cost_threshold_curve(res))

    st.markdown("### Default (0.5) vs Optimal の差分")
    df_dict = {
        "Metric": ["Recall", "Precision", "F1", "MCC", "FN", "FP"],
        "Default (0.5)": [
            f"{res['default_metrics']['recall']:.3f}",
            f"{res['default_metrics']['precision']:.3f}",
            f"{res['default_metrics']['f1']:.3f}",
            f"{res['default_metrics']['mcc']:.3f}",
            res["default_metrics"]["fn"],
            res["default_metrics"]["fp"],
        ],
        f"Optimal ({res['best_threshold']:.2f})": [
            f"{res['optimal_metrics']['recall']:.3f}",
            f"{res['optimal_metrics']['precision']:.3f}",
            f"{res['optimal_metrics']['f1']:.3f}",
            f"{res['optimal_metrics']['mcc']:.3f}",
            res["optimal_metrics"]["fn"],
            res["optimal_metrics"]["fp"],
        ],
    }
    st.table(df_dict)

# ------------------------------- Try It -------------------------------

elif tab == "Try It":
    st.title("🐟 画像から銀化スコアを推定")
    st.markdown(
        "魚画像をアップロードすると `silvering_score ∈ [0, 1]` を返します（"
        "1 に近いほど smolt 寄り）。"
    )

    uploaded = st.file_uploader(
        "画像をアップロード", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded is None:
        st.info("画像をアップロードしてください。")
    else:
        from PIL import Image

        image = Image.open(uploaded).convert("RGB")
        col_img, col_res = st.columns([2, 3])
        with col_img:
            st.image(image, caption="入力画像", width="stretch")

        try:
            from src.predict import predict_single

            tmp_path = ROOT / "models" / "_tmp_upload.png"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(tmp_path)
            result = predict_single(tmp_path)
            tmp_path.unlink(missing_ok=True)

            with col_res:
                st.metric(
                    "🥈 Silvering Score",
                    f"{result['silvering_score']:.3f}",
                    delta=f"{result['silvering_score'] - 0.5:+.3f} from neutral",
                )
                st.write(f"**Predicted:** {result['predicted_class']}")
                st.write(f"**Ensemble size:** {result['n_models']} folds")
                st.progress(result["silvering_score"])
                st.json(result["probabilities"])
        except Exception as e:  # noqa: BLE001
            with col_res:
                st.error(
                    "推論失敗。学習済みモデル `models/baseline_fold*.pt` が必要です。"
                )
                st.code(str(e))

# ------------------------------- About -------------------------------

elif tab == "About":
    st.title("📚 About")
    st.markdown(
        """
### 設計思想
- **小データレジーム**: 50〜200枚の銀化画像という制約に最適化（Stratified K-Fold + WeightedRandomSampler）
- **defendable な評価**: Accuracy だけでなく MCC / PR-AUC / コスト考慮を併用
- **説明可能性**: Grad-CAM でモデルが体表を見ていることを検証
- **養殖場デプロイ想定**: 分布シフトと非対称コストを評価に組み込み

### 参照
- 評価指標: [ghmagazine/evaluation_book](https://github.com/ghmagazine/evaluation_book) (Apache-2.0)
- 書籍: 高柳慎一・長田怜士『評価指標入門』技術評論社, 2023
- 北大: 北海道大学水産学部 清水研究室（サクラマス・ヤマメ生態）

### 修論テーマ案
「マルチモーダル深層学習によるサクラマス／ヤマメのスモルト化準備度評価システム」
"""
    )

    with st.expander("Tech Stack"):
        st.code(
            """
PyTorch 2.x         深層学習フレームワーク
timm                EfficientNet-B0 (ImageNet 1k pretrained)
albumentations      画像 augmentation
pytorch-grad-cam    説明可能性
scikit-learn        評価指標
matplotlib          可視化
Streamlit           ダッシュボード
"""
        )

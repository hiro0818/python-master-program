# 銀化画像収集ガイド

## 目標
- **parr（パー型）**: 25〜50 枚
- **smolt（スモルト型）**: 25〜50 枚
- 合計 **50〜100 枚** で初回プロトタイプは十分（augmentation で実効サイズが10倍以上に拡張される）

## 何を集めるか

| クラス | 特徴 | キーワード（検索用） |
|---|---|---|
| **parr** | 暗色・側面にパーマーク（斑紋）・丸みのある体型 | "Oncorhynchus masou parr", "ヤマメ parr mark", "salmon parr", "サクラマス 河川型" |
| **smolt** | 銀白色・流線型・パーマーク消失 | "Oncorhynchus masou smolt", "salmon smolt silver", "サクラマス スモルト", "降海型" |

## ライセンスがクリーンな画像ソース（推奨順）

### 1. Wikimedia Commons（最優先）
- URL: https://commons.wikimedia.org/
- 検索キーワード: "Oncorhynchus masou", "Salmo salar smolt", "Atlantic salmon parr"
- ライセンス: CC BY-SA / Public Domain → **商用利用も改変もOK**
- 注意: 必ず author を `attribution.csv` に記録

### 2. iNaturalist（研究向け）
- URL: https://www.inaturalist.org/
- 検索: "Oncorhynchus masou" → Photos
- ライセンス確認: 各写真ページ右下、CC BY-NC は研究OK・商用NG
- 大量にあるが、クオリティは混在

### 3. Flickr（CC フィルタ必須）
- URL: https://www.flickr.com/search/?text=salmon+smolt&license=2%2C3%2C4%2C5%2C6%2C9
- 上のリンクは CC ライセンスでフィルタ済み
- CC0 / CC BY / CC BY-SA を選ぶ

### 4. GBIF（学術データベース）
- URL: https://www.gbif.org/species/2424684 （Oncorhynchus masou）
- 各観察記録にメディアが付いている場合あり
- ライセンス記載が明確

### 5. 論文の Figure（清水研含む）
- ライセンス的にはグレー。**配布せず、自分の手元での研究用に限る**
- Open Access 論文（Frontiers, PLOS, eLife 等）は本文中の Figure も CC ライセンス
- 必ず論文の出典を `attribution.csv` に記録

## 何を集めては**いけない**

- ❌ 「All Rights Reserved」の画像（Google 画像検索の素のもの等）
- ❌ ライセンス記載のない個人ブログの写真
- ❌ 釣り雑誌・商業サイトの写真
- ❌ AI 生成画像（テスト汚染になる）
- ❌ 死魚・解剖写真（生体の体表色を学習したい）
- ❌ 真上から / 真下からの写真（側面写真でないと銀化の判定に向かない）

## 画像の質のチェックリスト

- [ ] 魚が画面の **30%以上** を占めている
- [ ] **側面** が写っている
- [ ] **体全体** が見えている（頭から尾まで）
- [ ] **背景がシンプル**（複雑だと過学習する）
- [ ] **照明** が極端でない（白飛び・黒つぶれを避ける）

## 配置方法

```bash
# parr 型
silvering-classifier/data/raw/parr/wiki_001.jpg
silvering-classifier/data/raw/parr/inat_042.jpg
...

# smolt 型
silvering-classifier/data/raw/smolt/wiki_023.jpg
silvering-classifier/data/raw/smolt/flickr_087.jpg
...
```

ファイル名のプレフィックスで出典が分かるようにしておくと後で楽。

## 出典・ライセンス記録（必須）

`data/attribution.csv` に1枚ごとに1行ずつ記録：

```csv
filename,source,url,author,license,collected_at,notes
wiki_001.jpg,Wikimedia,https://commons.wikimedia.org/wiki/File:Foo.jpg,User:Bar,CC BY-SA 4.0,2026-05-12,headshot crop
inat_042.jpg,iNaturalist,https://inaturalist.org/observations/12345,Tanaka,CC BY-NC 4.0,2026-05-12,
```

**これを残さないと修論・論文化のときに死にます。** 30秒の手間を惜しまない。

## 推奨ワークフロー（半日で 50 枚）

| 時間 | やること |
|---|---|
| 0〜30分 | Wikimedia で "salmon smolt" "salmon parr" 検索、CC0/BY/BY-SA のみ抽出 |
| 30〜90分 | iNaturalist で "Oncorhynchus masou" "Salmo salar" 検索 |
| 90〜120分 | Flickr CC フィルタで補充 |
| 120〜150分 | 質チェック（リサイズ・トリミング） |
| 150〜180分 | `attribution.csv` 記入 |

## 次のステップ

集まったら：

```bash
cd silvering-classifier
pip install -r requirements.txt
python scripts/03_run_baseline.py
```

5-fold CV で訓練 + 混同行列 + Grad-CAM までワンコマンドで走ります。

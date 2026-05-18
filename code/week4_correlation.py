# Week 4 Day 18: 相関係数を numpy で実感する
# 3つのデモで「相関 ≠ 因果」を体験する

import numpy as np

np.random.seed(42)  # 結果を再現可能にするための呪文(乱数の種)
n = 10_000

print("=" * 60)
print("Demo 1: 完全にランダムな2変数")
print("=" * 60)
x_random = np.random.randn(n)
y_random = np.random.randn(n)
r1 = np.corrcoef(x_random, y_random)[0, 1]
print(f"  r = {r1:.4f}")
print(f"  → ランダム同士は連動しない、ほぼ 0")
print()

print("=" * 60)
print("Demo 2: 真の正の相関(y = 0.7x + ノイズ)")
print("=" * 60)
x = np.random.randn(n)
noise = np.random.randn(n) * 0.5
y_correlated = 0.7 * x + noise
r2 = np.corrcoef(x, y_correlated)[0, 1]
print(f"  r = {r2:.4f}")
print(f"  → 実際に連動している場合は強い相関(0.7〜0.85)")
print()

print("=" * 60)
print("Demo 3: 見せかけの相関(第三変数Zが両方を動かす)")
print("=" * 60)
# Z = 気温(隠れた交絡因子)
z = np.random.randn(n)
# a = アイス売上(気温で決まる)
a = 0.6 * z + np.random.randn(n) * 0.3
# b = 水死者数(気温で決まる)
b = 0.7 * z + np.random.randn(n) * 0.3
# a と b の間に "直接の因果" はゼロ
r3 = np.corrcoef(a, b)[0, 1]
print(f"  r(アイス売上, 水死者数) = {r3:.4f}")
print(f"  → a と b の間に因果は無いのに、強い相関が出る!")
print(f"  → 共通の原因 Z(気温)が両方を引き起こすため")
print()

print("=" * 60)
print("Demo 4: Z を条件として 固定 すると相関が消える")
print("=" * 60)
# Z(気温)が一定の集団(例:夏の中の気温30度近辺だけ)を取り出す
mask = (z > 0.5) & (z < 0.6)  # 気温の "薄いスライス" だけ抽出
a_slice = a[mask]
b_slice = b[mask]
r4 = np.corrcoef(a_slice, b_slice)[0, 1]
print(f"  Z を固定した中での r(アイス, 水死) = {r4:.4f}")
print(f"  サンプル数: {mask.sum()} 件")
print(f"  → 気温を一定にすると、見せかけの相関が消える")
print(f"  → これが 交絡因子を統制する という統計の発想")

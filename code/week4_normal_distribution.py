# Week 4 Day 16: 正規分布の 68-95-99.7 ルールを検証する
# numpy で 10万個サンプル → ±1σ/±2σ/±3σ の比率を数える

import numpy as np

# 平均 0、標準偏差 1 の正規分布から 10万個サンプル
n = 100_000
mu = 0      # 平均
sigma = 1   # 標準偏差

# np.random.normal(平均, std, 個数)
data = np.random.normal(mu, sigma, n)

# 各範囲に入っている個数を数える
within_1sigma = np.sum(np.abs(data) < 1 * sigma)
within_2sigma = np.sum(np.abs(data) < 2 * sigma)
within_3sigma = np.sum(np.abs(data) < 3 * sigma)

print(f"=== 10万個サンプルの実測 vs 理論値 ===")
print()
print(f"±1σ の範囲に入った個数: {within_1sigma:>6,} 個")
print(f"  実測: {within_1sigma/n*100:.2f}%   理論: 68.27%")
print()
print(f"±2σ の範囲に入った個数: {within_2sigma:>6,} 個")
print(f"  実測: {within_2sigma/n*100:.2f}%   理論: 95.45%")
print()
print(f"±3σ の範囲に入った個数: {within_3sigma:>6,} 個")
print(f"  実測: {within_3sigma/n*100:.2f}%   理論: 99.73%")
print()

# おまけ:実際のデータの平均と std を確認
print(f"=== サンプルの統計量(理論との一致確認) ===")
print(f"平均 (理論 0):     {np.mean(data):.4f}")
print(f"std  (理論 1):     {np.std(data):.4f}")


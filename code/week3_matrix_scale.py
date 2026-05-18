# Week 3 復習: numpy のスケール感を体感する
# 100万人 × 100属性 の行列積を1行で

import numpy as np
import time

# 100万人、100属性のランダムデータ
n_users = 1_000_000
n_features = 100

players = np.random.rand(n_users, n_features)   # 100万 × 100 の行列
weights = np.random.rand(n_features)            # 100次元の重み

# 行列積を1行
start = time.perf_counter()
scores = players @ weights
elapsed = time.perf_counter() - start

# 比較: もし手計算したら何回の掛け算が必要か
total_multiplications = n_users * n_features

print(f"データサイズ: {n_users:,}人 × {n_features}属性")
print(f"必要な掛け算数: {total_multiplications:,} 回")
print(f"numpy の所要時間: {elapsed*1000:.2f} ミリ秒")
print(f"1秒あたりに処理した掛け算数: {total_multiplications/elapsed/1e9:.2f} 億回/秒")
print()
print(f"スコアの最初の5人:")
print(scores[:5])

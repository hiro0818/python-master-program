# Week 3 復習: 二分探索 vs 線形探索
# ソート済みなら bisect で爆速

import time
import bisect

# 100万件のソート済みリスト
n = 1_000_000
data = list(range(n))   # 0, 1, 2, ..., 999_999(ソート済み)
target = 999_999        # 一番最後の数字を探す(意地悪)

# 方法1: in 演算子(線形探索 O(n))
start = time.perf_counter()
found = target in data
linear_time = time.perf_counter() - start

# 方法2: bisect(二分探索 O(log n))
start = time.perf_counter()
i = bisect.bisect_left(data, target)
found = i < len(data) and data[i] == target
binary_time = time.perf_counter() - start

print(f"線形探索 (in):     {linear_time*1_000_000:>10.2f} マイクロ秒")
print(f"二分探索 (bisect): {binary_time*1_000_000:>10.2f} マイクロ秒")
print(f"差: 二分探索の方が {linear_time/binary_time:.0f} 倍速い")
print()
print(f"理論値: log2(1,000,000) ≈ 20 ステップで終わる")

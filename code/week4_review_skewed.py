# 正規分布 と 対数正規分布 を比べる

import numpy as np

np.random.seed(42)
n = 100_000

# 正規分布(第2問で君が信じたやつ)
normal_income = np.random.normal(600, 100, n)

print("【正規分布】")
print(f"  平均:     {np.mean(normal_income):.0f} 万円")
print(f"  中央値:   {np.median(normal_income):.0f} 万円")
print(f"  最大値:   {np.max(normal_income):.0f} 万円")
print(f"  800万以上: {np.sum(normal_income > 800) / n * 100:.2f}%")
print()

# 対数正規分布(現実の年収に近い形)
lognormal_income = np.random.lognormal(mean=6.3, sigma=0.5, size=n)

print("【対数正規分布】")
print(f"  平均:     {np.mean(lognormal_income):.0f} 万円")
print(f"  中央値:   {np.median(lognormal_income):.0f} 万円")
print(f"  最大値:   {np.max(lognormal_income):.0f} 万円")
print(f"  800万以上: {np.sum(lognormal_income > 800) / n * 100:.2f}%")

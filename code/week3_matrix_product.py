# Week 3 復習: 行列積を numpy で1行
# Day 14 のテスト問4 を再現する

import numpy as np

# 選手データ(2行2列): [得点, リバウンド]
players = np.array([
    [18, 6],   # 田中
    [24, 4],   # 佐藤
])

# 重みベクトル: [得点の重要度, RBの重要度]
weights = np.array([0.6, 0.4])

# 行列積を1行で(@ は行列積演算子)
scores = players @ weights

print("選手行列:")
print(players)
print()
print("重み:", weights)
print()
print("総合力スコア:", scores)
print()
print("田中:", scores[0])
print("佐藤:", scores[1])

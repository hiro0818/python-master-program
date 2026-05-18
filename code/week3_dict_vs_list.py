# Week 3 復習: dict の威力を体感する
# list 線形探索 vs dict ハッシュ検索の速度差を測る

import time

# 1万人の選手データを用意
players_list = [(f"player_{i}", i*2, i, i+1) for i in range(10000)]
players_dict = {f"player_{i}": (i*2, i, i+1) for i in range(10000)}

# 探したい選手(意地悪に最後の人)
target = "player_9999"

# 方法1: list を線形探索 O(n)
start = time.perf_counter()
for name, pts, rb, ast in players_list:
    if name == target:
        result = (pts, rb, ast)
        break
list_time = time.perf_counter() - start

# 方法2: dict でハッシュ検索 O(1)
start = time.perf_counter()
result = players_dict[target]
dict_time = time.perf_counter() - start

print(f"list 線形探索: {list_time*1_000_000:.2f} マイクロ秒")
print(f"dict ハッシュ: {dict_time*1_000_000:.2f} マイクロ秒")
print(f"差: dict の方が {list_time/dict_time:.0f} 倍速い")

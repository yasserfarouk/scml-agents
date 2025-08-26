# import numpy as np
# # nはラウンド数の平均
# def make_random_quantity(n):
#     xs = np.linspace(0, 1, 100)
#     skew = (n - 10) / 10  # -1〜1 の範囲

#     if abs(skew) < 1e-8:  # 厳密に中立にする
#         weights = np.ones_like(xs)
#     elif skew > 0:
#         weights = np.power(xs - xs.min(), 1 + 1 * skew)
#     else:
#         weights = np.power(xs.max() - xs, 1 - 1 * skew)

#     weights /= weights.sum()  # ★ここを追加

#     return np.random.choice(xs, p=weights)

# print(len([_ for _ in [make_random_quantity(10) for _ in range(1000)] if _ < 0.5]), len([_ for _ in [make_random_quantity(3) for _ in range(1000)] if _ >= 0.5]))

# def make_random_price(n):
#     # nは0~1の範囲の値
#     # 値が大きいほど1に近づくように重みづけさせたもの
#     xs = np.linspace(0, 1, 100)
#     skew = (n - 0.5) * 2  # -1〜1 の範囲
#     if abs(skew) < 1e-8:  # 厳密に中立にする
#         weights = np.ones_like(xs)
#     elif skew > 0:
#         weights = np.power(xs - xs.min(), 1 + 2 * skew)
#     else:
#         weights = np.power(xs.max() - xs, 1 - 2 * skew)
#     weights /= weights.sum()
#     random_num = np.random.choice(xs, p=weights)
#     if random_num < 0.5:
#         return 0
#     else:
#         return 1

# print(len([_ for _ in [make_random_price(0.3) for _ in range(1000)] if _ == 0]), len([_ for _ in [make_random_price(0.3) for _ in range(1000)] if _ == 1]))

# import random
# from collections import Counter

# # 例: 各エージェントのスコア辞書
# scores = {
#     "agent_A": 10,
#     "agent_B": 30,
#     "agent_C": 60,
# }

# # 合計スコアを計算
# sum_score = sum(scores.values())

# # トータルの needs（例えば10単位）のうち、
# # 各単位をスコア比率でランダムにエージェントへ割り振る
# needs = 1000  # ここは必要に応じて変更してください

# # random.choices を使って重みつきサンプリング
# agents      = list(scores.keys())
# weights     = list(scores.values())
# selected    = random.choices(agents, weights=weights, k=needs)

# # 割り振り結果をカウント
# allocation = Counter(selected)

# print("needs =", needs)
# print("allocation:", dict(allocation))

# partners = ("agent_A", "agent_B")  # 例: パートナーのリスト

#         # partnersに存在するpartnerのスコアの合計を取得
# total_score = sum(scores[p] for p in partners if p in scores)
#         # 各エージェントに対して、socores/total_scoreを使ってランダムにneedsを分配
# weights = list(scores[partner] for partner in partners if partner in scores)
# selected = random.choices(partners, weights=weights, k=needs)
# allocation = Counter(selected)

# print("allocation:", dict(allocation))

# dist = {1: [0, 1, 2, 3], 2: [4, 5, 6, 7], 3: [8, 9, 10, 11]}
# print(1 in dist)
# print(4 in dist)

# partners = ["agent_A", "agent_B", "agent_C", "agent_D"]

# def distribute_needs_by_score(partners, needs):
#         """
#         パートナーのスコアに基づいて、needsをパートナーに分配する。
#         """
#         if not partners or needs <= 0:
#             return {partner: 0 for partner in partners}
#         scores = {"agent_A": 10, "agent_B": 30, "agent_C": 60}  # 例: 各エージェントのスコア辞書
#         weights = [scores[partner] for partner in partners if partner in scores]
#         if not weights or len(weights) != len(partners):
#             # スコアが無いパートナーがいる場合は均等分配
#             weights = [1] * len(partners)
#         selected = random.choices(partners, weights=weights, k=needs)
#         allocation = Counter(selected)
#         return dict(allocation)

# dict = {"agent_A": 10, "agent_B": 30, "agent_C": 60}  # 例: 各エージェントのスコア辞書
# print("agent_A" in dict)

# def _current_threshold(r:float):
#         mn, mx = 0, 10 // 2
#         return mn + (mx - mn) * (r**4.0)

# for i in range(10):
#     r = i / 10
#     print(f"r: {r}, threshold: {_current_threshold(r)}")

# offers = {
#     "agent_A": { "unit_price": 0.5, "quantity": 10 },
#     "agent_B": { "unit_price": 0.7, "quantity": 20 },
#     "agent_C": { "unit_price": 0.6, "quantity": 15 }
# }

# for partner in partners:
#         if partner not in offers.keys():
#             offers[partner] = {"unit_price": 0, "quantity": 0}
# unit_price_dict = {
#         partner: offers[partner]["unit_price"] for partner in partners
#             }
# print(unit_price_dict)
# for partner in partners:
#     if partner not in offers.keys():
#         offers[partner] = (0, 0)
# should_distribute_quantity = -20
# # partnerのUNIT_PRICEで重み付けを行う（言い方が悪い法の２倍の確率で選ばれる）

# weights = {partner: offers[partner]["unit_price"] + 1 for partner in partners if partner in offers.keys()}
# total_weight = sum(weights.values())

# # 各エージェントに割り当てる整数値を計算
# raw_distribution = {agent: should_distribute_quantity * w / total_weight for agent, w in weights.items()}
# int_distribution = {agent: int(round(val)) for agent, val in raw_distribution.items()}

# distribution = int_distribution

# print(distribution)

# for i in range(10):
#     print(random.randint(1,2))

import numpy as np

def needs_parameter_sigmoid(needs, k=0.2, needs0=1, param_max=6.0, param_min=1.2):
    """
    needs: 必要量（1以上）
    k: シグモイドの傾き（大きいほど急激に切り替わる）
    needs0: シグモイドの中心（このneedsで中間値）
    param_max: needsが最小のときのパラメータ最大値
    param_min: needsが大きいときのパラメータ最小値（1.2以上にする）
    """
    s = 1 / (1 + np.exp(k * (needs - needs0)))
    param = param_min + (param_max - param_min) * s
    return param

for i in range(50):
     pass # print(f"num;{i}")
     pass # print(needs_parameter_sigmoid(i, k=0.2, needs0=1, param_max=6.0, param_min=1.2))
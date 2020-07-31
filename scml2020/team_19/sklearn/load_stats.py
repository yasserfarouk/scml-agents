import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path


def get_stats(directory):
    for f in Path(directory).glob("**/stats.csv"):
        yield f


# print(Path(__file__).resolve())
path_dir = Path(__file__).parent / "../worlds"  # 提出時はどういうパスにすればいい？
# print(path_dir.resolve())
# print(list(Path(path_dir).glob("**/stats.csv")))

stats = None
counter = 0
for path in get_stats(path_dir):
    counter += 1
    if counter > 3000:
        break

    stats_step = np.full(
        (200, 44), np.nan
    )  # 最大ステップ数200に合わせる, 製品番号数によっては44で足りなくなる（データ更新したらエラー起こりうるから注意）
    # print(stats_step.shape)
    # print(stats_step)

    print("LOADING", counter, path)
    read = np.genfromtxt(path, delimiter=",", skip_header=1)  # numpyによるCSVからの行列生成
    # print(read.shape)
    # print(read)

    stats_step[: read.shape[0], :5] = read[:, :5]  # n_bankruptより前のデータ
    stats_step[: read.shape[0], 5 : read.shape[1] - 21] = read[
        :, 5:-21
    ]  # ステップごとに可変長のデータ（trading_price_0など）
    stats_step[: read.shape[0], -21:] = read[:, -21:]  # productivityより後ろのデータ
    # print(stats_step)

    if stats is None:
        stats = stats_step
    else:
        stats = np.block([[[stats]], [[stats_step]]])

print(stats)

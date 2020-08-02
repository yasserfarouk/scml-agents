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

stats_X, stats_y = None, None
counter = 0
for path in get_stats(path_dir):
    counter += 1
    if counter > 6000:
        break

    print("\rLOADING", counter, path, end="")
    stats_step_X = np.genfromtxt(
        path, delimiter=",", skip_header=1, usecols=[8, 11]
    )  # trading_price_1,2だけ
    stats_step_y = np.genfromtxt(
        path, delimiter=",", skip_header=1, usecols=[14]
    )  # trading_price_2を予測
    # print(stats_step_y)
    if stats_X is None:
        stats_X = stats_step_X
        stats_y = stats_step_y
    else:
        stats_X = np.append(stats_X, stats_step_X, axis=0)
        stats_y = np.append(stats_y, stats_step_y, axis=0)


# stats_X = stats_X[:, np.newaxis]  # 入力が一次元のときはこれが必要

n_train = 500
# Split the data into training/testing sets
stats_X_train = stats_X[:-n_train]
stats_X_test = stats_X[-n_train:]

# Split the targets into training/testing sets
stats_y_train = stats_y[:-n_train]
stats_y_test = stats_y[-n_train:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(stats_X_train, stats_y_train)

# Make predictions using the testing set
stats_y_pred = regr.predict(stats_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(stats_y_test, stats_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(stats_y_test, stats_y_pred))


# # Plot outputs
# plt.scatter(stats_X_test, stats_y_test,  color='black')
# plt.plot(stats_X_test, stats_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()

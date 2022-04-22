from math import sqrt

import pandas as pd


def estimate_coef(x, y):
    n = len(x)

    SS_xy = sum(y * x.T) - y.sum() * x.sum() / n
    SS_xx = sum(x * x.T) - x.sum() * x.sum() / n
    SS_yy = sum(y * y.T) - y.sum() * y.sum() / n

    b_1 = SS_xy / SS_xx
    b_0 = y.mean() - b_1 * x.mean()

    r = SS_xy / (sqrt(SS_xx * SS_yy))

    return (b_0, b_1, r)


if __name__ == "__main__":
    PATH_FILE = "../dataset/simple_regression.csv"
    df = pd.read_csv(PATH_FILE)

    x = df.income.values
    y = df.happiness.values

    b_0, b_1, residual = estimate_coef(x, y)

    y_hat = b_1 * x + b_0

    from matplotlib import pyplot as plt

    plt.scatter(x, y, c="k", label="Original")
    plt.scatter(x, y_hat, c="g", label="Predicted")
    plt.title(f"y = {round(b_1,3)}x + {round(b_0,3)}")
    plt.legend()
    plt.show()

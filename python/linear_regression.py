import pandas as pd
from numpy.linalg import inv
from numpy import dot, append
from utils import r2_score


def estimate_coef(X, y):
    n_rows, _ = X.shape
    X = append(X, [[1]] * n_rows, axis=1)
    theta = dot(inv(dot(X.T, X)), dot(X.T, y))

    return theta


if __name__ == "__main__":
    df = pd.read_csv("../dataset/linear_regression.csv")
    y = df.z.values
    X = df[df.columns.drop("z")].values

    theta = estimate_coef(X, y)
    X = append(X, [[1]] * X.shape[0], axis=1)
    y_hat = dot(X, theta)

    r2 = r2_score(y, y_hat)

    from matplotlib import pyplot as plt

    plt.title(f"Linear Regression r2 = {r2}")
    plt.plot(y, c="k", label="Original")
    plt.plot(y_hat, c="g", label="Predicted")
    plt.legend()
    plt.show()

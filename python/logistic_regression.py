import pandas as pd
from numpy import dot, exp, log, mean, sum, zeros

from utils import split_dataset


def sigmoid(z):
    return 1 / (1 + exp(-z))


def estimate_coef(X, y, iterations, learning_rate):
    n_rows, n_cols = X.shape
    w = zeros((n_cols, 1))
    b = 0

    loss = []

    for i in range(iterations):
        z = dot(X, w) + b
        pred = sigmoid(z)

        cost = (-1 / n_rows) * sum(dot(y.T, log(pred)) + dot(1 - y.T, log(1 - pred)))
        loss.append(cost)

        dw = (1 / n_rows) * dot(X.T, (pred - y))
        db = (1 / n_rows) * sum(pred - y)

        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b, loss


if __name__ == "__main__":
    df = pd.read_csv("../dataset/logistic_regression.csv")

    y = df.y.values
    y = y.reshape(len(y), 1)
    X = df[df.columns.drop("y")].values

    X_train, y_train, X_test, y_test = split_dataset(X, y, 0.8)

    w, b, loss = estimate_coef(X_train, y_train, 1000, 0.01)

    y_hat = dot(X_test, w) + b

    print(f"Accuracy: {mean(y_test == (y_hat >= 0.5)*1)}")

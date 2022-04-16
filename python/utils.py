from numpy import mean, array
from numpy.random import permutation


def randomize_dataset(X, y):
    n_rows, _ = X.shape
    idx = list(range(0, n_rows))
    idx = permutation(idx)
    X = X[idx, :]
    y = y[idx, :]

    return X, y


def split_dataset(x, y, percent):

    indexs = permutation(len(y))
    index_train = indexs[: int(len(indexs) * percent)]
    index_test = indexs[int(len(indexs) * percent) :]

    return x[index_train], y[index_train], x[index_test], y[index_test]


def mse(y, y_hat):
    loss = mean((y_hat - y) ** 2)
    return loss


def r2_score(y, y_hat):
    return 1 - (
        sum((array(y_hat) - array(y)) ** 2) / sum((array(y) - mean(array(y))) ** 2)
    )

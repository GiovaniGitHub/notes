from numpy import dot, sum, abs, array, random
from utils import mse


def update_weights_mse(x, y, y_hat):
    n_rows, _ = x.shape

    dw = (-2/(n_rows))*dot(x.T, (y - y_hat))
    db = (-2/(n_rows))*sum((y - y_hat))

    return dw, db


def update_weights_mae(x, y, y_hat):

    dif = dot(x.T, (y - y_hat))
    dw = (-1/sum(abs(y - y_hat))*dif)
    db = (-1/sum(abs(y - y_hat)))*sum((y - y_hat))

    return dw, db


def update_weights_huber(x, y, y_hat, delta=1):
    n_rows, _ = x.shape
    dif = dot(x.T, (y_hat - y))

    if sum(abs(y - y_hat)) <= delta:
        dw = (1/n_rows)*dif
        db = (1/n_rows)*sum((y_hat - y))
    else:
        dw = delta*(1/sum(abs(dif))*dif)
        db = delta*(1/sum(abs(dif)))*sum((y_hat - y))

    return dw, array([db])


def adjust_weights_with_batch(X, y, w, b, epochs, batch, losses, lr,
                              func_adjust, is_stochastic=False):
    n_rows, _ = X.shape
    for _ in range(epochs):
        for i in range((n_rows-1)//batch + 1):

            start_i = i*batch
            end_i = start_i + batch
            idx = list(range(start_i, end_i))

            if is_stochastic:
                idx = random.permutation(idx)
            Xb = X[idx, :]
            yb = y[idx, :]
            y_hat = dot(Xb, w) + b

            dw, db = func_adjust(Xb, yb, y_hat)

            w -= lr*dw
            b -= lr*db
        error = mse(y, dot(X, w))
        losses.append(error)

    return w, b, losses


def adjust_weights(X, y, w, b, epochs, losses, lr, func_adjust,
                   is_stochastic=False):
    if is_stochastic:
        n_rows, _ = X.shape
        idx = list(range(0, n_rows))
        idx = random.permutation(idx)
        X = X[idx, :]
        y = y[idx, :]

    for _ in range(epochs):
        y_hat = dot(X, w) + b
        dw, db = func_adjust(X, y, y_hat)
        w -= lr*dw
        b -= lr*db

        error = mse(y, dot(X, w) + b)
        losses.append(error)

    return w, b, losses

from numpy import dot, sum, abs, array
from utils import mse


def gradient_mse(x, y, y_hat):
    n_rows,_ = x.shape
    
    dw = (1/2*n_rows)*dot(x.T, (y_hat - y))
    db = (1/2*n_rows)*sum((y_hat - y))

    return dw, db


def gradient_mae(x, y, y_hat):
    
    dif = dot(x.T, (y_hat - y))
    dw = (1/sum(abs(dif))*dif)
    db = (1/sum(abs(dif)))*sum((y_hat - y))

    return dw, db


def gradient_huber(x, y, y_hat, delta = 1):
    n_rows,_ = x.shape
    dif = dot(x.T, (y_hat - y))

    if sum(abs(y - y_hat)) <= delta:
        dw = (1/n_rows)*dot(x.T, (y_hat - y))
        db = (1/n_rows)*sum((y_hat - y))
    else:
        dw = delta*(1/sum(abs(dif))*dif)
        db = delta*(1/sum(abs(dif)))*sum((y_hat - y))

    return dw, array([db])
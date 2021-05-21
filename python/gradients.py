from numpy import zeros, dot
from .utils import mse

def gradient_descendent(x, y, y_hat):
    n_rows,_ = x.shape

    dw = (1/n_rows)*dot(x.T, (y_hat - y))
    db = (1/n_rows)*sum((y_hat - y))

    return dw, db

def gradient_descendent_mse(x, y, y_hat):
    n_rows,_ = x.shape
    
    dw = (2/n_rows)*dot(x.T, (y_hat - y))
    db = (2/n_rows)*sum((y_hat - y))

    return dw, db
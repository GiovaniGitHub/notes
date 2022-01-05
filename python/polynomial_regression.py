from gradients import (adjust_weights, adjust_weights_with_batch,
    update_weights_mae, update_weights_huber, update_weights_mse)
import pandas as pd
from numpy import dot, zeros, array, sqrt, sign, random, mean, abs
from numpy.linalg import solve
from utils import randomize_dataset
from rbf_regression import RBFRegression

def expand_matrix(x, max_coef, min_coef=0):
    result = []
    for v in x:
        result.append([v**i for i in range(min_coef, max_coef + 1)])

    return array(result)


def ols(X, y, degrees):
    X = expand_matrix(X, min_coef=0, max_coef=degrees)

    w = solve(dot(X.T, X), dot(X.T, y))

    return w


def get_coef_with_gradient(x, y, degrees, epochs, lr, func_adjust, batch=None,
                           is_stochastic=False):
    X = expand_matrix(x, degrees, 1)
    _, n_cols = X.shape
    w = zeros((n_cols, 1))
    losses = []
    b = 0
    if is_stochastic:
        X, y = randomize_dataset(X, y)
        
    if batch:
        w, b, losses = adjust_weights_with_batch(
            X, y, w, b, epochs, batch, losses, lr, func_adjust)
    else:
        w, b, losses = adjust_weights(
            X, y, w, b, epochs, losses, lr, func_adjust)

    return w, b, losses


def get_coef_with_elastic_net(x, y, degree, tol=1e-5, max_iterators = 1e6, learning_rate = 1e-4, ridge_coef = 0.6, lasso_coef = 0.2):
    X = expand_matrix(x, degree, 0)
    _, n_cols = X.shape
    w = random.randn(n_cols).reshape((n_cols,1)) / sqrt(n_cols)
    skip = False
    losses = []
    count_iter = 0
    while not skip:
        count_iter+=1
        y_hat = dot(X, w)
        dif = dot(X.T, (y_hat - y))
        
        dw = learning_rate *(dif + ridge_coef*sign(w) + lasso_coef*2*w)
        w = w - dw
        mse = ((y_hat - y).T.dot(y_hat - y)/n_cols).flatten()
        
        losses.append(mse[0])
        if mean(abs(dw)) <= tol:
            skip = True
        if count_iter==max_iterators:
            skip = True

    return w
    
if __name__ == "__main__":
    PATH_FILE = "/home/nobrega/Dados/Documentos/Estudos/notes/dataset/polynomial_regression_data.csv"
    df = pd.read_csv(PATH_FILE)

    X = df.x.values
    y = df.y.values
    y = y.reshape(len(y), 1)
    degree = 7
    weights = ols(X, y, degree)
    weights_gd, linear_coef_gd, losses = get_coef_with_gradient(
        X, y, degree, 2000, lr=0.01, func_adjust=update_weights_mse)

    weights_gd_batch, linear_coef_bd_batch, losses = get_coef_with_gradient(
        X, y, degree, 20, lr=0.01, batch=10, func_adjust=update_weights_mae, is_stochastic=True)

    rbf = RBFRegression(20, beta=4)
    rbf.fit(expand_matrix(X, degree, 0), y)
    weights_elastic_net = get_coef_with_elastic_net(X,y, degree)
    y_hat = dot(expand_matrix(X, degree, 0), weights)
    y_hat_gradient = dot(expand_matrix(X, degree, 1), weights_gd) + linear_coef_gd
    y_hat_gradient_batch = dot(expand_matrix(X, degree, 1), weights_gd_batch) + linear_coef_bd_batch
    y_hat_elastic_net = dot(expand_matrix(X, degree, 0), weights_elastic_net)
    y_hat_rbf = rbf.predict(expand_matrix(X, degree, 0))
    from matplotlib import pyplot as plt

    plt.plot(y, c='c', label='Original')
    plt.plot(y_hat, c='g', label='Predicted')
    plt.plot(y_hat_gradient, c='r', label='Predicted with Gradient')
    plt.plot(y_hat_gradient_batch, c='b',
             label='Predicted with Gradient and Batch')
    plt.plot(y_hat_elastic_net, c='k',
             label='Predicted with Elastic Net')
    plt.plot(y_hat_rbf, c='y',
             label='Predicted with RBF')
    plt.legend()
    plt.show()

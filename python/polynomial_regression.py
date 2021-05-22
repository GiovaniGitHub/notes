from utils import mse
from gradients import adjust_weights, adjust_weights_with_batch, update_weights_mse, update_weights_mae, \
update_weights_huber
import pandas as pd 
from numpy import dot, zeros, array, random
from numpy.linalg import inv


def expand_matrix(x, max_coef, min_coef = 0):
    result = []
    for v in x:
        result.append([v**i for i in range(min_coef, max_coef)])
    
    return array(result)


def estimate_coef(X, y, degrees):
    X = expand_matrix(X, min_coef=0 , max_coef=degrees)

    beta = dot(inv(dot(X.T, X)), dot(X.T, y))

    return beta


def get_coef_with_gradient(x,y, degrees, epochs, lr, func_adjust):
    X = expand_matrix(x, degrees, 1)
    _, n_cols = X.shape
    w = zeros((n_cols,1))
    losses = []
    b = 0
    
    w, b, losses = adjust_weights(X, y, w, b, epochs, losses, lr, func_adjust=func_adjust)

    return w, b, losses


def get_coef_with_grandient_and_batch(x, y, batch, degrees, epochs, lr, func_adjust, is_stochastic=False):
    X = expand_matrix(x, degrees, 1)
    _, n_cols = X.shape
    w = zeros((n_cols,1))
    losses = []
    b = 0

    w, b, losses = adjust_weights_with_batch(X, y, w, b, epochs, batch, losses, lr, func_adjust, is_stochastic)
    
    return w, b, losses


if __name__ == "__main__":
    PATH_FILE = "../dataset/polynomial_regression_data.csv"
    df = pd.read_csv(PATH_FILE)

    
    X = df.x.values
    y = df.y.values
    y = y.reshape(len(y),1)

    weights = estimate_coef(X, y, degrees=13)
    weights_gd, linear_coef_gd, losses = get_coef_with_gradient(X, y, 13, 10000, lr=0.01, 
                                                                func_adjust=update_weights_mae)
    weights_gd_batch, linear_coef_bd_batch, losses = get_coef_with_grandient_and_batch(X, y, 10, 13, 1000, lr=0.01,
                                                                                       func_adjust=update_weights_mae)
    
    y_hat = dot(expand_matrix(X, 13, 0), weights)
    y_hat_gradient = dot(expand_matrix(X, 13, 1), weights_gd) + linear_coef_gd
    y_hat_gradient_batch = dot(expand_matrix(X, 13, 1), weights_gd_batch) + linear_coef_bd_batch
    
    
    from matplotlib import pyplot as plt
    
    
    plt.plot(y, c='c', label='Original')
    plt.plot(y_hat, c='g', label= 'Predicted')
    plt.plot(y_hat_gradient, c='r', label= 'Predicted with Gradient')
    plt.plot(y_hat_gradient_batch, c='b', label= 'Predicted with Gradient and Batch')
    plt.legend()
    plt.show()
    
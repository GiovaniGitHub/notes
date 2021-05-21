from gradients import gradient_descendent_mse
import pandas as pd 
from numpy import dot, sum, mean, zeros, array, insert, ones
from numpy.linalg import inv


def expand_matrix(x, max_coef, min_coef = 0):
    result = []
    for v in x:
        result.append([v**i for i in range(min_coef, max_coef+1)])
    
    return array(result)


def mse(y, y_hat):
    loss = mean((y_hat - y)**2)
    return loss


def gradient_descendent(x, y, y_hat):
    n_rows,_ = x.shape
    dw = (1/n_rows)*dot(x.T, (y_hat - y))
    db = (1/n_rows)*sum(y_hat - y)

    return dw, db


def estimate_coef(X, y, degrees):
    X = expand_matrix(X, min_coef=0 , max_coef=degrees)

    beta = dot(inv(dot(X.T, X)), dot(X.T, y))

    return beta


def estimate_coef_with_gradient(x,y, degrees, epochs, lr):
    X = expand_matrix(x, degrees, 1)
    
    _, n_cols = X.shape
    
    w = zeros((n_cols,1))
    
    losses = []
    b = 0
    for _ in range(epochs):
        y_hat = dot(X, w) + b
        dw, db = gradient_descendent_mse(X, y, y_hat)
        w -= lr*dw
        b -= lr*db

        error = mse(y, dot(X, w) + b)
        losses.append(error)

    return w, b, losses


def estimate_coef_with_grandient_and_batch(x, y, bs, degrees, epochs, lr):
    X = expand_matrix(x, degrees, 1)
    
    n_rows, n_cols = X.shape
    
    w = zeros((n_cols,1))
    
    losses = []
    b = 0
    for _ in range(epochs):
        for i in range((n_rows-1)//bs + 1):
            
            start_i = i*bs
            end_i = start_i + bs
            Xb = X[start_i:end_i,:]
            yb = y[start_i:end_i,:]
            y_hat = dot(Xb, w)
            
            dw,db = gradient_descendent_mse(Xb, yb, y_hat)
            
            w -= lr*dw
            b -= lr*db
        error = mse(y, dot(X, w))
        losses.append(error)
        
    return w, db,losses


if __name__ == "__main__":
    PATH_FILE = "../dataset/polynomial_regression_data.csv"
    df = pd.read_csv(PATH_FILE)

    
    X = df.x.values
    y = df.y.values
    y = y.reshape(len(y),1)

    weights = estimate_coef(X, y, degrees=13)
    weights_gradients, linear_coef_gradients, losses = estimate_coef_with_gradient(X, y, 13, 10000, lr=0.01)
    weights_gradients_batch, linear_coef_gradients_batch, losses = estimate_coef_with_grandient_and_batch(X, y, 50, 13, 10000, lr=0.01)
    
    y_hat = dot(expand_matrix(X, 13, 0), weights)
    y_hat_gradient = dot(expand_matrix(X, 13, 1), weights_gradients) + linear_coef_gradients
    y_hat_gradient_batch = dot(expand_matrix(X, 13, 1), weights_gradients_batch) + linear_coef_gradients_batch
    
    
    from matplotlib import pyplot as plt
    
    
    plt.plot(y, c='c', label='Original')
    plt.plot(y_hat, c='g', label= 'Predicted')
    plt.plot(y_hat_gradient, c='r', label= 'Predicted with Gradient')
    plt.plot(y_hat_gradient_batch, c='b', label= 'Predicted with Gradient and Batch')
    plt.legend()
    plt.show()
    
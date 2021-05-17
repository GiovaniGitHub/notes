import pandas as pd 
from numpy import dot, sum, mean, zeros, array
from numpy.linalg import inv

def r2_score(y, y_hat):
    return 1 - (sum((array(y_hat)-array(y))**2)/
                sum((array(y)-mean(array(y)))**2))
    

def expand_matrix(x, degrees):
    result = []
    for v in x:
        result.append([v**i for i in range(degrees+1)])
    
    return array(result)


def mse(y, y_hat):
    loss = mean((y_hat - y)**2)
    return loss


def gradient_descendent(x, y, y_hat):
    n_rows,_ = x.shape
    dw = (1/n_rows)*dot(x.T, (y_hat - y))
    
    return dw

def estimate_coef(X, y, degrees):
    X = expand_matrix(X, degrees)
    beta = dot(inv(dot(X.T, X)), dot(X.T, y))

    return beta

def estimate_coef_with_gradient(x,y, degrees, epochs, lr):
    X = expand_matrix(x, degrees)
    
    _, n_cols = X.shape
    
    w = zeros((n_cols,1))
    
    losses = []
    for _ in range(epochs):
        y_hat = dot(X, w)
        dw = gradient_descendent(X, y, y_hat)
        w -= lr*dw

        error = mse(y, dot(X, w))
        losses.append(error)

    return w, losses


def estimate_coef_with_grandient_and_batch(x, y, bs, degrees, epochs, lr):
    X = expand_matrix(x, degrees)
    
    n_rows, n_cols = X.shape
    
    w = zeros((n_cols,1))
    
    losses = []
    
    for _ in range(epochs):
        for i in range((n_rows-1)//bs + 1):
            
            start_i = i*bs
            end_i = start_i + bs
            Xb = X[start_i:end_i]
            yb = y[start_i:end_i,:]
            y_hat = dot(Xb, w)
            
            dw = gradient_descendent(Xb, yb, y_hat)
            
            w -= lr*dw
        
        error = mse(y, dot(X, w))
        losses.append(error)
        
    return w, losses


if __name__ == "__main__":
    PATH_FILE = "../dataset/polynomial_regression_data.csv"
    df = pd.read_csv(PATH_FILE)

    
    X = df.x.values
    y = df.y.values
    y = y.reshape(len(y),1)

    weights = estimate_coef(X, y, degrees=13)
    weights_gradients, losses = estimate_coef_with_gradient(X, y, 13, 10000, lr=0.01)
    weights_gradients_batch, losses = estimate_coef_with_grandient_and_batch(X, y, 50, 13, 10000, lr=0.01)
    
    X_expanded = expand_matrix(X, 13)
    y_hat = dot(X_expanded, weights)
    y_hat_gradient = dot(X_expanded, weights_gradients)
    y_hat_gradient_batch = dot(X_expanded, weights_gradients_batch)
    
    
    from matplotlib import pyplot as plt
    
    
    plt.plot(y, c='c', label='Original')
    plt.plot(y_hat, c='g', label= 'Predicted')
    plt.plot(y_hat_gradient, c='r', label= 'Predicted with Gradient')
    plt.plot(y_hat_gradient_batch, c='b', label= 'Predicted with Gradient and Batch')
    plt.legend()
    plt.show()
    
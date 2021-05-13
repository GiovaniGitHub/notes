import pandas as pd 
from numpy.linalg import inv
from numpy import dot, append, mean

def mse(y, y_hat):
    loss = mean((y_hat - y)**2)
    return loss

def estimate_coef(X,y):
    n_rows, _ = X.shape
    X = append(X, [[1]]*n_rows, axis=1)
    beta = dot(inv(dot(X.T, X)), dot(X.T, y))

    return beta[:-1], beta[-1]
    

if __name__ == "__main__":
    df = pd.read_csv('../dataset/linear_regression.txt')
    y = df.MEDV.values
    X = df[df.columns.drop('MEDV')].values
    
    b_0, b_1 = estimate_coef(X, y)
    y_hat = dot(X, b_0) + b_1
    
    loss = mse(y, y_hat)
    
    from matplotlib import pyplot as plt
    
    plt.plot(y, c='k', label='Original')
    plt.plot(y_hat, c='g', label= 'Predicted')
    plt.legend()
    plt.show()
    
    
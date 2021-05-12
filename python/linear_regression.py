import pandas as pd 
from numpy.linalg import inv
from numpy import dot, append


def estimate_coef(X,y):
    n_rows, _ = X.shape
    X = append(X, [[1]]*n_rows, axis=1)
    beta = dot(inv(dot(X.T, X)), dot(X.T, y))

    return beta[:-1], beta[-1]
    

if __name__ == "__main__":
    df = pd.read_csv('../dataset/simple_regression.csv')
    y = df.MEDV.values
    X = df[df.columns.drop('MEDV')].values
    
    b_0, b_1 = estimate_coef(X, y)
    
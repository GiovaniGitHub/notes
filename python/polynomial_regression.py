import pandas as pd 
from numpy import dot, sum, mean, zeros, array


def r2_score(y, y_hat):
    return 1 - (sum((array(y_hat)-array(y))**2)/
                sum((array(y)-mean(array(y)))**2))
    
    
def predict(X, w, b, degrees):
    x1 = expand_matrix(X, degrees)
    return dot(x1, w) + b


def expand_matrix(x, degrees):
    result = []
    for v in x:
        result.append([v**i for i in range(degrees+1)])
    
    return array(result)


def mse(y, y_hat):
    loss = mean((y_hat - y)**2)
    return loss


def gradients(x, y, y_hat):
    n_rows,_ = x.shape
    dw = (1/n_rows)*dot(x.T, (y_hat - y))
    db = (1/n_rows)*sum((y_hat - y)) 
    
    return dw, db

def train(x, y, bs, degrees, epochs, lr):
    X = expand_matrix(x, degrees)
    
    n_rows, n_cols = X.shape
    
    w = zeros((1,n_cols))
    b = 0
    
    losses = []
    
    for epoch in range(epochs):
        for i in range((n_rows-1)//bs + 1):
            
            start_i = i*bs
            end_i = start_i + bs
            Xb = X[start_i:end_i]
            yb = y[start_i:end_i,:]
            # (1,n) (n, 1)
            y_hat = dot(Xb, w.T) + b
            
            dw, db = gradients(Xb, yb, y_hat)
            
            w -= lr*dw.T
            b -= lr*db
        
        l = mse(y, dot(X, w.T) + b)
        losses.append(l)
        
    return w, b, losses


if __name__ == "__main__":
    PATH_FILE = "dataset/simple_regression.csv"
    df = pd.read_csv(PATH_FILE)
    n_rows, n_cols = df.shape

    x = df.income.values
    y = df.happiness.values
    
    y = y.reshape((len(y),1))

    w, b, l = train(x, y, bs=10, degrees=4, epochs=2, lr=0.01)
    print(w)
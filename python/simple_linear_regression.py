import pandas as pd 
from math import sqrt

def estimate_coef(x, y):
    n = len(x)

    m_x = x.mean()
    m_y = y.mean()

    SS_xy = sum(y*x.T) - n*m_y*m_x
    SS_xx = sum(x*x.T) - n*m_x*m_x
    SS_yy = sum(y*y.T) - n*m_y*m_y

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    r = SS_xy/(sqrt(SS_xx*SS_yy))

    return (b_0, b_1, r) 


if __name__ == "__main__":
    PATH_FILE = "../dataset/simple_regression.csv"
    df = pd.read_csv(PATH_FILE)

    x = df.income.values
    y = df.happiness.values

    b_0, b_1, residual = estimate_coef(x,y)

    y_hat = b_1 * x + b_0
    
    from matplotlib import pyplot as plt
    
    plt.scatter(x, y, c='k', label='Original')
    plt.scatter(x, y_hat, c='g', label= 'Predicted')
    plt.legend()
    plt.show()
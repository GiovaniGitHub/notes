import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return np.sqrt(x) * np.sin(np.sqrt(x))


# generate points used to plot
x_plot = np.linspace(0, 100, 1000)

# generate points and keep a subset of them
x = np.linspace(0, 100, 1000)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x)
y = f(x)

plt.scatter(x,y)
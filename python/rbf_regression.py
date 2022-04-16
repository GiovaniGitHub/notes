import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import norm


class RBFRegression:
    def __init__(self, num_center, beta=8):
        self.num_center = num_center
        self.centers = None
        self.beta = beta
        self.w = None

    def radial_basis_function(self, vector):
        return np.exp(-self.beta * norm(vector) ** 2)

    def calculate_gradient(self, X):
        resp = []
        for c in self.centers:
            resp.append([self.radial_basis_function(c - row) for row in X])

        return np.transpose(resp)

    def fit(self, X, y):
        idx = np.random.permutation(X.shape[0])[: self.num_center]
        self.centers = X[idx, :]
        gradient = self.calculate_gradient(X)

        self.w = np.linalg.solve(np.dot(gradient.T, gradient), np.dot(gradient.T, y))

    def predict(self, X):
        gradient = self.calculate_gradient(X)
        return np.dot(gradient, self.w)

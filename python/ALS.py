from numpy.linalg import norm
from numpy import dot
from numpy.random import RandomState
from scipy import sparse
from scipy.sparse.linalg import spsolve


def cossine_distance(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def implicit_weighted_ALS(
    training_set, lambda_val=0.1, alpha=40, iterations=10, rank_size=20, seed=0
):
    conf = alpha * training_set
    num_user = conf.shape[0]
    num_item = conf.shape[1]
    rstate = RandomState(seed)

    X = sparse.csr_matrix(rstate.normal(size=(num_user, rank_size)))
    Y = sparse.csr_matrix(rstate.normal(size=(num_item, rank_size)))

    X_eye = sparse.eye(num_user)
    Y_eye = sparse.eye(num_item)
    lambda_eye = lambda_val * sparse.eye(rank_size)

    for _ in range(iterations):
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        for u in range(num_user):
            conf_samp = conf[u, :].toarray()
            pref = conf_samp.copy()
            pref[pref != 0] = 1
            CuI = sparse.diags(conf_samp, [0])
            yTCuIY = Y.T.dot(CuI).dot(Y)
            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T)

            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu)

        for i in range(num_item):
            conf_samp = conf[:, i].T.toarray()
            pref = conf_samp.copy()
            pref[pref != 0] = 1
            CiI = sparse.diags(conf_samp, [0])
            xTCiIX = X.T.dot(CiI).dot(X)
            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T)
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)

    return X, Y.T

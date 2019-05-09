import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit



def computemAP(q):
    assert type(q) == np.ndarray
    assert q.ndim == 2
    invvalues = np.divide(np.ones(q.shape[1]), np.ones(q.shape[1]).cumsum())

    map_ = 0
    prec_sum = q.cumsum(1)
    for i in range(prec_sum.shape[0]):
        idx_nonzero = np.nonzero(q[i])[0]
        if len(idx_nonzero) > 0:
            map_ += np.mean(prec_sum[i][idx_nonzero] * invvalues[idx_nonzero])

    return map_ / q.shape[0]


def getLaplaceIndicator(X, y_bin, idx_tr, model, params):
    X_tr = X[idx_tr]

    auxmodel = LogisticRegression(**params)
    auxmodel.fit(X_tr, 0.5 * (1 + y_bin[idx_tr]))
    auxgradients = -expit(-y_bin * np.dot(X, auxmodel.coef_.T))*y_bin*X

    hess_left = expit(np.dot(X_tr, auxmodel.coef_.T)) * X_tr
    hess_right = expit(-np.dot(X_tr, auxmodel.coef_.T)) * X_tr

    hess = (params['C'] * np.dot(hess_left.T, hess_right) / hess_right.shape[0]) + np.eye(hess_right.shape[1])
    hess_inverse = np.linalg.inv(hess)

    # \theta_1^* - \theta_0^*
    diff_theta = - np.dot(auxgradients, hess_inverse) / hess_right.shape[0]
    indic1 = - 2 * np.dot(auxgradients, model.coef_[0] - auxmodel.coef_[0])
    indic2 = np.sum(auxgradients * diff_theta, axis=1)
    # sigmoids = expit(np.dot(X, auxmodel.coef_.T)) * expit(- np.dot(X, auxmodel.coef_.T))
    # indic_3 = np.sum(np.dot(X, hess_inverse) * X, axis=1) * sigmoids
    # indic_3 = np.log(1 + indic_3)

    indic = expit(indic1) #+ indic2

    return indic#indic1, indic2

def getMultiLaplace(X, y_bin, idx_tr, n_tr, p, model, params):
    indic = np.zeros((X.shape[0]))
    for _ in range(p):
        indic += getLaplaceIndicator(X, y_bin, np.random.choice(idx_tr, n_tr, replace=False), model, params)

    return indic

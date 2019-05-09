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


def computeMATT(X, y_bin, idx_public, model, params):
    X_public = X[idx_public]

    # Training auxiliary regression model on public data
    auxmodel = LogisticRegression(**params)
    auxmodel.fit(X_public, 0.5 * (1 + y_bin[idx_public]).ravel())

    # Compute gradient of all samples according to this new model
    auxgradients = -expit(-y_bin * np.dot(X, auxmodel.coef_.T))*y_bin*X

    # Compute (and inverse) hessian of auxiliary model
    hess_left = expit(np.dot(X_public, auxmodel.coef_.T)) * X_public
    hess_right = expit(-np.dot(X_public, auxmodel.coef_.T)) * X_public
    hess = (params['C'] * np.dot(hess_left.T, hess_right) / hess_right.shape[0]) + np.eye(hess_right.shape[1])
    hess_inverse = np.linalg.inv(hess)

    # Compute MATT
    matt = - np.dot(auxgradients, model.coef_[0] - auxmodel.coef_[0])

    return matt

def computeMetrics(indicator_train, indicator_test):
    order = np.argsort(np.concatenate([indicator_train, indicator_test]))
    gt = np.concatenate([np.ones_like(indicator_train), np.zeros_like(indicator_test)])
    s = np.cumsum(gt[order])
    
    map_train = float(100*computemAP(gt[order][None,::-1]))
    map_test = float(100*computemAP(1 - gt[order][None,:]))
    acc = float(100*np.max((np.arange(gt.shape[0]) - 2 * s + s[-1]) / gt.shape[0]))

    return map_train, map_test, acc


#def getMultiLaplace(X, y_bin, idx_public, n_tr, p, model, params):
#    matt = np.zeros((X.shape[0]))
#    for _ in range(p):
#        matt += computeMATT(X, y_bin, np.random.choice(idx_public, n_tr, replace=False), model, params)
#
#    return matt

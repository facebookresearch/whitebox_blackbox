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
    auxgradients = -expit(-y_bin * np.dot(X, auxmodel.coef_.T)) * y_bin * X

    # Check gradient formula
    # auxgrad2 = approx_grad(auxmodel, X, y_bin)
    # assert np.allclose(auxgradients, auxgrad2)

    # Compute MATT
    matt = - np.dot(auxgradients, model.coef_[0] - auxmodel.coef_[0])

    return matt


def computeMetrics(indicator_train, indicator_test):
    """
    Compute accuracy and mean average precision given by indicators
    We assume that the indicators are supposed to have low value for test elements (e.g. - loss of the model)
    """
    assert indicator_train.shape[0] == indicator_test.shape[0]

    order = np.argsort(np.concatenate([indicator_train, indicator_test]))
    gt = np.concatenate([np.ones_like(indicator_train), np.zeros_like(indicator_test)])

    accuracies = [(np.sum(gt[order[:n0]] == 0) + np.sum(gt[order[n0:]] == 1)) for n0 in range(gt.shape[0])]
    # s = np.cumsum(gt[order])
    # s = np.insert(s, 0, 0)
    # assert np.all(accuracies == (np.arange(gt.shape[0] + 1) - 2 * s + s[-1])[:gt.shape[0]])

    map_train = float(100 * computemAP(gt[order][None,::-1]))
    map_test =  float(100 * computemAP(1 - gt[order][None,:]))
    acc =       float(100 * np.max(accuracies) / gt.shape[0])

    return map_train, map_test, acc


def approx_grad(model, X, y_bin):
    eps = 1e-3
    grads = np.zeros((X.shape[0], model.coef_.shape[1]))
    for i in range(model.coef_.shape[1]):
        model.coef_[0, i] += eps
        loss_plus  = - model.predict_log_proba(X)[np.arange(X.shape[0]), ((1 + y_bin) / 2).ravel().astype(int)]
        model.coef_[0, i] -= 2 * eps
        loss_minus = - model.predict_log_proba(X)[np.arange(X.shape[0]), ((1 + y_bin) / 2).ravel().astype(int)]
        model.coef_[0, i] += eps

        grads[:, i] += (loss_plus - loss_minus) / (2 * eps)

    return grads

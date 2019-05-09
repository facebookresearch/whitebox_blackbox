# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit



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

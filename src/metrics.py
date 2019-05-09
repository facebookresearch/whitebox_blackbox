# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np


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



def computeMetrics(indicator_train, indicator_test):
    """
    Compute accuracy and mean average precision given by indicators
    We assume that the indicators are supposed to have low value for test elements (e.g. - loss of the model)
    Ideally, sigmoid(indicator) is equal to the membership variable (1 for train elements, 0 for test elements)
    """
    assert indicator_train.shape[0] == indicator_test.shape[0]

    order = np.argsort(np.concatenate([indicator_train, indicator_test]))
    gt = np.concatenate([np.ones_like(indicator_train), np.zeros_like(indicator_test)])

    accuracies = [np.sum(gt[order[:n0]] == 0) + np.sum(gt[order[n0:]] == 1) for n0 in range(gt.shape[0])]

    map_train = float(100 * computemAP(gt[order][None,::-1]))
    map_test =  float(100 * computemAP(1 - gt[order][None,:]))
    acc =       float(100 * np.max(accuracies) / gt.shape[0])

    cutoff = np.argmax(accuracies)
    precision_train = float(np.mean(gt[order[cutoff:]]))
    recall_train = float(np.sum(gt[order[cutoff:]]) / np.sum(gt))

    return map_train, map_test, acc, precision_train, recall_train

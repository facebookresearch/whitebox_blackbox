import numpy as np
from os.path import join
from sklearn.linear_model import LogisticRegression
import argparse
from functions import computeMATT, computeMetrics
from lib.logger import create_logger
from lib.utils import boolify


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", type=float, default=1e4)
    parser.add_argument("-p", type=int, default=1)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root_data", type=str, default="/private/home/asablayrolles/code/projects/whitebox_blackbox/")
    parser.add_argument("--normed", type=boolify, default=True)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = create_logger()

    state = np.random.RandomState(args.seed)

    # Loading data
    X = np.load(join(args.root_data, "cifar10_cnn.npy"))
    label = np.load(join(args.root_data, "cifar10_label.npy")).astype(int).ravel()
    if args.normed:
        X /= np.linalg.norm(X, axis=1, keepdims=True)

    # Shuffling data
    indices = np.nonzero(label >= 8)[0]
    indices = indices[state.permutation(indices.shape[0])]
    X = X[indices]
    y = (label[indices] == 9).astype(int)

    # Add a constant term for bias (simplifies computations for Hessian)
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    params = {
        'tol': 0.0001,
        'C': args.C,
        'fit_intercept': False,
        'solver': 'liblinear',
    }

    # Create splits
    ntrain = args.ntrain
    train_set  = np.arange(0, ntrain)
    test_set   = np.arange(ntrain, 2*ntrain)
    public_set = np.arange(2*ntrain, X.shape[0])

    # Fit logistic model
    model = LogisticRegression(**params)
    model.fit(X[train_set], y[train_set])
    logger.info("rawlogs: " + str({
        "train_accuracy": model.score(X[train_set], y[train_set]),
        "val_accuracy":   model.score(X[test_set],  y[test_set])
    }))

    # MALT uses the loss of each element as indicator of train_set membership
    losses_train = - model.predict_log_proba(X[train_set])[np.arange(train_set.shape[0]), y[train_set]]
    losses_test  = - model.predict_log_proba(X[test_set])[np.arange(test_set.shape[0]), y[test_set]]
    map_train, map_test, acc = computeMetrics(- losses_train, - losses_test)
    logger.info("rawlogs: " + str({
        'map_train': map_train,
        'map_test': map_test,
        'accuracy': acc,
        'method': 'malt'
    }))

    # MATT use a Taylor expansion to compute train_set membership
    y_bin = 2 * (y - 0.5).reshape(-1, 1)
    matt = computeMATT(X, y_bin, state.choice(public_set, ntrain, replace=False), model, params)
    map_train, map_test, acc = computeMetrics(matt[train_set], matt[test_set])
    logger.info("rawlogs: " + str({
        'map_train': map_train,
        'map_test': map_test,
        'accuracy': acc,
        'method': 'matt'
    }))

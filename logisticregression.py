import numpy as np
from os.path import join
from sklearn.linear_model import LogisticRegression
import argparse
from functions import computemAP, getLaplaceIndicator, getMultiLaplace
from logger import create_logger

parser = argparse.ArgumentParser()
parser.add_argument("-C", type=float, default=1e4)
parser.add_argument("-p", type=int, default=1)
parser.add_argument("--ntrain", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

state = np.random.RandomState(args.seed)

root_data = "/private/home/asablayrolles/code/projects/dejalight"
# X = np.load(join(root_data, "cifar10_gist.npy"))
X = np.load(join(root_data, "cifar10_cnn.npy"))
X /= np.linalg.norm(X, axis=1, keepdims=True)
label = np.load(join(root_data, "cifar10_label.npy")).astype(int)[:,0]

indices = np.nonzero(label >= 8)[0]
indices = indices[state.permutation(indices.shape[0])]

X = X[indices]
X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
y = (label[indices] == 9).astype(int)

params = {
    'tol': 0.0001,
    'C': args.C,
    'fit_intercept': False
}

model = LogisticRegression(**params)
logger = create_logger()

ntrain = args.ntrain
model.fit(X[:ntrain], y[:ntrain])
print("Train accuracy: %.2f" % model.score(X[:ntrain], y[:ntrain]))
print("Val accuracy %.2f" % model.score(X[ntrain:2*ntrain], y[ntrain:2*ntrain]))

logger.info("rawlogs: " + str({
    "train_accuracy": model.score(X[:ntrain], y[:ntrain]),
    "val_accuracy": model.score(X[ntrain:2*ntrain], y[ntrain:2*ntrain])
}))

losses_train = - model.predict_log_proba(X[:ntrain])[np.arange(ntrain), y[:ntrain]]
losses_test = - model.predict_log_proba(X[ntrain:])[np.arange(X.shape[0] - ntrain), y[ntrain:]]

order = np.argsort(np.concatenate([-losses_train, -losses_test[:ntrain]]))
gt = np.concatenate([np.ones_like(losses_train), np.zeros_like(losses_test[:ntrain])])
s = np.cumsum(gt[order])

print('MAP (test first): %.2f' % float(100*computemAP(1 - gt[order][None,:])))
print('MAP (train first): %.2f' % float(100*computemAP(gt[order][None,::-1])))
print("Best accuracy: %.2f" % float(100*np.max((np.arange(gt.shape[0]) - 2 * s + s[-1]) / gt.shape[0])))

logger.info("rawlogs: " + str({
    'map_train': float(100*computemAP(gt[order][None,::-1])),
    'map_test': float(100*computemAP(1 - gt[order][None,:])),
    'accuracy': float(100*np.max((np.arange(gt.shape[0]) - 2 * s + s[-1]) / gt.shape[0])),
    'method': 'loss'
}))
print(50*"=")

y_bin = 2*(y - 0.5).reshape(-1, 1)

# indic = getLaplaceIndicator(X, y_bin, np.arange(2*ntrain, 3*ntrain), model, params)
indic = getMultiLaplace(X, y_bin, np.arange(2*ntrain, X.shape[0]), n_tr=ntrain, p=args.p, model=model, params=params)
log_indic = np.log(indic.max() + 1e-8 - indic)

order = np.argsort(np.concatenate([indic[:ntrain], indic[ntrain:2*ntrain]]))
gt = np.concatenate([np.ones_like(indic[:ntrain]), np.zeros_like(indic[ntrain:2*ntrain])])
s = np.cumsum(gt[order])

logger.info("rawlogs: " + str({
    'map_train': float(100*computemAP(gt[order][None,::-1])),
    'map_test': float(100*computemAP(1 - gt[order][None,:])),
    'accuracy': float(100*np.max((np.arange(gt.shape[0]) - 2 * s + s[-1]) / gt.shape[0])),
    'method': 'hess'
}))

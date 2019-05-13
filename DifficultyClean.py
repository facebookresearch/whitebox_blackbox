from __future__ import division
import numpy as np
import lib.models as models
import torch
from os.path import join
from lib.utils import getTransform
from cifar10.datasets.cifar import CIFAR10
import torch.nn as nn
from itertools import chain
from os.path import dirname


def getLosses(img, lbl, net):
    criterion = nn.CrossEntropyLoss(reduction='none')
    net = net.eval()
    with torch.no_grad():
        out = net(img)
        loss = criterion(out, lbl)

    return loss.numpy()


#root_data = "/private/home/asablayrolles/data/cifar-dejalight/idx_public_%d.npy"
root_data = "/private/home/asablayrolles/data/cifar-dejalight2/idx_public_%d.npy"
n_img = 60000 # 50000


indices = [np.load(root_data % i) for i in range(30)]
public_indices = set(chain.from_iterable([set(t) for t in indices]))
public_indices = np.array(list(public_indices))

seen = np.zeros((30, n_img), dtype=bool)

for i_idx, idx in enumerate(indices):
    seen[i_idx][idx] = 1


seen = seen.T[public_indices]

# _ = plt.hist(np.mean(seen, axis=1), bins=200)

ckpt_root = "/checkpoint/asablayrolles/dejalight_cifar/"
epoch = 50

nets = []
for i in range(30):
    net = models.smallnet(10)
    #state_dict = torch.load(join(ckpt_root, "train_estim_all/_name=public_%d/smallnet_epoch=%d.pth" % (i, epoch)))
    state_dict = torch.load(join(ckpt_root, "train_estim_smallbatch_final/_name=public_%d/smallnet_epoch=%d.pth" % (i, epoch)))
    net.load_state_dict(state_dict["net"])
    nets.append(net)



transform = getTransform(0)

root_data = dirname(root_data)
trainset = CIFAR10(root=root_data, name='public', download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=2)

img, lbl = next(iter(trainloader))
print(img.size())
print(lbl.size())

import time

losses = []
start = time.time()
for i_net, net in enumerate(nets):
    losses.append(getLosses(img, lbl, net))
    eta = (len(nets) - (i_net + 1)) * (time.time() - start) / (i_net + 1)
    if i_net % 10 == 0:
        print("ETA: %.2f seconds" % eta)


losses = np.vstack(losses).T
losses_heldout = losses[:, -1]  # (n_images, )
losses = losses[:, :-1]         # (n_images, n_models - 1)

seen_heldout = seen[:, -1]  # (n_images, )
seen = seen[:, :-1]         # (n_images, n_models - 1)

logloss = np.log(1e-8 + losses)

# id_element = 0#np.random.choice(35000)
# print("id element: %d" % id_element)
# loss_seen = logloss[id_element][seen[id_element]]
# loss_unseen = logloss[id_element][np.logical_not(seen[id_element])]
# plt.hist([loss_seen, loss_unseen], density=True)


# In[225]:


best_taus = []
smoothed_taus = []

print("There are %d elements" % logloss.shape[0])
for id_element in range(logloss.shape[0]):
    loss_seen = logloss[id_element][seen[id_element]]
    loss_unseen = logloss[id_element][np.logical_not(seen[id_element])]

    num_pos = np.sum(seen[id_element])
    num_neg = seen[id_element].shape[0] - num_pos

    values = np.concatenate([loss_seen, loss_unseen])
    sugar = np.concatenate([np.ones((num_pos)) / num_pos, - np.ones(num_neg) / num_neg])
    order = np.argsort(values)

    best_pos = np.argmax(np.cumsum(sugar[order]))
    best_tau = values[order[best_pos]]

    if best_pos >= 2 and best_pos < order.shape[0] - 2:
        smoothed_tau = np.mean([values[order[best_pos-2:best_pos]], values[order[best_pos+1:best_pos+3]]])
    else:
        smoothed_tau = best_tau

    best_taus.append(best_tau)
    smoothed_taus.append(smoothed_tau)

best_taus = np.array(best_taus)
smoothed_taus = np.array(smoothed_taus)

print("Smoothed tau", np.mean(np.exp(smoothed_taus)))
print("Best tau", np.mean(np.exp(best_taus)))

print("Perf", np.mean(
    np.logical_xor(
        losses_heldout - np.exp(smoothed_taus) >= 0,
        seen_heldout
    )
))

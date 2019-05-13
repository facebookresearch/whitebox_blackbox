#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import division
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[133]:


import numpy as np
import lib.models as models
import torch
from os.path import join
from lib.utils import getTransform
from cifar10.datasets.cifar import CIFAR10
import torch.nn as nn


def getLosses(img, lbl, net):
    criterion = nn.CrossEntropyLoss(reduction='none')
    net = net.eval()
    with torch.no_grad():
        out = net(img)
        loss = criterion(out, lbl)
        
    return loss.numpy()


# In[216]:


from itertools import chain

#root_data = "/private/home/asablayrolles/data/cifar-dejalight/idx_public_%d.npy"
root_data = "/private/home/asablayrolles/data/cifar-dejalight2/idx_public_%d.npy"
n_img = 60000 # 50000


indices = [np.load(root_data % i) for i in range(30)]
public_indices = set(chain.from_iterable([set(t) for t in indices]))
public_indices = np.array(list(public_indices))


# In[217]:


seen = np.zeros((30, n_img), dtype=bool)


# In[218]:


for i_idx, idx in enumerate(indices):
    seen[i_idx][idx] = 1


# In[219]:


seen = seen.T[public_indices]


# In[220]:


_ = plt.hist(np.mean(seen, axis=1), bins=200)


# In[185]:


ckpt_root = "/checkpoint/asablayrolles/dejalight_cifar/"
epoch = 50

nets = []
for i in range(30):
    net = models.smallnet(10)
    #state_dict = torch.load(join(ckpt_root, "train_estim_all/_name=public_%d/smallnet_epoch=%d.pth" % (i, epoch)))
    state_dict = torch.load(join(ckpt_root, "train_estim_smallbatch_final/_name=public_%d/smallnet_epoch=%d.pth" % (i, epoch)))
    net.load_state_dict(state_dict["net"])
    nets.append(net)


# In[199]:


from os.path import dirname

transform = getTransform(0)

root_data = dirname(root_data)
trainset = CIFAR10(root=root_data, name='public', download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=2)

img, lbl = next(iter(trainloader))
print(img.size())
print(lbl.size())


# In[200]:


import time

losses = []
start = time.time()
for i_net, net in enumerate(nets):
    losses.append(getLosses(img, lbl, net))
    eta = (len(nets) - (i_net + 1)) * (time.time() - start) / (i_net + 1)
    if i_net % 10 == 0:
        print("ETA: %.2f seconds" % eta)


# In[201]:


losses = np.vstack(losses).T


# In[202]:


losses_heldout = losses[:, -1]  # (n_images, )
losses = losses[:, :-1]         # (n_images, n_models - 1)


# In[221]:


seen_heldout = seen[:, -1]  # (n_images, )
seen = seen[:, :-1]         # (n_images, n_models - 1)


# In[222]:


logloss = np.log(1e-8 + losses)


# In[223]:


id_element = 0#np.random.choice(35000)
print("id element: %d" % id_element)
loss_seen = logloss[id_element][seen[id_element]]
loss_unseen = logloss[id_element][np.logical_not(seen[id_element])]
plt.hist([loss_seen, loss_unseen], density=True)


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


# In[226]:


plt.hist([loss_seen, loss_unseen])
print(best_tau)
print(loss_seen)


# In[149]:


_ = plt.hist(smoothed_taus, bins=200)


# In[150]:


_ = plt.hist(best_taus, bins=200)


# In[151]:


np.mean(np.exp(best_taus))


# In[152]:


num_pos = np.sum(seen[id_element])
num_neg = seen[id_element].shape[0] - num_pos

values = np.concatenate([loss_seen, loss_unseen])
sugar = np.concatenate([np.ones((num_pos)) / num_pos, - np.ones(num_neg) / num_neg])
order = np.argsort(values)

best_pos = np.argmax(np.cumsum(sugar[order]))


# In[153]:


plt.plot(np.cumsum(sugar[order]))


# In[154]:


best_pos


# In[155]:


np.percentile(smoothed_taus, q=np.arange(0, 100, 10))


# In[156]:


x = torch.randn(32, 10, 11)


# In[157]:


ids = torch.from_numpy(np.random.choice(10, size=32))


# In[158]:


x[torch.arange(32), ids, :].size()


# In[159]:


x[2,ids[2]]


# In[160]:


x[torch.arange(32), ids, :][2]


# In[161]:


idx_public = np.load("/private/home/asablayrolles/data/cifar-dejalight/idx_public.npy")


# In[162]:


idx_public.shape


# In[163]:


id_pval = np.random.choice(idx_public.shape[0], 5000, replace=False)


# In[112]:


idb_pval = np.zeros((idx_public.shape[0]), dtype=bool)
idb_pval[id_pval] = 1


# In[113]:


#np.save("/private/home/asablayrolles/data/cifar-dejalight/idx_val_among_public.npy", id_pval)


# In[114]:


id_ptrain = np.nonzero(idb_pval == False)[0]


# In[115]:


#np.save("/private/home/asablayrolles/data/cifar-dejalight/idx_train_among_public.npy", id_ptrain)


# In[116]:


nquant = 3
percentiles = np.percentile(smoothed_taus, np.linspace(0, 100, num=nquant + 1))
percentiles[0] -= 1e-3
percentiles[-1] += 1e-3


# In[117]:


is_higher = np.vstack([(smoothed_taus >= percentiles[i]) for i in range(0, nquant + 1)]).T.astype(int)


# In[118]:


ides = smoothed_taus.argsort()[:10]


# In[119]:


for ide in ides:
    x = 0.5*(1+publicset[ide][1]).numpy()
    x = np.transpose(x, (1, 2, 0))
    
    plt.imshow(x)
    plt.show()


# In[120]:


_ = plt.hist(np.exp(best_taus), bins=200)


# In[121]:


np.mean((smoothed_taus - np.mean(smoothed_taus))**2)


# In[122]:


np.mean(np.exp(best_taus))


# In[123]:


np.mean(smoothed_taus)


# In[124]:


np.mean(best_taus)


# In[125]:


torch.log(1e-5 * torch.ones(1))


# In[227]:


print("Smoothed tau", np.mean(np.exp(smoothed_taus)))
print("Best tau", np.mean(np.exp(best_taus)))


# In[127]:


binary_order = np.argsort(seen_heldout)


# In[128]:


np.mean(seen_heldout[binary_order[5000:]])


# In[236]:


np.mean(
    np.logical_xor(
        losses_heldout - np.exp(smoothed_taus) >= 0, 
        seen_heldout
    )
)



# In[72]:


order = np.argsort(-losses_heldout[binary_order[5000:]])
gt = seen_heldout[binary_order[5000:]].astype(float)
s = np.cumsum(gt[order])


# In[165]:


plt.plot((np.arange(gt.shape[0]) - 2 * s + s[-1]) / gt.shape[0])


# In[232]:


smoothed_taus.shape


# In[229]:


seen.mean()


# In[230]:


seen_heldout.shape


# In[231]:


binary_order.shape


# In[233]:


losses_heldout


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import torch
import numpy as np
from os.path import join


# In[96]:


import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

        def getImgLbl(files, root='/private/home/asablayrolles/data'):
    base_folder = 'cifar-10-batches-py'
    #train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    all_data = []
    all_labels = []
    for f in files:
        fname = os.path.join(root, base_folder, f)
        fo = open(fname, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        all_data.append(entry['data'])
        if 'labels' in entry:
            all_labels += entry['labels']
        else:
            all_labels += entry['fine_labels']
        fo.close()

    all_data = np.concatenate(all_data)
    all_data = all_data.reshape((-1, 3, 32, 32))
    all_data = all_data.transpose((0, 2, 3, 1))  # convert to HWC
    all_labels = np.array(all_labels)
    
    return all_data, all_labels


# In[97]:


dest = "/private/home/asablayrolles/data/cifar-dejalight/"


# In[98]:


data, label = getImgLbl(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'])
testdata, testlabel = getImgLbl(['test_batch'])


# In[99]:


print data.shape, label.shape
print testdata.shape, testlabel.shape


# In[131]:


rs = np.random.RandomState(0)

idx_private = rs.choice(len(data), 15000, replace=False)


# In[132]:


idx_public = np.array(list(set(range(len(data))).symmetric_difference(set(idx_private.tolist()))))


# In[133]:


np.sort(idx_private)[:10]


# In[134]:


np.array(list(idx_public))[:16]


# In[135]:


np.bincount(label[idx_private])


# In[136]:


np.save(join(dest, "idx_private.npy"), idx_private)
np.save(join(dest, "idx_public.npy"), idx_public)


# In[137]:


np.save(join(dest, "data_private.npy"), data[idx_private])
np.save(join(dest, "label_private.npy"), label[idx_private])


# In[145]:


np.save(join(dest, "data_public.npy"), data[idx_public])
np.save(join(dest, "label_public.npy"), label[idx_public])


# In[146]:


data[idx_public].shape


# In[139]:


idx_val = rs.choice(len(testdata), 1000, replace=False)
idx_noval = set(range(len(testdata))).symmetric_difference(set(idx_val.tolist()))


# In[140]:


np.save(join(dest, "data_val.npy"), testdata[idx_val])
np.save(join(dest, "label_val.npy"), testlabel[idx_val])


# In[141]:


print len(idx_val)


# In[142]:


n_public = 30

for i in range(n_public):
    idx = rs.choice(idx_public, 15000, replace=False)
    
    #print len(set(idx).intersection(set(idx_private)))
    
    np.save(join(dest, "idx_public_%d.npy" % i), idx)
    np.save(join(dest, "data_public_%d.npy" % i), data[idx])
    np.save(join(dest, "label_public_%d.npy" % i), label[idx])


# In[114]:


len(idx_public)


# In[143]:


idx_noval = np.array(list(idx_noval))


# In[144]:


np.save(join(dest, "idx_noval.npy"), idx_noval)
np.save(join(dest, "data_noval.npy"), testdata[idx_noval])
np.save(join(dest, "label_noval.npy"), testlabel[idx_noval])


# In[147]:


label.shape


# In[148]:


label[:100]


# In[150]:


plt.imshow(data[1])


# In[ ]:





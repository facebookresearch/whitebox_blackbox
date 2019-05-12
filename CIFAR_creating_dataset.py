import numpy as np
from os.path import join
import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def savecheck(path, x):
    if os.path.exists(path):
        x_saved = np.load(path)
        assert np.array_equal(x_saved, x)
        # print("Saved path %s is correct" % path)
    else:
        np.save(path, x)

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

data, label = getImgLbl(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'])
testdata, testlabel = getImgLbl(['test_batch'])

print(data.shape, label.shape)
print(testdata.shape, testlabel.shape)


if False:
    dest = "/private/home/asablayrolles/data/cifar-dejalight/"

    rs = np.random.RandomState(0)
    idx_private = rs.choice(len(data), 15000, replace=False)
    idx_public = np.array(list(set(range(len(data))).symmetric_difference(set(idx_private.tolist()))))

    savecheck(join(dest, "idx_private.npy"), idx_private)
    savecheck(join(dest, "idx_public.npy"), idx_public)

    savecheck(join(dest, "data_private.npy"), data[idx_private])
    savecheck(join(dest, "label_private.npy"), label[idx_private])

    savecheck(join(dest, "data_public.npy"), data[idx_public])
    savecheck(join(dest, "label_public.npy"), label[idx_public])

    idx_val = rs.choice(len(testdata), 1000, replace=False)
    idx_noval = set(range(len(testdata))).symmetric_difference(set(idx_val.tolist()))

    savecheck(join(dest, "data_val.npy"), testdata[idx_val])
    savecheck(join(dest, "label_val.npy"), testlabel[idx_val])

    n_public = 30
    for i in range(n_public):
        idx = rs.choice(idx_public, 15000, replace=False)
        savecheck(join(dest, "idx_public_%d.npy" % i), idx)
        savecheck(join(dest, "data_public_%d.npy" % i), data[idx])
        savecheck(join(dest, "label_public_%d.npy" % i), label[idx])


    idx_noval = np.array(list(idx_noval))

    savecheck(join(dest, "idx_noval.npy"), idx_noval)
    savecheck(join(dest, "data_noval.npy"), testdata[idx_noval])
    savecheck(join(dest, "label_noval.npy"), testlabel[idx_noval])
else:
    dest = "/private/home/asablayrolles/data/cifar-dejalight2/"

    data = np.concatenate([data, testdata], axis=0)
    label = np.concatenate([label, testlabel], axis=0)

    rs = np.random.RandomState(0)
    indices = rs.choice(len(data), 31000, replace=False)
    idx_private = indices[:15000]
    idx_test = indices[15000:30000]
    idx_val  = indices[30000:31000]

    idx_public = np.array(list(set(range(len(data))).symmetric_difference(set(indices.tolist()))))

    savecheck(join(dest, "idx_private.npy"), idx_private)
    savecheck(join(dest, "idx_public.npy"), idx_public)

    savecheck(join(dest, "data_private.npy"), data[idx_private])
    savecheck(join(dest, "label_private.npy"), label[idx_private])

    savecheck(join(dest, "data_public.npy"), data[idx_public])
    savecheck(join(dest, "label_public.npy"), label[idx_public])

    # validation set
    savecheck(join(dest, "idx_val.npy"), idx_val)
    savecheck(join(dest, "data_val.npy"), data[idx_val])
    savecheck(join(dest, "label_val.npy"), label[idx_val])

    # test set
    savecheck(join(dest, "idx_test.npy"), idx_test)
    savecheck(join(dest, "data_test.npy"), data[idx_test])
    savecheck(join(dest, "label_test.npy"), label[idx_test])


    n_public = 30
    for i in range(n_public):
        idx = rs.choice(idx_public, 15000, replace=False)
        savecheck(join(dest, "idx_public_%d.npy" % i), idx)
        savecheck(join(dest, "data_public_%d.npy" % i), data[idx])
        savecheck(join(dest, "label_public_%d.npy" % i), label[idx])

    assert len(set(idx_private) & set(idx_val)) == 0

    assert len(set(idx_private) & set(idx_public)) == 0
    assert len(set(idx_test)    & set(idx_public)) == 0
    assert len(set(idx_test)    & set(idx_private)) == 0

assert len(set(idx_private)) == idx_private.shape[0]
assert len(set(idx_public))  == idx_public.shape[0]
assert len(set(idx_test))    == idx_test.shape[0]

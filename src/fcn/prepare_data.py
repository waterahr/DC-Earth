import numpy as np
from skimage import io
import os
import random

def generate_img_list(data_root='../../data/round1/'):
    X = []
    Y = []
    for fold_id in range(1, 5):
        for img_id in range(1, 212):
            X.append(data_root+'Z{fid}/Z{fid}-{iid:03}.tif'.format(fid=fold_id, iid=img_id))
            Y.append(data_root+'Z{fid}/Z{fid}-{iid:03}.tif'.format(fid=fold_id, iid=img_id+1))
    return np.asarray(X), np.asarray(Y)

def generate_img_from_namelist(X_path, Y_path, batch_size=4096):
    while True:
        cnt = 0
        X = []
        Y = []
        indices = np.arange(len(X_path))
        random.shuffle(indices)
        X_path = X_path[list(indices)]
        Y_path = Y_path[list(indices)]
        for i in range(len(X_path)):
            img = io.imread(X_path[i])
            X.append(img / 10000)
            img = io.imread(Y_path[i])
            Y.append(img / 10000)
            cnt += 1
            if cnt == batch_size:
                indices = np.arange(batch_size)
                random.shuffle(indices)
                X = np.array(X)[list(indices)][..., np.newaxis]
                Y = np.array(Y)[list(indices)][..., np.newaxis]
                #print(X.shape)
                yield (X, Y)
                cnt = 0
                X = []
                Y = []
                
def generate_img(data_root='../../data/round1/'):
    X = []
    Y = []
    for fold_id in range(1, 5):
        for img_id in range(1, 212):
            img = io.imread(data_root+'Z{fid}/Z{fid}-{iid:03}.tif'.format(fid=fold_id, iid=img_id))
            X.append(img / 10000)
            img = io.imread(data_root+'Z{fid}/Z{fid}-{iid:03}.tif'.format(fid=fold_id, iid=img_id+1))
            Y.append(img / 10000)
    return np.asarray(X), np.asarray(Y)
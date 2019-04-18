import numpy as np
from skimage import io
import os
import random

def generate_img_list(data_root='../../data/round1/', nb_timesteps=12, nb_out=3):
    X = []
    Y = []
    for fold_id in range(1, 5):
        for img_id in range(1, 213-nb_timesteps-nb_out+1):
            x = []
            for j in range(nb_timesteps):
                x.append(data_root+'Z{fid}/Z{fid}-{iid:03}.tif'.format(fid=fold_id, iid=img_id+j))
            X.append(np.asarray(x))
            y = []
            for j in range(nb_out):
                y.append(data_root+'Z{fid}/Z{fid}-{iid:03}.tif'.format(fid=fold_id, iid=img_id+nb_timesteps+j))
            Y.append(np.asarray(y))
    return np.asarray(X), np.asarray(Y)

def generate_img_from_namelist(Xnms, Ynms, nb_variables, nb_timesteps, nb_out, strides=1, batch_size=4096):
    while True:
        cnt = 0
        X = []
        Y = []
        for i in range(len(Xnms)):
            X_container = []
            Y_container = []
            for j in range(nb_timesteps):
                img = io.imread(Xnms[i, j])
                X_container.append(img / 10000)
            for j in range(nb_out):
                img = io.imread(Ynms[i, j])
                Y_container.append(img / 10000)
            X_container = np.asarray(X_container)
            Y_container = np.asarray(Y_container)
            _, h, w = X_container.shape
            padding = nb_variables // 2
            for r in range(padding, h-padding, strides):
                for c in range(padding, w-padding, strides):
                    x = X_container[:, r-padding:r+padding+1, c-padding:c+padding+1]
                    x = x.reshape((nb_timesteps, -1))
                    X.append(x.T)
                    y = Y_container[:, r, c]
                    Y.append(y)
                    cnt = cnt + 1
                    if cnt == batch_size:
                        indices = np.arange(batch_size)
                        random.shuffle(indices)
                        X = np.asarray(X)[list(indices)]#[..., np.newaxis]
                        Y = np.asarray(Y)[list(indices)]#[..., np.newaxis]
                        #print(X.shape)
                        yield (X, Y)
                        cnt = 0
                        X = []
                        Y = []
                        
def generate_datatime_from_namelist(Xnms, Ynms, nb_timesteps, nb_out, batch_size=4096):
    while True:
        cnt = 0
        X = []
        Y = []
        for i in range(len(Xnms)):
            X_container = []
            Y_container = []
            for j in range(nb_timesteps):
                img = io.imread(Xnms[i, j])
                X_container.append(img / 10000)
            for j in range(nb_out):
                img = io.imread(Ynms[i, j])
                Y_container.append(img / 10000)
            X_container = np.asarray(X_container)
            Y_container = np.asarray(Y_container)
            _, h, w = X_container.shape
            for r in range(h):
                for c in range(w):
                    x = X_container[:, r, c]
                    X.append(x[np.newaxis, ...])
                    y = Y_container[:, r, c]
                    Y.append(y)
                    cnt = cnt + 1
                    if cnt == batch_size:
                        indices = np.arange(batch_size)
                        random.shuffle(indices)
                        X = np.asarray(X)[list(indices)]#[..., np.newaxis]
                        Y = np.asarray(Y)[list(indices)]#[..., np.newaxis]
                        #print(X.shape)
                        yield (X, Y)
                        cnt = 0
                        X = []
                        Y = []
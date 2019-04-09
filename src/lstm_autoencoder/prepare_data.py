import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io
import os
import glob

class EarthDataset(Dataset):
    def __init__(self, seq_len=12, out_len=3):
        self.dataset_len = 1
        self.data_root = '../../data/round1/'
        self.seq_len = seq_len
        self.out_len = out_len
        self.data_container = np.zeros((1200, 1200, self.seq_len), dtype=np.float32)
        self.label_container = np.zeros((1200, 1200, self.out_len), dtype=np.float32)
        self.data = None
        self.label = None

    def __getitem__(self, index):
        dt = self.data[index, ...].view(-1, 1)
        lb = self.label[index, ...]
        return dt, lb

    def __len__(self):
        return self.dataset_len

    def update_sampled_img(self):
        fold_id = np.random.randint(1, 5)
        img_id = np.random.randint(1, 212-self.seq_len-self.out_len+1)
        for s in range(self.seq_len):
            img = io.imread(self.data_root+'Z{fid}/Z{fid}-{iid:03}.tif'.format(fid=fold_id, iid=img_id+s))
            self.data_container[..., s] = img / 10000

        for s in range(self.out_len):
            img = io.imread(self.data_root+'Z{fid}/Z{fid}-{iid:03}.tif'.format(fid=fold_id, iid=img_id+self.seq_len+s))
            # print(img.shape)
            # print(img.dtype)
            self.label_container[..., s] = img / 10000

        mask = np.sum(self.data_container > 0, axis=-1) > self.seq_len/2
        self.data = torch.from_numpy(self.data_container[mask, :])
        self.label = torch.from_numpy(self.label_container[mask, :])
        self.dataset_len = self.data.shape[0]
        """
        (1200, 1200)
        1126826
        (1126826, 3)
        data shape:  torch.Size([1126826, 12]) label shape:  torch.Size([1126826, 3])
        torch.float32
        """
        #print(mask)
        #print(mask.shape)
        #print(sum(sum(mask)))
        #print(self.label_container[mask, :].shape)
        #print('data shape: ', self.data.shape, 'label shape: ', self.label.shape)
        #print(self.label.dtype)
        
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    EarthDataset().update_sampled_img()
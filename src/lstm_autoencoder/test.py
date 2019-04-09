import torch
from LSTMAutoEncoder import LSTMAutoEncoder
import numpy as np
from skimage import io
import os
import glob
import argparse
import re
import tqdm

def parse_arg():
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-g', '--gpus', type=str, default='',
                        help='The gpu device\'s ID need to be used')
    parser.add_argument('-w', '--weights', type=str, default='',
                        help='The weights need to be tested')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

def infer(weights_path="../../models/LSTMAutoEncoder/", seq_len=12, out_len=3, gpu_id=0):
    device = torch.device("cuda:{gid}".format(gid=gpu_id) if torch.cuda.is_available() else "cpu")
    print(device)
    
    reg = "epoch_*"
    #print(reg)
    weights = [s for s in os.listdir(weights_path) if re.match(reg, s)]
    print(weights)
    for w in tqdm.tqdm(weights):
        sv_path = '../../results/LSTMAutoEncoder/' + w[:w.index("_loss")] + "/"
        os.makedirs(sv_path, exist_ok=True)
        model = LSTMAutoEncoder()
        state_dict = torch.load(weights_path + w)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        for fold_id in range(1, 5):
            img_nms = sorted(glob.glob("../../data/round1/Z{fold_id}/*tif".format(fold_id=fold_id)))[-seq_len:]
            data_container = torch.zeros((1200, 1200, seq_len), dtype=torch.float32)
            for s, img_nm in enumerate(img_nms):
                img = io.imread(img_nm)
                data_container[..., s] = torch.from_numpy(img / 10000)
            #print(data_container.shape)#torch.Size([1200, 1200, 12])
            data_container = data_container.reshape(-1, seq_len, 1)
            #print(data_container.shape)#torch.Size([1440000, 12, 1])
            data_container = data_container.to(device)
            out_img = []
            with torch.set_grad_enabled(False):
                for i in range(1200):
                    out = model(data_container[i*1200:(i+1)*1200, ...]).reshape(1, 1200, out_len).cpu().detach().numpy()*10000
                    out = np.ceil(out).astype(np.int16)
                    out_img.append(out)
            out = np.concatenate(tuple(out_img), axis=0)
            #print(out.shape)#(1200,1200,3)
            for i in range(out_len):
                io.imsave(sv_path + "Z{fold_id}-21{img_id}.tif".format(fold_id=fold_id, img_id=i+out_len, weight=w), out[..., i])
            
if __name__ == "__main__":
    args = parse_arg()
    infer()
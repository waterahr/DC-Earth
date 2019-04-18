import numpy as np
import os
from skimage import io
import tqdm
import glob
import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=int, default=0,
                        help='The model number used to merge.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arg()
    root_dir = "./"
    weights = [0.7, 0.3]#, 0.1, 0.1, 0.1
    result_dirs = ["MLSTM_FCN/3v_12t_3o_epoch007/", "MALSTM_FCN/3v_12t_3o_3s_epoch036/"]
    #, "MALSTM_FCN/3v_12t_3o_epoch010/", "LSTMAutoEncoder/epoch_100/"
    #, "LSTMAutoEncoder/epoch_080/", "ALSTM_FCN/3v_12t_3o_epoch030/", "LSTM_FCN/3v_12t_3o_epoch030/"
    sv_path = root_dir + "wave_MLSTMFCN007_MALSTMFCN036/"
    #_MALSTMFCN010_LSTMAutoEncoder100
    #_LSTMAutoEncoder080_ALSTMFCN030_LSTMFCN030
    os.makedirs(sv_path, exist_ok=True)
    img_nms = []
    for fold_id in range(1, 5):
        for img_id in range(213, 216):
            img_nms.append("Z{fid}-{iid}.tif".format(fid=fold_id, iid=img_id))
    for i in range(args.model):
        result_dirs.append(input("The Results' Directionary Path of the " + str(i+1) + " Model:"))
        
    for img in tqdm.tqdm(img_nms):
        data_container = np.zeros((1200, 1200), dtype=np.float32)
        idx = 0
        for rdir in tqdm.tqdm(result_dirs):
            # print(root_dir + rdir + "/*.tif")
            # img_nms = sorted(glob.glob(root_dir + rdir + "/*.tif"))
            # print(img_nms)
            # data_container += io.imread(root_dir + rdir + img)
            data_container += weights[idx] * io.imread(root_dir + rdir + img)
            idx += 1
        # data_container /= len(result_dirs)
        io.imsave(sv_path + img, data_container)
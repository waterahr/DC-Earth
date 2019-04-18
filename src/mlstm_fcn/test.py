import numpy as np
import os
import glob
import argparse
import re
import tqdm

from prepare_data import *
from build_models import *

def parse_arg():
    model_nms = ["MLSTM_FCN", "MALSTM_FCN", "LSTM_FCN", "ALSTM_FCN", "LSTM", "ALSTM"]
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=str, default="",
                        help='The model name.')
    parser.add_argument('-g', '--gpus', type=str, default="",
                        help='The gpu device\'s ID need to be used.')
    parser.add_argument('-w', '--weights', type=str, default="",
                        help='The weight file need to be loaded.')
    args = parser.parse_args()
    if args.model == "" or args.model not in model_nms:
        raise RuntimeError('NO MODEL FOUND IN ' + str(model_nms))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

if __name__ == "__main__":
    print("-----------------testing begining---------------------")
    args = parse_arg()
    shape = (1200, 1200)
    nb_variables = 3#3*3=9
    nb_timesteps = 12
    nb_out = 3
    model_prefix = "../../models/" + args.model + "/"
    
    
    if args.model == "MLSTM_FCN":
        model = build_MLSTM_FCN(nb_variables*nb_variables, nb_timesteps, nb_out)
    elif args.model == "MALSTM_FCN":
        model = build_MALSTM_FCN(nb_variables*nb_variables, nb_timesteps, nb_out)
    elif args.model == "LSTM_FCN":
        model = build_LSTM_FCN(nb_variables*nb_variables, nb_timesteps, nb_out)
    elif args.model == "ALSTM_FCN":
        model = build_ALSTM_FCN(nb_variables*nb_variables, nb_timesteps, nb_out)
    elif args.model == "LSTM":
        model = build_LSTM(nb_timesteps, nb_out)
    elif args.model == "ALSTM":
        model = build_ALSTM(nb_timesteps, nb_out)
    
    
    if args.weights == "":
        reg = str(nb_variables) + "v_" + str(nb_timesteps) + "t_" + str(nb_out) + "o_(e|f)1_*"
    else:
        reg = args.weights[args.weights.index(str(nb_variables) + "v_" + str(nb_timesteps) + "t_" + str(nb_out) + "o"):]
    print(reg)
    weights = [s for s in os.listdir(model_prefix) if re.match(reg, s)]
    print(weights)
    for w in tqdm.tqdm(weights):
        if w.find("_valloss") != -1:
            sv_path = '../../results/' + args.model + '/' + w[:w.index("_valloss")] + "/"
        else:
            sv_path = '../../results/' + args.model + '/' + w[:w.index("_model")] + "/"
        os.makedirs(sv_path, exist_ok=True)
        model.load_weights(model_prefix + w)
        for fold_id in tqdm.tqdm(range(1, 5)):
            img_nms = sorted(glob.glob("../../data/round1/Z{fold_id}/*tif".format(fold_id=fold_id)))[-nb_timesteps:]
            data_container = np.zeros((nb_timesteps, 1200, 1200), dtype=np.float32)
            out_container = np.zeros((nb_out, 1200, 1200), dtype=np.float32)
            for s, img_nm in enumerate(img_nms):
                img = io.imread(img_nm)
                data_container[s] = img / 10000
            if args.model in ["MLSTM_FCN", "MALSTM_FCN", "LSTM_FCN", "ALSTM_FCN"]:
                pad_width = nb_variables // 2
                data_container = np.pad(data_container, pad_width=((0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode="constant")
                for r in tqdm.tqdm(range(pad_width, shape[0]+pad_width)):
                    for c in tqdm.tqdm(range(pad_width, shape[1]+pad_width)):
                        x = data_container[:, r-pad_width:r+pad_width+1, c-pad_width:c+pad_width+1]
                        x = x.reshape((nb_timesteps, -1))
                        x = x.T
                        out_container[:, r-pad_width, c-pad_width] = model.predict(x[np.newaxis, ...])[0] * 10000
            else:
                for r in tqdm.tqdm(range(shape[0])):
                    for c in tqdm.tqdm(range(shape[1])):
                        x = data_container[:, r, c]
                        x = x[np.newaxis, ...]
                        out_container[:, r, c] = model.predict(x[np.newaxis, ...])[0] * 10000
            for i in range(nb_out):
                io.imsave(sv_path + "Z{fold_id}-{img_id}.tif".format(fold_id=fold_id, img_id=i+213), out_container[i, ...])
    print("-----------------testing endding---------------------")
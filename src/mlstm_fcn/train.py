import numpy as np
import os
import glob
import argparse
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger

from prepare_data import *
from build_models import *

def parse_arg():
    model_nms = ["MLSTM_FCN", "MALSTM_FCN", "LSTM_FCN", "ALSTM_FCN", "LSTM", "ALSTM"]
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=str, default="",
                        help='The model name.')
    parser.add_argument('-g', '--gpus', type=str, default="",
                        help='The gpu device\'s ID need to be used.')
    parser.add_argument('-w', '--weight', type=str, default="",
                        help='The weight used to do initial.')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The epochs need to be trained')
    parser.add_argument('-b', '--batch', type=int, default=4096,
                        help='The batch size in the training progress.')
    args = parser.parse_args()
    if args.model == "" or args.model not in model_nms:
        raise RuntimeError('NO MODEL FOUND IN ' + str(model_nms))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

if __name__ == "__main__":
    print("-----------------training begining---------------------")
    args = parse_arg()
    nb_epoch = args.epochs
    batch_size = args.batch
    monitor = 'val_loss'
    shape = 1200
    strides = 3
    nb_variables = 3#3*3=9
    nb_timesteps = 12
    nb_out = 3
    model_prefix = "../../models/" + args.model + "/"
    os.makedirs(model_prefix, exist_ok=True)
    model_prefix = model_prefix + str(nb_variables) + "v_" + str(nb_timesteps) + "t_" + str(nb_out) + "o_" + str(strides) + "s_newopt_"
    
    
    if args.model == "MLSTM_FCN":
        model = build_MLSTM_FCN(nb_variables*nb_variables, nb_timesteps, nb_out, optimizer="adadelta")#adam#rmsprop
    elif args.model == "MALSTM_FCN":
        model = build_MALSTM_FCN(nb_variables*nb_variables, nb_timesteps, nb_out, optimizer="adadelta")
    elif args.model == "LSTM_FCN":
        model = build_LSTM_FCN(nb_variables*nb_variables, nb_timesteps, nb_out)
    elif args.model == "ALSTM_FCN":
        model = build_ALSTM_FCN(nb_variables*nb_variables, nb_timesteps, nb_out)
    elif args.model == "LSTM":
        model = build_LSTM(nb_timesteps, nb_out)
    elif args.model == "ALSTM":
        model = build_ALSTM(nb_timesteps, nb_out)
    
    
    X, Y = generate_img_list(nb_timesteps=nb_timesteps, nb_out=nb_out)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print("The length of the X_train is: ", X_train.shape)
    print("The length of the y_train is: ", Y_train.shape)
    print("The length of the X_test is: ", X_test.shape)
    print("The length of the y_test is: ", Y_test.shape)
    if args.model in ["MLSTM_FCN", "MALSTM_FCN", "LSTM_FCN", "ALSTM_FCN"]:
        train_generator = generate_img_from_namelist(X_train, Y_train, 
                                      nb_variables=nb_variables, nb_timesteps=nb_timesteps, nb_out=nb_out,
                                      strides=strides, batch_size=batch_size)
        val_generator = generate_img_from_namelist(X_test, Y_test,  
                                      nb_variables=nb_variables, nb_timesteps=nb_timesteps, nb_out=nb_out,
                                      strides=strides, batch_size=batch_size)
        steps_train = int(X_train.shape[0] * ((shape - nb_variables // 2 * 2) / strides)**2 / batch_size)
        steps_val = int(X_test.shape[0] * ((shape - nb_variables // 2 * 2) / strides)**2 / batch_size)
    else:
        train_generator = generate_datatime_from_namelist(X_train, Y_train, nb_timesteps=nb_timesteps, nb_out=nb_out, batch_size=batch_size)
        val_generator = generate_datatime_from_namelist(X_test, Y_test, nb_timesteps=nb_timesteps, nb_out=nb_out, batch_size=batch_size)
        steps_train = int(X_train.shape[0] * shape**2 / batch_size)
        steps_val = int(X_test.shape[0] * shape**2 / batch_size)
    
    
    checkpointer = ModelCheckpoint(filepath = model_prefix + 'epoch{epoch:03d}_valloss{'+ monitor + ':.6f}.hdf5',
                        monitor = monitor,
                        verbose=1, 
                        save_best_only=True, 
                        save_weights_only=True,
                        mode='auto', 
                        period=1)
    csvlog = CSVLogger(model_prefix + str(args.epochs) + 'iter' + '_log.csv', append=True)
    if args.weight != "":
        model.load_weights(args.weight)
    model.fit_generator(train_generator,
            steps_per_epoch = steps_train,
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = steps_val,
            callbacks = [checkpointer, csvlog], 
            workers = 1)#, initial_epoch = 200
    model.save_weights(model_prefix + 'final' + str(args.epochs) + 'iter_model.h5')
    print("-----------------training endding---------------------")
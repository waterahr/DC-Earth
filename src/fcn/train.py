import numpy as np
import os
import glob
import argparse
from sklearn.model_selection import train_test_split
from keras.metrics import binary_accuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger

from prepare_data import *
from FCN import *
from metrics import *
from loss_functions import *

def parse_arg():
    model_nms = ["FCN_Vgg16_32s", "AtrousFCN_Vgg16_16s", "FCN_Resnet50_32s", "AtrousFCN_Resnet50_16s", "Atrous_DenseNet", "DenseNet_FCN"]
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=str, default="",
                        help='The model name.')
    parser.add_argument('-g', '--gpus', type=str, default="",
                        help='The gpu device\'s ID need to be used.')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The epochs need to be trained')
    parser.add_argument('-b', '--batch', type=int, default=16,
                        help='The batch size in the training progress.')
    args = parser.parse_args()
    if args.model == "" or args.model not in model_nms:
        raise RuntimeError('NO MODEL FOUND IN ' + str(model_nms))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args



if __name__ == "__main__":
    args = parse_arg()
    model_prefix = "../../models/" + args.model + "/"
    os.makedirs(model_prefix, exist_ok=True)
    nb_epoch = args.epochs
    batch_size = args.batch
    input_shape = (1200, 1200, 1)
    monitor = 'val_loss'
    batchnorm_momentum = 0.95
    if args.model is 'AtrousFCN_Resnet50_16s':
        weight_decay = 0.0001 / 2
    else:
        weight_decay = 1e-4
    
    
    model = globals()[args.model](weight_decay=weight_decay,
                        input_shape=input_shape,
                        batch_momentum=batchnorm_momentum,
                        classes=1)
    loss_func = softmax_sparse_crossentropy_ignoring_last_label
    loss_weights = None
    metrics=[sparse_accuracy_ignoring_last_label]
    model.compile(loss=loss_func, optimizer='adam', loss_weights=loss_weights, metrics=metrics)
    #opt_sgd = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    #model.compile(loss=loss_func, optimizer=opt_sgd, loss_weights=loss_weights, metrics=metrics)
    model.summary()
    
    
    X, Y = generate_img_list()
    #X, Y = generate_img()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print("The length of the X_train is: ", X_train.shape)
    print("The length of the y_train is: ", Y_train.shape)
    print("The length of the X_test is: ", X_test.shape)
    print("The length of the y_test is: ", Y_test.shape)
    train_generator = generate_img_from_namelist(X_train, Y_train, batch_size=batch_size)
    val_generator = generate_img_from_namelist(X_test, Y_test, batch_size=batch_size)
    
    
    checkpointer = ModelCheckpoint(filepath = model_prefix + 'epoch{epoch:03d}_valloss{'+ monitor + ':.6f}.hdf5',
                        monitor = monitor,
                        verbose=1, 
                        #save_best_only=True, 
                        save_weights_only=True,
                        #mode='max', 
                        period=1)
    csvlog = CSVLogger(model_prefix + str(args.epochs) + 'iter' + '_log.csv', append=True)
    model.fit_generator(train_generator,
            steps_per_epoch = int(X_train.shape[0] / batch_size),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test.shape[0] / batch_size),
            callbacks = [checkpointer, csvlog], 
            workers = 1)#, initial_epoch = 200
    model.save_weights(model_prefix + 'final' + str(args.epochs) + 'iter_model.h5')
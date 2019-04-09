import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
#from keras_contrib.applications import densenet
#from keras.applications.densenet import DenseNet201
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.layers.merge import Add
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf

from BilinearUpSampling import *


# The original help functions from keras does not have weight regularizers, so I modified them.
# Also, I changed these two functions into functional style
def identity_block(kernel_size, filters, stage, block, weight_decay=0., batch_momentum=0.99):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                          padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    return f

def conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(2, 2), batch_momentum=0.99):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                          name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                          name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                                 name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    return f

# Atrous-Convolution version of residual blocks
def atrous_identity_block(kernel_size, filters, stage, block, weight_decay=0., atrous_rate=(2, 2), batch_momentum=0.99):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,
                          padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    return f

def atrous_conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(1, 1), atrous_rate=(2, 2), batch_momentum=0.99):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), strides=strides,
                          name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,
                          name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                                 name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    return f

def conv_relu(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_relu'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                                 kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = Activation("relu")(x)
        return x
    return f

def conv_bn(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_bn'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                              kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
        return x
    return f

def conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('conv_bn_relu'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                              kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
        return x
    return f

def bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('bn_relu_conv'):
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), stride=subsample, use_bias=bias,
                              kernel_initializer="he_normal", W_regularizer=l2(w_decay), border_mode=border_mode)(x)
        return x
    return f

def atrous_conv_bn(nb_filter, nb_row, nb_col, atrous_rate=(2, 2), subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('atrous_conv_bn'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), dilation_rate=atrous_rate, stride=subsample, use_bias=bias,
                       kernel_initializer="he_normal", kernel_regularizer=l2(w_decay), padding=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
        return x
    return f

def atrous_conv_bn_relu(nb_filter, nb_row, nb_col, atrous_rate=(2, 2), subsample=(1, 1), border_mode='same', bias = True, w_decay = 0.01):
    def f(x):
        with tf.name_scope('atrous_conv_bn_relu'):
            x = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), dilation_rate=atrous_rate, stride=subsample, use_bias=bias,
                       kernel_initializer="he_normal", kernel_regularizer=l2(w_decay), padding=border_mode)(x)
            x = BatchNormalization(mode=0, axis=-1)(x)
            x = Activation("relu")(x)
        return x
    return f

def top(x, input_shape, classes, activation, weight_decay):

    x = Conv2D(classes, (1, 1), activation='linear',
               padding='same', kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)

    if K.image_data_format() == 'channels_first':
        channel, row, col = input_shape
    else:
        row, col, channel = input_shape

    # TODO(ahundt) this is modified for the sigmoid case! also use loss_shape
    if activation is 'sigmoid':
        x = Reshape((row * col * classes,))(x)

    return x


def FCN_Vgg16_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)

    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    #model.load_weights(weights_path, by_name=True)
    return model


def AtrousFCN_Vgg16_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2),
                      name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)

    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    #model.load_weights(weights_path, by_name=True)
    return model


def FCN_Resnet50_32s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

    x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)
    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    #model.load_weights(weights_path, by_name=True)
    return model


def AtrousFCN_Resnet50_16s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)
    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    #model.load_weights(weights_path, by_name=True)
    return model

"""
def Atrous_DenseNet(input_shape=None, weight_decay=1E-4,
                    batch_momentum=0.9, batch_shape=None, classes=21,
                    include_top=False, activation='sigmoid'):
    # TODO(ahundt) pass the parameters but use defaults for now
    if include_top is True:
        # TODO(ahundt) Softmax is pre-applied, so need different train, inference, evaluate.
        # TODO(ahundt) for multi-label try per class sigmoid top as follows:
        # x = Reshape((row * col * classes))(x)
        # x = Activation('sigmoid')(x)
        # x = Reshape((row, col, classes))(x)
        return densenet.DenseNet(depth=None, nb_dense_block=3, growth_rate=32,
                                 nb_filter=-1, nb_layers_per_block=[6, 12, 24, 16],
                                 bottleneck=True, reduction=0.5, dropout_rate=0.2,
                                 weight_decay=1E-4,
                                 include_top=True, top='segmentation',
                                 weights=None, input_tensor=None,
                                 input_shape=input_shape,
                                 classes=classes, transition_dilation_rate=2,
                                 transition_kernel_size=(1, 1),
                                 transition_pooling=None)

    # if batch_shape:
    #     img_input = Input(batch_shape=batch_shape)
    #     image_size = batch_shape[1:3]
    # else:
    #     img_input = Input(shape=input_shape)
    #     image_size = input_shape[0:2]

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=16,
                                      data_format=K.image_data_format(),
                                      include_top=False)
    img_input = Input(shape=input_shape)

    x = densenet.__create_dense_net(classes, img_input,
                                    depth=None, nb_dense_block=3, growth_rate=32,
                                    nb_filter=-1, nb_layers_per_block=[6, 12, 24, 16],
                                    bottleneck=True, reduction=0.5, dropout_rate=0.2,
                                    weight_decay=1E-4, top='segmentation',
                                    input_shape=input_shape,
                                    transition_dilation_rate=2,
                                    transition_kernel_size=(1, 1),
                                    transition_pooling=None,
                                    include_top=include_top)

    x = top(x, input_shape, classes, activation, weight_decay)

    model = Model(img_input, x, name='Atrous_DenseNet')
    # TODO(ahundt) add weight loading
    return model


def DenseNet_FCN(input_shape=None, weight_decay=1E-4,
                 batch_momentum=0.9, batch_shape=None, classes=21,
                 include_top=False, activation='sigmoid'):
    if include_top is True:
        # TODO(ahundt) Softmax is pre-applied, so need different train, inference, evaluate.
        # TODO(ahundt) for multi-label try per class sigmoid top as follows:
        # x = Reshape((row * col * classes))(x)
        # x = Activation('sigmoid')(x)
        # x = Reshape((row, col, classes))(x)
        return densenet.DenseNetFCN(input_shape=input_shape,
                                    weights=None, classes=classes,
                                    nb_layers_per_block=[4, 5, 7, 10, 12, 15],
                                    growth_rate=16,
                                    dropout_rate=0.2)

    # if batch_shape:
    #     img_input = Input(batch_shape=batch_shape)
    #     image_size = batch_shape[1:3]
    # else:
    #     img_input = Input(shape=input_shape)
    #     image_size = input_shape[0:2]

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=16,
                                      data_format=K.image_data_format(),
                                      include_top=False)
    img_input = Input(shape=input_shape)

    x = densenet.__create_fcn_dense_net(classes, img_input,
                                        input_shape=input_shape,
                                        nb_layers_per_block=[4, 5, 7, 10, 12, 15],
                                        growth_rate=16,
                                        dropout_rate=0.2,
                                        include_top=include_top)

    x = top(x, input_shape, classes, activation, weight_decay)
    # TODO(ahundt) add weight loading
    model = Model(img_input, x, name='DenseNet_FCN')
    return model
    """
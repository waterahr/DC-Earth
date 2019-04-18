import os
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from MLSTM_FCN.utils.layer_utils import AttentionLSTM


def squeeze_excite_block(inpt):
    """
    Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    """
    filters = inpt._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(inpt)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([inpt, se])
    return se

def build_LSTM(nb_timesteps, nb_out, nb_lstm=32, optimizer="adam", loss="mean_squared_error", metrics=["mae"]):
    inpt = Input(shape=(1, nb_timesteps))
    
    x = Masking()(inpt)
    x = LSTM(nb_lstm)(x)
    x = Dropout(0.8)(x)
    
    out = Dense(nb_out, activation="sigmoid")(x)

    model = Model(inpt, out, name="LSTM")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

def build_ALSTM(nb_timesteps, nb_out, nb_lstm=32, optimizer="adam", loss="mean_squared_error", metrics=["mae"]):
    inpt = Input(shape=(1, nb_timesteps))
    
    x = Masking()(inpt)
    x = AttentionLSTM(nb_lstm)(x)
    x = Dropout(0.8)(x)
    
    out = Dense(nb_out, activation="sigmoid")(x)

    model = Model(inpt, out, name="ALSTM")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

def build_MLSTM_FCN(nb_variables, nb_timesteps, nb_out, nb_lstm=8, optimizer="adam", loss="mean_squared_error", metrics=["mae"]):
    inpt = Input(shape=(nb_variables, nb_timesteps))
    
    x = Masking()(inpt)
    x = LSTM(nb_lstm)(x)
    x = Dropout(0.8)(x)
    
    y = Permute((2, 1))(inpt)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])
    
    out = Dense(nb_out)(x)#, activation="sigmoid"

    model = Model(inpt, out, name="MLSTM_FCN")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

def build_MALSTM_FCN(nb_variables, nb_timesteps, nb_out, nb_lstm=32, stride=1, optimizer="adam", loss="mean_squared_error", metrics=["mae"]):
    inpt = Input(shape=(nb_variables, nb_timesteps))
    
    ''' sabsample timesteps to prevent OOM due to Attention LSTM '''
    # stride = 2
    
    x = Permute((2, 1))(inpt)
    x = Conv1D(nb_variables // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
               kernel_initializer='he_uniform')(x) # (None, variables / stride, timesteps)
    x = Permute((2, 1))(x)

    x = Masking()(x)
    x = AttentionLSTM(nb_lstm)(x)
    x = Dropout(0.8)(x)
    
    y = Permute((2, 1))(inpt)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])
    
    out = Dense(nb_out)(x)#, activation="sigmoid"

    model = Model(inpt, out, name="MALSTM_FCN")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

def build_LSTM_FCN(nb_variables, nb_timesteps, nb_out, nb_lstm=8, optimizer="adam", loss="mean_squared_error", metrics=["mae"]):
    inpt = Input(shape=(nb_variables, nb_timesteps))
    
    x = Masking()(inpt)
    x = LSTM(nb_lstm)(x)
    x = Dropout(0.8)(x)
    
    y = Permute((2, 1))(inpt)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])
    
    out = Dense(nb_out, activation="sigmoid")(x)

    model = Model(inpt, out, name="LSTM_FCN")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

def build_ALSTM_FCN(nb_variables, nb_timesteps, nb_out, nb_lstm=8, optimizer="adam", loss="mean_squared_error", metrics=["mae"]):
    inpt = Input(shape=(nb_variables, nb_timesteps))
    
    ''' sabsample timesteps to prevent OOM due to Attention LSTM '''
    # stride = 2
    
    # x = Permute((2, 1))(inpt)
    # x = Conv1D(nb_variables // stride, 8, strides=stride, padding='same',
    #            activation='relu', use_bias=False, kernel_initializer='he_uniform')(x) # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    x = Masking()(inpt)
    x = AttentionLSTM(nb_lstm)(x)
    x = Dropout(0.8)(x)
    
    y = Permute((2, 1))(inpt)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    # y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])
    
    out = Dense(nb_out)(x)

    model = Model(inpt, out, name="ALSTM_FCN")
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    build_LSTM(12, 3)
    #build_MLSTM_FCN(6, 12, 3)
    #build_MALSTM_FCN(6, 12, 3)
    #build_LSTM_FCN(6, 12, 3)
    #build_ALSTM_FCN(6, 12, 3)
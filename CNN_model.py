
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.utils.data_utils import get_file
from keras.layers import Input, Dense


def MusicClassifierCNN(melgram_input, input_shape=None):

    if input_shape:
        input_shape = input_shape
    else:
        input_shape = melgram_input.shape

    channel_axis = 3
    freq_axis = 1
    time_axis = 2

    # Input block
    x = BatchNormalization(axis=time_axis, name='bn_0_freq', trainable=True)(melgram_input)

    # Conv block 1
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv1', trainable=True)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1', trainable=True)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1', trainable=True)(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2', trainable=True)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2', trainable=True)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3', trainable=True)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3', trainable=True)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(192, 3, 3, border_mode='same', name='conv4', trainable=True)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4', trainable=True)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4', trainable=True)(x)

    # Conv block 5
    x = Convolution2D(256, 3, 3, border_mode='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    # Output
    x = Flatten(name='Flatten_1')(x) ## we can get the features of this layer if we want to run unsupervised approach on music mel spectrogram

    # Create model
    x = Dense(10, activation='softmax', name='output')(x) ## Here just set arbitrary 10 classes --> to be adjusted to correct number of subgeneres we want to classify
    model = Model(melgram_input, x)
    return model
   
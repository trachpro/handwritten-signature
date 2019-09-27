from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

from keras.layers import MaxPool2D

def SegNet(input_shape, num_classes, trained_weights=None):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(96, (11, 11),strides=4)(inputs)
    conv_1 = BatchNormalization()(conv_1)
    pool_1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(conv_1)
    
    conv_2 = Convolution2D(256, (5, 5), strides=2, padding='same')(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    pool_2 = MaxPool2D(pool_size=(3, 3), strides=2,)(conv_2)

    conv_3 = Convolution2D(384, (3, 3), strides=1, padding='same')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Convolution2D(384, (3, 3), strides=2, padding='same')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Convolution2D(256, (3, 3), strides=2, padding='same')(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    pool_3 = MaxPool2D(pool_size=(3, 3), strides=2,)(conv_3)

    flat = Flatten()(pool_3)
    
    fc1 = Dense(units=2048)(flat)
    fc1 = BatchNormalization()(fc1)
    
    fc2 = Dense(units=2048, name="fc2")(fc1)
    fc2 = BatchNormalization()(fc2)
    
    outputs = Dense(units=num_classes, activation='softmax', name='predictions')(fc2)
    
    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model
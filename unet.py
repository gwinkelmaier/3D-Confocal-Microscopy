import tensorflow as tf
from tensorflow import math
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import Loss

def define_model():
    _input = Input(shape=[128,128,32,1])

    _n_neuron_base = 64
    # Down-Convolution Block
    conv1 = Conv3D(_n_neuron_base*1, (3,3,3), activation='relu', padding='same')(_input)
    conv1 = Conv3D(_n_neuron_base*1, (3,3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2,2,2))(conv1)
    # pool1 = Dropout(0.25)(pool1)
    pool1 = BatchNormalization()(pool1)

    # Down-Convolution Block
    conv2 = Conv3D(_n_neuron_base*2, (3,3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(_n_neuron_base*2, (3,3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2,2,2))(conv2)
    # pool2 = Dropout(0.5)(pool2)
    pool2 = BatchNormalization()(pool2)

    # Down-Convolution Block
    conv3 = Conv3D(_n_neuron_base*4, (3,3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(_n_neuron_base*4, (3,3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D((2,2,2))(conv3)
    # pool3 = Dropout(0.5)(pool3)
    pool3 = BatchNormalization()(pool3)

    # Bottom Block
    convm = Conv3D(_n_neuron_base*8, (3,3,3), activation="relu", padding='same')(pool3)
    convm = Conv3D(_n_neuron_base*8, (3,3,3), activation="relu", padding='same')(convm)

    # Up-Convolution Block
    deconv3 = Conv3DTranspose(_n_neuron_base*4, (3,3,3), strides=(2,2,2), activation='relu', padding='same')(convm)
    uconv3 = Concatenate()([deconv3, conv3])
    # uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv3D(_n_neuron_base*4, (3,3,3), activation='relu', padding='same')(uconv3)
    uconv3 = Conv3D(_n_neuron_base*4, (3,3,3), activation='relu', padding='same')(uconv3)

    # Up-Convolution Block
    deconv2 = Conv3DTranspose(_n_neuron_base*2, (3,3,3), strides=(2,2,2), activation='relu', padding='same')(uconv3)
    uconv2 = Concatenate()([deconv2, conv2])
    # uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv3D(_n_neuron_base*2, (3,3,3), activation='relu', padding='same')(uconv2)
    uconv2 = Conv3D(_n_neuron_base*2, (3,3,3), activation='relu', padding='same')(uconv2)

    # Up-Convolution Block
    deconv1 = Conv3DTranspose(_n_neuron_base*1, (3,3,3), strides=(2,2,2), activation='relu', padding='same')(uconv2)
    uconv1 = Concatenate()([deconv1, conv1])
    # uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv3D(_n_neuron_base*1, (3,3,3), activation='relu', padding='same')(uconv1)
    uconv1 = Conv3D(_n_neuron_base*1, (3,3,3), activation='relu', padding='same')(uconv1)

    # Output Layer
    _output = Conv3D(2, (1,1,1), activation='sigmoid', padding='same')(uconv1)

    # Model Construction
    return Model(_input, _output)


if __name__ == "__main__":
    '''
    Verify model is working and print the architecture
    '''
    import numpy as np
    from tensorflow.keras.utils import plot_model

    x = np.zeros([4,128,128,32,1])

    model = define_model()
    model.compile(loss='binary_crossentropy', optimizer='Adam')

    y_hat = model.predict( x )
    model.summary()
    plot_model( model, to_file='images/UNet.png', show_shapes=True,
                show_layer_names=True)

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.python.keras import regularizers

import tensorflow as tf
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True, regularization=0.01):
    """
    Convolutional downsampling block

    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns:
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    ## the 2 convolutions:
    conv = Conv2D(n_filters,  # Number of filters
                  (3, 3),  # Kernel size
                  activation='relu',
                  kernel_regularizer=regularizers.l2(regularization),
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters,  # Number of filters
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_regularizer=regularizers.l2(regularization),
                  kernel_initializer='he_normal')(conv)

    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D()(conv)

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32, regularization=0.01):
    """
    Convolutional upsampling block

    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns:
        conv -- Tensor output
    """

    up = Conv2DTranspose(
        n_filters,  # number of filters
        3,  # Kernel size
        strides=(2, 2),
        padding='same')(expansive_input)

    # Merge the previous output and the contractive_input

    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,  # Number of filters
                  (3, 3),  # Kernel size
                  activation='relu',
                  kernel_regularizer=regularizers.l2(regularization),
                  padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,  # Number of filters
                  (3, 3),  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_regularizer=regularizers.l2(regularization),
                  kernel_initializer='he_normal')(conv)

    return conv

def unet_model(input_size=(256, 256, 1), n_filters=32,n_classes=5, dropout_probability=0, regularization=0.01):
    """
    Unet model

    Arguments:
        input_size -- Input shape
        n_filters -- Number of filters for the convolutional layers
    Returns:
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    filters = n_filters
    cblock1 = conv_block(inputs, n_filters, dropout_prob=dropout_probability)
    # Chain the first element of the output of each block to be the input of the next conv_block.
    # Double the number of filters at each new step
    cblock2 = conv_block(cblock1[0], 2 * filters, dropout_prob=dropout_probability, regularization=regularization)
    cblock3 = conv_block(cblock2[0], 4 * filters, dropout_prob=dropout_probability, regularization=regularization)
    cblock4 = conv_block(cblock3[0], 8 * filters, dropout_prob=dropout_probability, regularization=regularization)  # Include a dropout_prob of 0.3 for this layer
    # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
    cblock5 = conv_block(cblock4[0], 16 * filters, dropout_prob=dropout_probability, max_pooling=False, regularization=regularization)

    # Expanding Path (decoding)
    # Add the first upsampling_block.
    ublock6 = upsampling_block(cblock5[0], cblock4[1], 8 * filters, regularization=regularization)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer.
    # At each step, use half the number of filters of the previous block
    ublock7 = upsampling_block(ublock6, cblock3[1], 4 * filters, regularization=regularization)
    ublock8 = upsampling_block(ublock7, cblock2[1], 2 * filters, regularization=regularization)
    ublock9 = upsampling_block(ublock8, cblock1[1], filters, regularization=regularization)

    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer = regularizers.l2(regularization),
                   kernel_initializer='he_normal')(ublock9)

    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_regularizer = regularizers.l2(regularization),
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1,activation = 'softmax', padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model
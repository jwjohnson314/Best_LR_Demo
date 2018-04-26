from keras.models import Model, Sequential
from keras.layers import Conv2D, Input, Dense, Dropout, BatchNormalization
from keras.layers import Flatten, MaxPool2D, AveragePooling2D, Add, Activation
from keras.regularizers import l2


# conv block for residual cifar10 model
def conv_block(input_depth, output_depth, strides, prev_layer, init,
        activation, padding, regularizer):
    if input_depth == output_depth:
        x = BatchNormalization()(prev_layer)
        x = Activation(activation)(x)
        x = Conv2D(output_depth, 3, strides=strides, padding=padding,
                kernel_initializer=init, kernel_regularizer=regularizer,
                bias_regularizer=regularizer)(x)
    else:
        x = prev_layer
        # downsample on width increase
        x = Conv2D(output_depth, 3, strides=(2, 2), padding=padding,
                kernel_initializer=init, kernel_regularizer=regularizer,
                bias_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(output_depth, 3, padding=padding, kernel_initializer=init,
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    return x

# skip block for residual cifar10 model
def skip_block(input_depth, output_depth, strides, 
        prev_layer, init, padding, regularizer):
    if input_depth != output_depth:
        prev_layer = Conv2D(output_depth, 1, strides=(2, 2), padding=padding,
                kernel_initializer=init, kernel_regularizer=regularizer,
                use_bias=False)(prev_layer)
    return prev_layer


# residual block for cifar10 model
def residual_block(input_depth, output_depth, strides, prev_layer, init,
        activation, padding, regularizer):
    conv = conv_block(input_depth, output_depth, strides, prev_layer, init,
            activation, padding, regularizer)
    skip = skip_block(input_depth, output_depth, strides, prev_layer, init,
            padding, regularizer)
    add = Add()([conv, skip])
    return add


def mnist_model(activation='relu', padding='same'):
    model = Sequential()
    model.add(Conv2D(16, 5, activation=activation, padding=padding,
        input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(Conv2D(32, 5, activation=activation, padding=padding))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dense(10, activation='softmax'))
    return model


def cifar_model(init='he_normal', strides=(1, 1), padding='same',
        activation='relu', weight_decay=0.0005):
    # WRN-28-10, without dropout https://arxiv.org/abs/1605.07146
    regularizer=l2(weight_decay)

    inputs = Input((32, 32, 3))
    conv0 = Conv2D(16, 3, strides=strides, padding=padding, kernel_initializer=init,
                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(inputs)
    batch0 = BatchNormalization()(conv0)
    act0 = Activation(activation)(batch0)

    # first residual block slightly different from the rest
    # because there is no downsampling
    conv1_l = Conv2D(160, 3, padding=padding, strides=strides, kernel_initializer=init,
            bias_initializer=init, kernel_regularizer=regularizer,
            bias_regularizer=regularizer)(act0)
    batch1_l = BatchNormalization()(conv1_l)
    act1_l = Activation(activation)(batch1_l)
    conv2_l = Conv2D(160, 3, padding=padding, strides=strides, kernel_initializer=init,
            bias_initializer=init, kernel_regularizer=regularizer,
            bias_regularizer=regularizer)(act1_l)

    conv1_r = Conv2D(160, 1, padding=padding, strides=strides, kernel_initializer=init,
            bias_initializer=init, kernel_regularizer=regularizer,
            bias_regularizer=regularizer)(act0)
    add1 = Add()([conv2_l, conv1_r])
    
    # now the 'normal' residual blocks with downsampling at each width change
    res1 = residual_block(160, 160, strides, add1, init, activation, padding, regularizer)
    res2 = residual_block(160, 160, strides, res1, init, activation, padding, regularizer)
    res3 = residual_block(160, 160, strides, res2, init, activation, padding, regularizer)
    block1_batch = BatchNormalization()(res3)
    block1_act = Activation(activation)(block1_batch)
    res4 = residual_block(160, 320, strides, block1_act, init, activation, padding, regularizer)
    res5 = residual_block(320, 320, strides, res4, init, activation, padding, regularizer)
    res6 = residual_block(320, 320, strides, res5, init, activation, padding, regularizer)
    res7 = residual_block(320, 320, strides, res6, init, activation, padding, regularizer)
    block2_batch = BatchNormalization()(res7)
    block2_act = Activation(activation)(block2_batch)
    res8 = residual_block(320, 640, strides, block2_act, init, activation, padding, regularizer)
    res9 = residual_block(640, 640, strides, res8, init, activation, padding, regularizer)
    res10 = residual_block(640, 640, strides, res9, init, activation, padding, regularizer)
    res11 = residual_block(640, 640, strides, res10, init, activation, padding, regularizer)
    block3_batch = BatchNormalization()(res11)
    block3_act = Activation(activation)(block3_batch)
    pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1))(block3_act)
    flat = Flatten()(pool)
    outputs = Dense(10, kernel_initializer=init, kernel_regularizer=regularizer,
            bias_regularizer=regularizer, activation='softmax')(flat)

    return Model(inputs=inputs, outputs=outputs)

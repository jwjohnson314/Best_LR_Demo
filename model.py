from keras.models import Model, Sequential
from keras.layers import Conv2D, Input, Dense, Dropout, BatchNormalization, Flatten, MaxPool2D, GlobalAveragePooling2D, Add, Activation
from keras.regularizers import l2


regularizer = l2(0.0005)

def conv_block(input_depth, output_depth, strides, prev_layer, init, activation, padding):
    x = BatchNormalization()(prev_layer)
    x = Activation(activation)(x)
    # downsample on width increase
    if input_depth != output_depth:
        x = Conv2D(output_depth, 3, strides=(2, 2), padding=padding, kernel_initializer=init,
                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    else:
        x = Conv2D(output_depth, 3, strides=strides, padding=padding, kernel_initializer=init,
                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(output_depth, 3, padding=padding, kernel_initializer=init,
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    return x


def skip_block(input_depth, output_depth, strides, prev_layer, init, padding):
    if input_depth != output_depth:
        prev_layer = Conv2D(output_depth, 1, strides=(2, 2), padding=padding, kernel_initializer=init,
                            kernel_regularizer=regularizer, use_bias=False)(prev_layer)
    return prev_layer


def residual_block(input_depth, output_depth, strides, prev_layer, init, activation, padding):
    conv = conv_block(input_depth, output_depth, strides, prev_layer, init, activation, padding)
    skip = skip_block(input_depth, output_depth, strides, prev_layer, init, padding)
    add = Add()([conv, skip])
    add = Activation(activation)(add)
    return add


def mnist_model(activation='relu', padding='same'):
    model = Sequential()
    model.add(Conv2D(16, 5, activation=activation, padding=padding, input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(Conv2D(32, 5, activation=activation, padding=padding))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dense(10, activation='softmax'))
    return model


def cifar_model(init='he_normal', strides=(1, 1), padding='same', activation='relu'):
    inputs = Input((32, 32, 3))
    conv1 = Conv2D(16, 3, strides=strides, padding=padding, kernel_initializer=init,
                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(inputs)
    res1 = residual_block(16, 64, strides, conv1, init, activation, padding)
    res2 = residual_block(64, 64, strides, res1, init, activation, padding)
    res3 = residual_block(64, 128, strides, res2, init, activation, padding)
    res4 = residual_block(128, 128, strides, res3, init, activation, padding)
    res5 = residual_block(128, 256, strides, res4, init, activation, padding)
    res6 = residual_block(256, 256, strides, res5, init, activation, padding)
    batch = BatchNormalization()(res6)
    act = Activation(activation)(batch)
    pool = GlobalAveragePooling2D()(act)
    outputs = Dense(10, kernel_initializer=init, kernel_regularizer=regularizer, bias_regularizer=regularizer,
                    activation='softmax')(pool)

    return Model(inputs=inputs, outputs=outputs)

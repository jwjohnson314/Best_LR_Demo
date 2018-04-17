import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def preprocess_image_array(arr, dataset):
    if not len(arr.shape) == 4:
        arr = np.expand_dims(arr, -1)
    
    arr = arr.astype('float32') / 255.

    if dataset == 'cifar10':
        mean_pixels = arr.mean(axis=0)
        for i in range(len(arr)):
            arr[i, :, :, :] = arr[i, :, :, :] - mean_pixels
    return arr


def preprocess_label_array(arr):
    return np.array([[1 if arr[i] == j else 0 for j in range(10)] for i in range(len(arr))])


def plot_lr_curve(lrs, train_losses, smoothed_losses, clip=5):
    plt.semilogx(lrs[clip:len(train_losses) - clip], train_losses[clip:-clip], label='train', alpha=0.2, color='red')
    plt.semilogx(lrs[clip:len(smoothed_losses) - clip], smoothed_losses[clip:-clip], color='red')
    plt.xlabel('learning rate (log scale)')
    plt.ylabel('loss')
    plt.ylim(0, np.max(smoothed_losses[clip:-clip]) + 2)
    plt.legend()
    plt.show()
    plt.pause(0.001)
    input('press [enter] to begin training model')


def plot_training_curves(history):    
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    ax[0].plot(history.history['acc'], label='train acc')
    ax[0].plot(history.history['val_acc'], label='val acc')
    ax[1].plot(history.history['loss'], label='train loss')
    ax[1].plot(history.history['val_loss'], label='val loss')
    ax[2].plot(history.history['lr'], label='learning rate')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()


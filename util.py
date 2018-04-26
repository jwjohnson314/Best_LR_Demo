import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from tqdm import tqdm
plt.style.use('bmh')


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
    plt.semilogx(lrs[clip:len(train_losses) - clip], train_losses[clip:-clip],
            label='train', alpha=0.2, color='red')
    plt.semilogx(lrs[clip:len(smoothed_losses) - clip],
            smoothed_losses[clip:-clip], color='red')
    plt.xlabel('learning rate (log scale)')
    plt.ylabel('loss')
    plt.ylim(0, np.max(smoothed_losses[clip:-clip]) + 2)
    plt.legend()
    plt.show()
    plt.pause(0.001)
    input('press [enter] to begin training model')


def plot_training_curves(history):    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(history.history['acc'], label='train acc')
    ax[0].plot(history.history['val_acc'], label='val acc')
    ax[1].plot(history.history['loss'], label='train loss')
    ax[1].plot(history.history['val_loss'], label='val loss')
    ax[0].legend()
    ax[1].legend()
    plt.show()


def find_best_lr(model, xtr, ytr, batch_size, num_learning_batches,
                 batches_per_epoch, lr_min, lr_max, optimizer, no_plots=False):
    # log scale
    q = (lr_max / lr_min)**(1 / num_learning_batches)
    lrs = [(q ** i) * lr_min for i in range(num_learning_batches)]

    beta = 0.98

    train_losses = []
    smoothed_losses = []
    avg_loss = 0.0
    min_smoothed_loss = np.inf

    print('optimizer is {}'.format(optimizer))
    print('finding best learning rate by training \
           for {} minibatches'.format(num_learning_batches))

    # shuffle the data
    idx = list(range(len(xtr)))
    np.random.shuffle(idx)
    xtr = xtr[idx, :, :, :]
    ytr = ytr[idx, :]

    for i in tqdm(range(num_learning_batches)):
        K.set_value(model.optimizer.lr, lrs[i])
        batch_index = i
        if batch_index >= batches_per_epoch:
            batch_index %= batches_per_epoch
        batch_loss, _ = model.train_on_batch(
            xtr[batch_size * batch_index: batch_size * (batch_index + 1), :, :, :],
            ytr[batch_size * batch_index: batch_size * (batch_index + 1), :])

        avg_loss = beta * avg_loss + (1 - beta) * batch_loss    
        train_losses.append(batch_loss)

        # exponential smoothing
        smoothed_loss = avg_loss / (1 - beta**(i + 1))
        if smoothed_loss < min_smoothed_loss:
            min_smoothed_loss = smoothed_loss
        smoothed_losses.append(smoothed_loss)
        if smoothed_loss > 4 * min_smoothed_loss:
            break
       
    print('min_smoothed_loss: {}'.format(min_smoothed_loss))
    print('learning rate at min: {}'.format(
        lrs[smoothed_losses.index(min_smoothed_loss)]))

    if no_plots:
        plot_lr_curve(lrs, train_losses, smoothed_losses, clip=5)

    return lrs[smoothed_losses.index(min_smoothed_loss)]

import argparse
import numpy as np
from keras.optimizers import SGD, adam, RMSprop
from keras.datasets import mnist, cifar10 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import keras.backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import mnist_model, cifar_model
from util import preprocess_image_array, preprocess_label_array, plot_lr_curve, plot_training_curves, get_session 


# allow GPU growth
K.tensorflow_backend.set_session(get_session())

# command line
parser = argparse.ArgumentParser()

# dataset and model parameters
parser.add_argument('--dataset', default='mnist', help='one of mnist, cifar10')
parser.add_argument('--kernel_initializer', default='he_normal', help='kernel initializer, only modifies cifar10 model')
parser.add_argument('--activation', default='relu', help='activation function, only modifies cifar10 model')

# training and learning rate parameters
parser.add_argument('--optimizer', default='sgd', help='one of sgd, adam, or rmsprop')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum, only applied with sgd optimizer')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_learning_batches', default=4000, help='how many batches to determine optimum learning rate?')
parser.add_argument('--num_epochs', default=55, type=int, help='number of epochs to train for')
parser.add_argument('--lr_min', default=1e-4, type=float, help='minimum learning rate to try')
parser.add_argument('--lr_max', default=10, type=float, help='maximum learning rate to try')
parser.add_argument('--lr_multiplier', default=0.1, type=float, help='train at lr_multiplier * <lr with min loss>')

# plotting and saving options
parser.add_argument('--no_plots', action='store_false', help='suppress plotting')
parser.add_argument('--save_model', action='store_true', help='save the model?')

args = parser.parse_args()

print('loading and preprocessing {} data'.format(args.dataset))
if args.dataset == 'mnist':
    (xtr, ytr), (xte, yte) = mnist.load_data()
else:
    (xtr, ytr), (xte, yte) = cifar10.load_data()

xtr = preprocess_image_array(xtr, args.dataset)
xte = preprocess_image_array(xte, args.dataset)
ytr = preprocess_label_array(ytr)
yte = preprocess_label_array(yte)
print('data shape: {}'.format(xtr.shape))
print('label shape: {}'.format(ytr.shape))

print('loading model')
if args.dataset == 'mnist':
    model = mnist_model(activation='relu', padding='same')
else:
    model = cifar_model(init=args.kernel_initializer, activation=args.activation, padding='same')
print(model.summary())

if args.optimizer == 'adam':
    optim = adam()
elif args.optimizer == 'sgd':
    optim=SGD(momentum=args.momentum, nesterov=True)
else:
    optim = RMSprop()

model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])

# log scale
q = (args.lr_max / args.lr_min)**(1 / args.num_learning_batches)
lrs = [(q ** i) * args.lr_min for i in range(args.num_learning_batches)]

beta = 0.98

train_losses = []
smoothed_losses = []
avg_loss = 0.0
min_smoothed_loss = np.inf

print('optimizer is {}'.format(args.optimizer))
print('finding best learning rate by training for {} minibatches'.format(args.num_learning_batches))

# shuffle the data
idx = list(range(len(xtr)))
np.random.shuffle(idx)
xtr = xtr[idx, :, :, :]
ytr = ytr[idx, :]

batches_per_epoch = int(len(xtr) // args.batch_size)
for i in tqdm(range(args.num_learning_batches)):
    K.set_value(model.optimizer.lr, lrs[i])
    batch_index = i
    if batch_index >= batches_per_epoch:
        batch_index %= batches_per_epoch
    batch_loss, _ = model.train_on_batch(xtr[args.batch_size * batch_index: args.batch_size * (batch_index + 1), :, :, :],
                                      ytr[args.batch_size * batch_index: args.batch_size * (batch_index + 1), :])

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
print('learning rate at min: {}'.format(lrs[smoothed_losses.index(min_smoothed_loss)]))

if args.no_plots:
    plot_lr_curve(lrs, train_losses, smoothed_losses, clip=5)

# train at fixed best learning rate
lr = args.lr_multiplier * lrs[smoothed_losses.index(min_smoothed_loss)]
print('Starting training with lr={}'.format(lr))
K.set_value(optim.lr, lr)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])

early = EarlyStopping(monitor='val_loss', patience=10, verbose=True)
rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=True)
hist = model.fit(xtr, ytr, validation_split=0.2, batch_size=args.batch_size, shuffle=True, epochs=args.num_epochs, callbacks=[early, rlr])

if args.no_plots:
    plot_training_curves(hist)

print(model.evaluate(xte, yte))

if args.save_model:
    print('saving model')
    model.save('{}_lr_{}.h5'.format(args.dataset, lr))
print('Done!')

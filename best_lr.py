import argparse
import numpy as np
import os
from keras.optimizers import SGD, adam, RMSprop
from keras.datasets import mnist, cifar10 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import matplotlib.pyplot as plt

from model import mnist_model, cifar_model
from util import preprocess_image_array, preprocess_label_array
from util import find_best_lr, plot_training_curves, get_session 

from clr_callback import CyclicLR

# allow GPU growth
K.tensorflow_backend.set_session(get_session())

# command line
parser = argparse.ArgumentParser()

# dataset and model parameters
parser.add_argument('--dataset', default='mnist', help='one of mnist, cifar10')
parser.add_argument('--kernel_initializer', default='he_normal',
        help='kernel initializer, only modifies cifar10 model')
parser.add_argument('--activation', default='relu',
        help='activation function, only modifies cifar10 model')
parser.add_argument('--weight_decay', default=0.0005, type=float,
        help='weight decay? (only applied to cifar model)')

# training and learning rate parameters
parser.add_argument('--optimizer', default='sgd',
        help='one of sgd, adam, or rmsprop')
parser.add_argument('--momentum', default=0.9, type=float,
        help='momentum, only applied with sgd optimizer')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_learning_batches', default=4000, type=int,
        help='how many batches to determine optimum learning rate?')
parser.add_argument('--num_epochs', default=55, type=int,
        help='number of epochs to train for')
parser.add_argument('--lr_min', default=1e-4, type=float,
        help='minimum learning rate to try')
parser.add_argument('--lr_max', default=10, type=float,
        help='maximum learning rate to try')
parser.add_argument('--lr_max_multiplier', default=0.5, type=float,
        help='lr_max_multiplier * <lr with min loss> is the learning rate \
        when using a fixed learning rate, and the max learning rate when cycling.')
parser.add_argument('--lr_min_multiplier', default=0.0001, type=float,
        help='min learning rate during cycle lr_min_multiplier * <lr with min loss> \
                (only used when cycling learning rate)')
parser.add_argument('--cycle', action='store_true', help='cycle the learning rate?')
parser.add_argument('--skip_test', default=None, type=float,
        help='skip the learning rate test and set a learning rate')

# plotting and saving options
parser.add_argument('--no_plots', action='store_false', help='suppress plotting')
parser.add_argument('--save_model', action='store_true', help='save the model?')
parser.add_argument('--log_dir', default='./logs', help='where to keep logs/save model')
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
    model = cifar_model(init=args.kernel_initializer,
            activation=args.activation,
            weight_decay=args.weight_decay,
            padding='same')
print(model.summary())

if args.optimizer == 'adam':
    optim = adam()
elif args.optimizer == 'sgd':
    optim=SGD(momentum=args.momentum, nesterov=True)
else:
    optim = RMSprop()

model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])
batches_per_epoch = int(len(xtr) // args.batch_size)

if args.skip_test is None:
    best_lr = find_best_lr(model, xtr, ytr, args.batch_size,
                           args.num_learning_batches, batches_per_epoch,
                           args.lr_min, args.lr_max, args.optimizer,
                           args.no_plots)
    lr = args.lr_max_multiplier * best_lr
else:
    lr = args.lr_max_multiplier * args.skip_test


print('Training for {} epochs with {} batches per epoch'.format(
    args.num_epochs, batches_per_epoch))
K.set_value(optim.lr, lr)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])

# make sure we have a log directory
if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)

# instantiate callbacks
early = EarlyStopping(monitor='val_loss', patience=20, verbose=True)
tb = TensorBoard(log_dir=args.log_dir, histogram_freq=0, write_graph=False)
callbacks = [early, tb]

if args.cycle:
    if args.skip_test is not None:
        base_lr = args.lr_min_multiplier * args.skip_test
    else:
        base_lr = args.lr_min_multiplier * best_lr
    step_size = 4 * batches_per_epoch
    cycle = CyclicLR(base_lr=base_lr, max_lr=lr, step_size=step_size)
    callbacks.append(cycle)
    print('Cycling learning rate from {} to {} over {} steps'.format(
        base_lr, lr, step_size))
else:
    print('Starting training with lr={}'.format(lr))
    print('Learning rate will be reduced on plateau')
    rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=True)
    callbacks.append(rlr)

if args.save_model:
    if not os.path.isdir(os.path.join(args.log_dir, 'models')):
        os.mkdir(os.path.join(args.log_dir, 'models'))
    check = ModelCheckpoint(os.path.join(args.log_dir, 'models',
        args.dataset + '_epoch_{epoch:02d}.h5'))
    callbacks.append(check)

hist = model.fit(xtr, ytr, validation_split=0.2, batch_size=args.batch_size,
        shuffle=True, epochs=args.num_epochs, callbacks=callbacks)

if args.no_plots:
    plot_training_curves(hist)

print(model.evaluate(xte, yte))
print('Done!')

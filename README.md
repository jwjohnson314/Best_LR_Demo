# Learning Rate Finder and Cyclic Learning Rate Demo
This is a demo of 
- The method described [here](https://arxiv.org/pdf/1506.01186.pdf) for finding the best learning rate; and
- The use of cyclical learning rates (from the same paper).
It's not a drop-in module to help find a good learning rate for other problems, though the method is transferrable.

# What It Does
Takes a 'warm-up' run of a simple LeNet-style MNIST or a WRN-28-10 Cifar10 model ([https://arxiv.org/abs/1605.07146](https://arxiv.org/abs/1605.07146)), varying the learning rate from a very small number to a very large number, to determine the value at which the loss is as small as possible. Then, either trains the model at a multiple of that learning rate or cycles the learning rate, depending on flags passed at runtime.

# Requirements
In addition to Keras with Tensorflow backend and the usual NumPy/Matplotlib/SciPy stack, you need the custom clr_callback for Keras found [here](https://github.com/bckenstler/CLR/blob/master/clr_callback.py).  

# Usage
Basic usage: `python best_lr.py`

With Cifar10 and some additional flags: `python best_lr.py --dataset=cifar10 --num_epochs=100 --lr_min=1e-6 --cycle --save_model`

# Options

## Data options and some model parameters

Both the MNIST and the Cifar10 models are pretty much hardcoded, but options to change the kernel initializer, the activation function, and the amount of weight decay for the Cifar10 model are provided.

- `--dataset`: default=`'mnist'`, one of `'mnist', 'cifar10'`
- `--kernel_initializer`: default is `'he_normal'`, only modifies the cifar10 model
- `--activation`: default is `'relu'`, only modifies the cifar10 model
- `--weight_decay`: default is `0.0005`, only modifies cifar10 model

## Training and learning rate options
- `--optimizer`: one of adam, rmsprop, or sgd. Default is sgd
- `--momentum`: default is `0.9`. Only used with sgd
- `--batch_size`': default is `32`
- `--num_epochs`: default is `55`
- `--num_learning_batches`: How many batches to vary the learning rate over to determine the best learning rate? default is `4000`
- `--lr_min`: default is `1e-4`, minimum learning rate to try when determining best learning rate
- `--lr_max`: default is `10`, maximum learning rate to try when determining best learning rate
- `--lr_max_multiplier`: when training with fixed learning rate, that rate will be `lr_max_multiplier * <lr-with-min-loss>`. When cycling learning rate, this will be the maximum learning rate. Default is `1`.
- `--lr_min_multiplier`: When cycling the learning rate, the minimum learning rate will be `lr_min_multiplier * <lr-with-min-loss>`. Default is `0.1`.
- `--cycle`: Cycle the learning rate? Default is false
- `--skip_test`: Want to skip the learning rate test and just get on with training? Specify learning rate here. This learning rate will be treated as `<lr-with-max-loss>` (See `lr_max_multiplier` and `lr_min_multiplier` above). Should be a float and default is `None`.

## Plotting and saving options
- `--no_plots`: Suppress plot generation?
- `--save_model`: Default is false
- `--log_dir`: Where to save tensorboard logs/model weights. default is `'./logs'`

# Performance
On Cifar10, with learning rate cycling, batch size 128, and all other parameters as default will typically get a test accuracy in the high 80's or low 90's in 20-30 epochs. On MNIST, batch size 64, cycling the learning rate, typically acheives greater than 99% test accuracy in under 10 epochs.


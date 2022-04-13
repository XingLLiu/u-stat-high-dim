import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tfkd = tf.keras.datasets

import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import os
import random

# from data.dataset_loader import load_and_preprocess_mnist, inverse_logit
# from normalizingflows.flow_catalog import Made, BatchNorm, get_trainable_variables
# from utils.train_utils import train_density_estimation, nll

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


def logit(z, beta=10e-6):
    """
    Conversion to logit space according to equation (24) in [Papamakarios et al. (2017)].
    Includes scaling the input image to [0, 1] and conversion to logit space.
    :param z: Input tensor, e.g. image. Type: tf.float32.
    :param beta: Small value. Default: 10e-6.
    :return: Input tensor in logit space.
    """

    inter = beta + (1 - 2 * beta) * (z / 256)
    return tf.math.log(inter/(1-inter))  # logit function


def inverse_logit(x, beta=10e-6):
    """
    Reverts the preprocessing steps and conversion to logit space and outputs an image in
    range [0, 256]. Inverse of equation (24) in [Papamakarios et al. (2017)].
    :param x: Input tensor in logit space. Type: tf.float32.
    :param beta: Small value. Default: 10e-6.
    :return: Input tensor in logit space.
    """

    x = tf.math.sigmoid(x)
    return (x-beta)*256 / (1 - 2*beta)

def load_and_preprocess_mnist(logit_space=True, batch_size=128, shuffle=True, classes=-1, channels=False):
    """
     Loads and preprocesses the MNIST dataset. Train set: 50000, val set: 10000,
     test set: 10000.
    :param logit_space: If True, the data is converted to logit space.
    :param batch_size: batch size
    :param shuffle: bool. If True, dataset will be shuffled.
    :param classes: int of class to take, defaults to -1 = ALL
    :return: Three batched TensorFlow datasets:
    batched_train_data, batched_val_data, batched_test_data.
    """

    (x_train, y_train), (x_test, y_test) = tfkd.mnist.load_data()

    # reserve last 10000 training samples as validation set
    x_train, x_val = x_train[:-10000], x_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # if logit_space: convert to logit space, else: scale to [0, 1]
    if logit_space:
        x_train = logit(tf.cast(x_train, tf.float32))
        x_test = logit(tf.cast(x_test, tf.float32))
        x_val = logit(tf.cast(x_val, tf.float32))
        interval = 256
    else:
        x_train = tf.cast(x_train / 256, tf.float32)
        x_test = tf.cast(x_test / 256, tf.float32)
        x_val = tf.cast(x_val / 256, tf.float32)
        interval = 1


    if classes == -1:
        pass
    else:
        #TODO: Extract Multiple classes: How to to the train,val split,
        # Do we need to to a class balance???
        x_train = np.take(x_train, tf.where(np.isin(y_train, classes)), axis=0)
        x_val = np.take(x_val, tf.where(np.isin(y_val, classes)), axis=0)
        x_test = np.take(x_test, tf.where(np.isin(y_test, classes)), axis=0)
        
        for name, arr in zip(["train", "val", "test"], [y_train, y_val, y_test]):
          print(f"num in classes in {name}:", 
                [(classes[i], np.sum(arr == classes[i])) for i in classes])

    # reshape if necessary
    if channels:
        x_train = tf.reshape(x_train, (x_train.shape[0], 28, 28, 1))
        x_val = tf.reshape(x_val, (x_val.shape[0], 28, 28, 1))
        x_test = tf.reshape(x_test, (x_test.shape[0], 28, 28, 1))
    else:
        x_train = tf.reshape(x_train, (x_train.shape[0], 28, 28))
        x_val = tf.reshape(x_val, (x_val.shape[0], 28, 28))
        x_test = tf.reshape(x_test, (x_test.shape[0], 28, 28))

    if shuffle:
        shuffled_train_data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)

    batched_train_data = shuffled_train_data.batch(batch_size)
    batched_val_data = tf.data.Dataset.from_tensor_slices(x_val).batch(batch_size)
    batched_test_data = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)    
    
    return batched_train_data, batched_val_data, batched_test_data, interval


class BatchNorm(tfb.Bijector):
    """
    Implementation of a Batch Normalization layer for use in normalizing flows according to [Papamakarios et al. (2017)].
    The moving average of the layer statistics is adapted from [Dinh et al. (2016)].
    :param eps: Hyperparameter that ensures numerical stability, if any of the elements of v is near zero.
    :param decay: Weight for the update of the moving average, e.g. avg = (1-decay)*avg + decay*new_value.
    """

    def __init__(self, eps=1e-5, decay=0.95, validate_args=False, name="batch_norm"):
        super(BatchNorm, self).__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            validate_args=validate_args,
            name=name)

        self._vars_created = False
        self.eps = eps
        self.decay = decay

    def _create_vars(self, x):
        # account for 1xd and dx1 vectors
        if len(x.get_shape()) == 1:
            n = x.get_shape().as_list()[0]
        if len(x.get_shape()) == 2: 
            n = x.get_shape().as_list()[1]

        self.beta = tf.compat.v1.get_variable('beta', [1, n], dtype=tf.float32)
        self.gamma = tf.compat.v1.get_variable('gamma', [1, n], dtype=tf.float32)
        self.train_m = tf.compat.v1.get_variable(
            'mean', [1, n], dtype=tf.float32, trainable=False)
        self.train_v = tf.compat.v1.get_variable(
            'var', [1, n], dtype=tf.float32, trainable=False)

        self._vars_created = True

    def _forward(self, u):
        if not self._vars_created:
            self._create_vars(u)
        return (u - self.beta) * tf.exp(-self.gamma) * tf.sqrt(self.train_v + self.eps) + self.train_m

    def _inverse(self, x):
        # Eq. 22 of [Papamakarios et al. (2017)]. Called during training of a normalizing flow.
        if not self._vars_created:
            self._create_vars(x)

        # statistics of current minibatch
        m, v = tf.nn.moments(x, axes=[0], keepdims=True)
        
        # update train statistics via exponential moving average
        self.train_v.assign_sub(self.decay * (self.train_v - v))
        self.train_m.assign_sub(self.decay * (self.train_m - m))

        # normalize using current minibatch statistics, followed by BN scale and shift
        return (x - m) * 1. / tf.sqrt(v + self.eps) * tf.exp(self.gamma) + self.beta

    def _inverse_log_det_jacobian(self, x):
        # at training time, the log_det_jacobian is computed from statistics of the
        # current minibatch.
        if not self._vars_created:
            self._create_vars(x)
            
        _, v = tf.nn.moments(x, axes=[0], keepdims=True)
        abs_log_det_J_inv = tf.reduce_sum(
            self.gamma - .5 * tf.math.log(v + self.eps))
        return abs_log_det_J_inv

    
class Made(tfk.layers.Layer):
    """
    Implementation of a Masked Autoencoder for Distribution Estimation (MADE) [Germain et al. (2015)].
    The existing TensorFlow bijector "AutoregressiveNetwork" is used. The output is reshaped to output one shift vector
    and one log_scale vector.
    :param params: Python integer specifying the number of parameters to output per input.
    :param event_shape: Python list-like of positive integers (or a single int), specifying the shape of the input to this layer, which is also the event_shape of the distribution parameterized by this layer. Currently only rank-1 shapes are supported. That is, event_shape must be a single integer. If not specified, the event shape is inferred when this layer is first called or built.
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: An activation function. See tf.keras.layers.Dense. Default: None.
    :param use_bias: Whether or not the dense layers constructed in this layer should have a bias term. See tf.keras.layers.Dense. Default: True.
    :param kernel_regularizer: Regularizer function applied to the Dense kernel weight matrices. Default: None.
    :param bias_regularizer: Regularizer function applied to the Dense bias weight vectors. Default: None.
    """

    def __init__(self, params, event_shape=None, hidden_units=None, activation=None, use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, name="made"):

        super(Made, self).__init__(name=name)

        self.params = params
        self.event_shape = event_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.network = tfb.AutoregressiveNetwork(params=params, event_shape=event_shape, hidden_units=hidden_units,
                                                 activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer, 
                                                 bias_regularizer=bias_regularizer)

    def call(self, x):
        shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)


def get_trainable_variables(flow):
    """
    Returns the number of trainable variables/weights of a flow.
    :param flow: A normalizing flow in the form of a TensorFlow Transformed Distribution.
    :return: n_trainable_variables
    """
    # number of trainable variables
    n_trainable_variables = 0
    for weights in flow.trainable_variables:
        n_trainable_variables = n_trainable_variables + np.prod(weights.shape)

    return n_trainable_variables


@tf.function
def nll(distribution, data):
    """
    Computes the negative log liklihood loss for a given distribution and given data.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param data: Data or a batch from data.
    :return: Negative Log Likelihodd loss.
    """
    return -tf.reduce_mean(distribution.log_prob(data))


'''--------------------------------------------- Train function -----------------------------------------------------'''

@tf.function
def train_density_estimation(distribution, optimizer, batch):
    """
    Train function for density estimation normalizing flows.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    with tf.GradientTape() as tape:
        tape.watch(distribution.trainable_variables)
        loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
    gradients = tape.gradient(loss, distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))

    return loss


def show(xx):
    for i in range(xx.shape[0]):
        plt.figure()
        data = xx[i]
        data = inverse_logit(data)
        plt.imshow(data, cmap='gray')

def init_model():
    tf.random.set_seed(1234)

    # parameters
    batch_size = 128
    dataset = "mnist"
    layers = 8
    base_lr = 1e-3
    end_lr = 1e-4
    max_epochs = 700 # 300 # 700 is enough
    shape = [256, 256]
    exp_number = 1
    mnist_trainsize = 50000

    # define if training should happen on all classes or one specific class
    # possibilities: [0, 1], [1, 5, 8, 9], -1
    category = [0, 1]

    # train: 50000, validation: 10000, test: 10000
    batched_train_data, batched_val_data, batched_test_data, _ = load_and_preprocess_mnist(
        logit_space=True, batch_size=batch_size, classes=category)

    # get shape of images
    sample_batch = next(iter(batched_train_data))
    if sample_batch.shape[-1] == sample_batch.shape[-2]:
        size = sample_batch.shape[-1]
        input_shape = size*size
    else:
        print("Height and width of input data are not equal!")
        
    permutation = tf.cast(np.concatenate((np.arange(input_shape/2,input_shape),np.arange(0,input_shape/2))), tf.int32)
    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(shape=input_shape, dtype=tf.float32))
    
    # initialise MAF model
    bijectors = []
    event_shape = [size*size]

    # According to [Papamakarios et al. (2017)]:
    # BatchNorm between the last autoregressive layer and the base distribution, and every two autoregressive layers

    bijectors.append(BatchNorm(eps=10e-5, decay=0.95))

    for i in range(0, layers):

        bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=shape, activation="relu")))
        bijectors.append(tfb.Permute(permutation=permutation)) # Permutation improves denstiy estimation results
        
        # add BatchNorm every two layers
        if (i+1) % int(2) == 0: 
            bijectors.append(BatchNorm(eps=10e-5, decay=0.95))


    bijectors.append(tfb.Reshape(event_shape_out=(size,size), event_shape_in=(size*size,))) # reshape array to image shape, before: (size*size,)

    bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='chain_of_maf')


    maf = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=bijector
        # event_shape=[event_shape]
    )

    # important: initialize with log_prob to initialize the moving average of the layer statistics in the batch norm layers
    maf.log_prob(sample_batch)  # initialize
    print("Successfully initialized!")

    # learning rate scheduling
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)

    # initialize checkpoints
    checkpoint_directory = "res/tmp_{}_{}".format(layers, shape[0])
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=maf)
    # return checkpoint, maf

    # load best model with min validation loss
    checkpoint.restore(checkpoint_prefix)
    print("Successfully loaded trained model!")

    return checkpoint, maf, batched_train_data


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions

def resample(log_probs, x):
    log_probs = tf.reshape(log_probs, (-1, ))
    # print(np.exp(log_probs))
    ind = tfd.Categorical(logits=log_probs).sample(log_probs.shape[0])
    ind = tf.reshape(ind, (-1, 1))
    # print(ind)
    return tf.gather_nd(x, ind)
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions

def resample(log_probs, x):
    log_probs = tf.reshape(log_probs, (-1,))
    ind = tfd.Categorical(logits=log_probs).sample(log_probs.shape[0])
    ind = tf.reshape(ind, (-1, 1))
    return tf.gather_nd(x, ind)

def effective_sample_size(log_weights):
    # scale for numerical stability
    log_weights -= tf.reduce_max(log_weights)
    weights = tf.exp(log_weights)
    ess = tf.reduce_sum(weights) ** 2 / tf.reduce_sum(weights ** 2)
    return ess

def weighted_sum(f, x, log_w=None):
    """Compute \sum_i w_i * f(x_i) / \sum_i w_i
    """
    y = f(x) # n x 1
    if log_w is not None:
        assert y.shape == log_w.shape, "f(x) should have shape (n, 1)"
        w = tf.exp(log_w)
    else:
        w = tf.ones_like(y)
    total_w = tf.reduce_sum(w)
    return tf.reduce_sum(y * w) / total_w

class IS:
    def __init__(self, p_prior, p_target):
        self.logp_prior = logp_prior
        self.logp_target = logp_target
    def compute_log_weights(x):
        return self.logp_target(x) - self.logp_prior(x)

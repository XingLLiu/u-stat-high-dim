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
    weights = tf.exp(log_weights)
    ess = 1 / tf.reduce_sum(weights ** 2)
    return ess

class IS:
    def __init__(self, p_prior, p_target):
        self.logp_prior = logp_prior
        self.logp_target = logp_target
    def compute_log_weights(x):
        return self.logp_target(x) - self.logp_prior(x)

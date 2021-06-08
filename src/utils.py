import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def resample(log_probs, x):
    ind = tfd.Categorical(logits=tf.transpose(log_probs)).sample(log_probs.shape[0])
    return tf.gather_nd(x, ind)
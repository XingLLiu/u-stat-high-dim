import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def create_mixture_gaussian(dim, delta, ratio=0.5, return_logprob=False):
    """Bimodal Gaussian mixture with mean shift only in the first dim"""
    e1 = tf.eye(dim)[:, 0]
    mean1 = -delta * e1
    mean2 = delta * e1
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[ratio, 1-ratio]),
      components=[
        tfd.MultivariateNormalDiag(mean1),
        tfd.MultivariateNormalDiag(mean2)
    ])

    if not return_logprob:
      return mix_gauss
    else:      
      def log_prob_fn(x):
        '''fast implementation of log_prob'''
        exp1 = tf.reduce_sum((x - mean1)**2, axis=-1) # n
        exp2 = tf.reduce_sum((x - mean2)**2, axis=-1) # n
        return tf.math.log(
            ratio * tf.math.exp(- 0.5 * exp1) + (1-ratio) * tf.math.exp(- 0.5 * exp2)
        )
      
      return mix_gauss, log_prob_fn


def create_mixture_gaussian_kdim(dim, k, delta, ratio=0.5, return_logprob=False):
    """Bimodal Gaussian mixture with mean shift of dist delta in the first k dims"""
    a = [1. if x < k else 0. for x in range(dim)]
    a = tf.constant(a)
    multiplier = delta/tf.math.sqrt(float(k))
    mean1 = -multiplier * a
    mean2 = multiplier * a
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[ratio, 1-ratio]),
      components=[
        tfd.MultivariateNormalDiag(mean1),
        tfd.MultivariateNormalDiag(mean2)
    ])
    
    if not return_logprob:
      return mix_gauss
    else:      
      def log_prob_fn(x):
        '''fast implementation of log_prob'''
        exp1 = tf.reduce_sum((x - mean1)**2, axis=-1) # n
        exp2 = tf.reduce_sum((x - mean2)**2, axis=-1) # n
        return tf.math.log(
            ratio * tf.math.exp(- 0.5 * exp1) + (1-ratio) * tf.math.exp(- 0.5 * exp2)
        )
      
      return mix_gauss, log_prob_fn  




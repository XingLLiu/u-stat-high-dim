import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def create_mixture_gaussian(dim, delta, ratio=0.5, return_logprob=False):
    """Bimodal Gaussian mixture with mean shift only in the first dim"""
    e1 = tf.eye(dim)[:, 0]
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[ratio, 1-ratio]),
      components=[
        tfd.MultivariateNormalDiag(-delta * e1),
        tfd.MultivariateNormalDiag(delta * e1)
    ])

    if not return_logprob:
      return mix_gauss
    else:      
      def log_prob_fn(x):
        '''fast implementation of log_prob'''
        exp1 = tf.reduce_sum((x - delta * e1)**2, axis=-1) # n
        exp2 = tf.reduce_sum((x + delta * e1)**2, axis=-1) # n
        return tf.math.log(
            tf.math.exp(- 0.5 * exp1) + tf.math.exp(- 0.5 * exp2)
        )
      
      return mix_gauss, log_prob_fn


def create_mixture_gaussian_kdim(dim, k, delta, ratio=0.5):
    """Bimodal Gaussian mixture with mean shift of dist delta in the first k dims"""
    a = [1. if x < k else 0. for x in range(dim)]
    a = tf.constant(a)
    multiplier = delta/tf.math.sqrt(float(k))
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[ratio, 1-ratio]),
      components=[
        tfd.MultivariateNormalDiag(- multiplier * a),
        tfd.MultivariateNormalDiag(multiplier * a)
    ])
    single_component = tfd.MultivariateNormalDiag(- multiplier * a)
    return mix_gauss, single_component    




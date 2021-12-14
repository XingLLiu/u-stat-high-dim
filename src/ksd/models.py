import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def create_mixture_gaussian(dim, delta, ratio=0.5):
    """Bimodal Gaussian mixture with mean shift only in the first dim"""
    e1 = tf.eye(dim)[:, 0]
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[ratio, 1-ratio]),
      components=[
        tfd.MultivariateNormalDiag(-delta * e1),
        tfd.MultivariateNormalDiag(delta * e1)
    ])
    return mix_gauss    





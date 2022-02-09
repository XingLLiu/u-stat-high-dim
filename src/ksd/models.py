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
            ratio * tf.math.exp(- 0.5 * exp1) + 
            (1-ratio) * tf.math.exp(- 0.5 * exp2)
        )
      
      return mix_gauss, log_prob_fn  


class Banana(tfp.bijectors.Bijector):
  """Bijector class for banana distributions"""
  def __init__(self, b=0.03, name="banana"):
    super(Banana, self).__init__(inverse_min_event_ndims=1,
                                  is_constant_jacobian=True,
                                  name=name)
    self.b = b

  def _forward(self, x):
    y_0 = x[..., 0:1]
    y_1 = x[..., 1:2] + self.b * y_0**2 - 100 * self.b
    y_tail = x[..., 2:]

    return tf.concat([y_0, y_1, y_tail], axis=-1)

  def _inverse(self, y):
    x_0 = y[..., 0:1]
    x_1 = y[..., 1:2] - self.b * x_0**2 + 100 * self.b
    x_tail = y[..., 2:]

    return tf.concat([x_0, x_1, x_tail], axis=-1)

  def _inverse_log_det_jacobian(self, y):
    return tf.zeros(shape=())

def create_banana(dim: int, loc: tf.Tensor, **kwargs):
  id_mat = tf.eye(dim)
  scale_mat = tf.concat([id_mat[:, :1] * 10, id_mat[:, 1:]], axis=-1)
  t_dist = tfd.MultivariateStudentTLinearOperator(
    df=7, loc=loc, scale=tf.linalg.LinearOperatorLowerTriangular(scale_mat))
  banana = tfd.TransformedDistribution(distribution=t_dist, bijector=Banana(**kwargs))
  return banana

def create_mixture_t_banana(dim: int, ratio: tf.Tensor, loc: tf.Tensor, return_logprob=False, **kwargs):
  """Create a mixture of t and t-tailed banana distributions. The first 5 modes are
  banana distributions, and the rest are t distributions.
  
  ratio: mixture proportions. Must have length >= 5
  loc: location vectors of shape (len(ratio), dim). The first 5 rows are the loc vectors for the
    banana distributions, and the rest are for the t-distributions
  """
  nmodes = len(ratio)
  assert nmodes >= 5, "number of mixtures must be >= 5"

  banana_component = [create_banana(dim, loc=loc[i, :], **kwargs) for i in range(5)]
  cov_mat = tf.math.sqrt(0.01 * tf.math.sqrt(float(dim)) * tf.eye(dim))
  t_component = [
      tfd.MultivariateStudentTLinearOperator(
        df=7, 
        loc=loc[5+i, :], 
        scale=tf.linalg.LinearOperatorLowerTriangular(cov_mat)
      ) for i in range(nmodes-5)
    ]
  mixture_dist = tfd.Mixture(
      cat=tfd.Categorical(probs=ratio),
      components=banana_component + t_component)
  
  if not return_logprob:
    return mixture_dist
  else:
    def log_prob(**kwargs):
      return mixture_dist.log_prob(**kwargs)
    return mixture_dist, mixture_dist.log_prob


def create_mixture_20_gaussian(means, ratio=0.5, scale=0.1, return_logprob=False):
    """Mixture of 20 Gaussian
    Args:
      means: Must have shape nmodes x dim
      ratio: Must either be a float or have shape (nmodes,)
    """
    nmodes = means.shape[0]
    ratio = [0.5] * nmodes if isinstance(ratio, float) else ratio
    components = [tfd.MultivariateNormalDiag(
      loc=means[i, :], 
      scale_identity_multiplier=scale) for i in range(means.shape[0])]
    
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=ratio),
      components=components)
    
    if not return_logprob:
      return mix_gauss
    else:
      ratio_expand = tf.expand_dims(ratio, axis=1) # nmodes x 1
      means_expand = tf.expand_dims(means, axis=1) # nmodes x 1 x dim
      def log_prob_fn(x):
        '''fast implementation of log_prob'''
        diff = tf.expand_dims(x, axis=0) - means_expand # nmodes x n x dim
        diff_norm_sq = tf.reduce_sum(diff**2, axis=-1) # nmodes x n
        p_component = ratio_expand * tf.math.exp(- 0.5 * diff_norm_sq / (scale**2)) # nmodes x n
        sum_p = tf.reduce_sum(p_component, axis=0) # n
        return tf.math.log(sum_p)
      
      return mix_gauss, log_prob_fn  

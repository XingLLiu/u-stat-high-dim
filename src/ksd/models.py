import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import kgof.density as density
import numpy as np

def check_log_prob(dist, log_prob):
  """Check if log_prob == dist.log_prob + const."""
  x = dist.sample(10)
  diff = dist.log_prob(x) - log_prob(x)
  res = tf.experimental.numpy.allclose(diff, diff[0])
  assert res, "log_prob function is not implemented correctly"

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
        """fast implementation of log_prob"""
        exp1 = tf.reduce_sum((x - mean1)**2, axis=-1) # n
        exp2 = tf.reduce_sum((x - mean2)**2, axis=-1) # n
        return tf.math.log(
            ratio * tf.math.exp(- 0.5 * exp1) + (1-ratio) * tf.math.exp(- 0.5 * exp2)
        )
      
      return mix_gauss, log_prob_fn


def create_mixture_gaussian_kdim(dim, k, delta, ratio=0.5, shift=0., return_logprob=False):
    """Bimodal Gaussian mixture with mean shift of dist delta in the first k dims"""
    a = [1. if x < k else 0. for x in range(dim)]
    a = tf.constant(a)
    multiplier = delta/tf.math.sqrt(float(k))
    mean1 = -multiplier * a + shift
    mean2 = multiplier * a + shift
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
        """fast implementation of log_prob"""
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

def create_banana(dim: int, loc: tf.Tensor, scale: float=10., **kwargs):
  id_mat = tf.eye(dim)
  scale_mat = tf.concat([id_mat[:, :1] * scale, id_mat[:, 1:]], axis=-1)
  t_dist = tfd.MultivariateStudentTLinearOperator(
    df=7, loc=loc, scale=tf.linalg.LinearOperatorLowerTriangular(scale_mat))
  banana = tfd.TransformedDistribution(distribution=t_dist, bijector=Banana(**kwargs))
  return banana

def t_prob_fast(x, loc, sigma_inv, df=7):
  """Fast implementation of log_prob for t-dist"""
  dim = x.shape[-1]
  y = tf.matmul(x - loc, sigma_inv, transpose_b=True) # n x dim
  y_norm_sq = tf.einsum(
      "ij,ij->i",
      y,
      x - loc) # n
  factor = -0.5 * (df + dim)
  prob = (1 + y_norm_sq / df)**factor # n
  return prob

def banana_prob_fast(x, loc, b, sigma_inv):
  """Fast implementation of log_prob for banana dist"""
  y0 = x[..., 0:1]
  y = tf.Variable(x)
  y[..., 1:2].assign(x[..., 1:2] - b * y0**2 + 100 * b)
  return t_prob_fast(y, loc=loc, sigma_inv=sigma_inv)

def create_mixture_t_banana_fast(dim, nbanana, locs, ratio, cov_mat_t, b=0.03, scale=10.):
  """Fast implementation of log_prob for banana-t model"""
  nmodes = len(ratio)
  id_mat = tf.eye(dim)
  scale_mat = tf.concat([id_mat[:, :1] * scale, id_mat[:, 1:]], axis=-1)
  sigma_inv = tf.linalg.inv(tf.matmul(scale_mat, scale_mat, transpose_b=True))
  
  sigma_inv_t = tf.linalg.inv(tf.matmul(cov_mat_t, cov_mat_t, transpose_b=True))
  
  normalizer_ratio = tf.math.sqrt(
      tf.linalg.det(sigma_inv_t) / tf.linalg.det(sigma_inv))

  def log_prob_mix_fast(x):
      prob_banana_mix = 0.
      for i in range(nbanana):
          prob_banana_mix += ratio[i] * banana_prob_fast(
            x, locs[i, :], b, sigma_inv)

      prob_t_mix = 0.
      for i in range(nmodes-nbanana):
          prob_t_mix += ratio[nbanana+i] * t_prob_fast(x, locs[nbanana+i], sigma_inv_t)

      log_prob = tf.math.log(prob_banana_mix + prob_t_mix * normalizer_ratio)
      return log_prob
  
  return log_prob_mix_fast

def create_mixture_t_banana(dim: int, ratio: tf.Tensor, loc: tf.Tensor, 
  return_logprob=False, nbanana: int=5, **kwargs):
  """Create a mixture of t and t-tailed banana distributions. The first 5 modes are
  banana distributions, and the rest are t distributions.
  
  ratio: mixture proportions. Must have length >= nbanana
  loc: location vectors of shape (len(ratio), dim). The first 5 rows are the loc vectors for the
    banana distributions, and the rest are for the t-distributions
  nbanana: number of banana distributions in the mixture
  """
  nmodes = len(ratio)
  assert nmodes >= nbanana, f"number of mixtures {nmodes} must be >= {nbanana}"

  banana_component = [create_banana(dim, loc=loc[i, :], **kwargs) for i in range(nbanana)]
  cov_mat = tf.math.sqrt(0.01 * tf.math.sqrt(float(dim)) * tf.eye(dim))
  t_component = [
      tfd.MultivariateStudentTLinearOperator(
        df=7, 
        loc=loc[nbanana+i, :], 
        scale=tf.linalg.LinearOperatorLowerTriangular(cov_mat)
      ) for i in range(nmodes-nbanana)
    ]
  mixture_dist = tfd.Mixture(
      cat=tfd.Categorical(probs=ratio),
      components=banana_component + t_component)
  
  if not return_logprob:
    return mixture_dist
  else:
    # log_prob = create_mixture_t_banana_fast(
    #   dim=dim, nbanana=nbanana, locs=loc, ratio=ratio, cov_mat_t=cov_mat, **kwargs)
    return mixture_dist, mixture_dist.log_prob


def create_mixture_20_gaussian(means, ratio=0.5, scale=0.1, return_logprob=False):
    """Mixture of 20 Gaussian
    Args:
      means: Must have shape nmodes x dim
      ratio: Must either be a float or have shape (nmodes,)
    """
    nmodes = means.shape[0]
    ratio = [1/nmodes] * nmodes if isinstance(ratio, float) else ratio
    components = [tfd.MultivariateNormalDiag(
      loc=means[i, :], 
      scale_identity_multiplier=scale) for i in range(nmodes)]
    
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=ratio),
      components=components)
    
    if not return_logprob:
      return mix_gauss
    else:
      ratio_expand = tf.expand_dims(ratio, axis=1) # nmodes x 1
      means_expand = tf.expand_dims(means, axis=1) # nmodes x 1 x dim
      def log_prob_fn(x):
        """fast implementation of log_prob"""
        diff = tf.expand_dims(x, axis=0) - means_expand # nmodes x n x dim
        diff_norm_sq = tf.reduce_sum(diff**2, axis=-1) # nmodes x n
        p_component = ratio_expand * (
            # (scale**2 * tf.experimental.numpy.pi * 2)**(-0.5)
            1/scale
          ) * tf.math.exp(- 0.5 * diff_norm_sq / (scale**2)) # nmodes x n
        sum_p = tf.reduce_sum(p_component, axis=0) # n
        return tf.math.log(sum_p)
      
      return mix_gauss, log_prob_fn  

def create_mixture_gaussian_scaled(ratio=0.5, return_logprob=False):
    """Bimodal Gaussian mixture with mean shift of dist delta in the first k dims"""
    mean1 = [4., 1.]
    mean2 = [-4., -4.]
    # mean1 = [10.]
    # mean2 = [-10.]

    cov1 = tf.constant([[1., 0.8], [0.8, 1]])
    cov2 = tf.constant([[1., -0.8], [-0.8, 1.]]) * 3.
    # cov1 = tf.constant([[1.]])
    # cov2 = tf.constant([[4.]])

    scale1 = tf.linalg.cholesky(cov1)
    scale2 = tf.linalg.cholesky(cov2)

    dist1 = tfd.MultivariateNormalTriL(mean1, scale_tril=scale1)
    dist2 = tfd.MultivariateNormalTriL(mean2, scale_tril=scale2)
    
    mix_gauss = tfd.Mixture(
      cat=tfd.Categorical(probs=[ratio, 1-ratio]),
      components=[dist1, dist2])
    
    if not return_logprob:
      return mix_gauss
    else:
      cov1_inv = tf.linalg.inv(cov1) # dim x dim
      cov2_inv = tf.linalg.inv(cov2) # dim x dim
      det1 = tf.linalg.det(cov1)
      det2 = tf.linalg.det(cov2)
      def log_prob_fn(x):
        """fast implementation of log_prob"""
        exp1 = tf.einsum("ij, ij -> i", x-mean1, (x-mean1)@cov1_inv) # n
        exp2 = tf.einsum("ij, ij -> i", x-mean2, (x-mean2)@cov2_inv) # n
        return tf.math.log(
            ratio * (1/tf.math.sqrt(det1)) * tf.math.exp(- 0.5 * exp1) + 
            (1-ratio) * (1/tf.math.sqrt(det2)) * tf.math.exp(- 0.5 * exp2)
        )
      
      return mix_gauss, log_prob_fn  

def create_rbm(
  seed,
  c_loc,
  dx=50,
  dh=40,
  burnin_number=2000,
  return_logprob=False):
  """
  Generate data for the Gaussian-Bernoulli Restricted Boltzmann Machine (RBM) experiment.
  The entries of the matrix B are perturbed.
  This experiment was first proposed by Liu et al., 2016 (Section 6)
  inputs: seed: non-negative integer
          m: number of samples
          sigma: standard deviation of Gaussian noise
          dx: dimension of observed output variable
          dh: dimension of binary latent variable
          burnin_number: number of burn-in iterations for Gibbs sampler
  outputs: 2-tuple consisting of
          (m,dx) array of samples generated using the perturbed RBM
          (m,dx) array of scores computed using the non-perturbed RBM (model)
  """
  # the perturbed model is fixed, randomness comes from sampling
  tf.random.set_seed(seed)

  # Model p
  B = tf.cast(tf.experimental.numpy.random.randint(0, 2, (dx, dh)), dtype=tf.float32) * 6. - 3.
  b = tf.cast(tf.experimental.numpy.random.randn(dx), dtype=tf.float32)
  c = tf.cast(tf.experimental.numpy.random.randn(dh), dtype=tf.float32) + c_loc
  dist = density.GaussBernRBM(B, b, c)
  dist.event_shape = [dx]
  dist.log_prob = dist.log_den

  ds = dist.get_datasource()
  ds.burnin = burnin_number
  dist.sample = lambda shape: tf.cast(ds.sample(shape).data(), dtype=tf.float32) #TODO not setting seed!

  if not return_logprob:
    return dist
  else:
    return dist, dist.log_prob 

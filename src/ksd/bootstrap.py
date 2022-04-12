import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import trange

class Bootstrap:
  def __init__(self, ksd, n):
    """
    Inputs:
      ksd: instance of KSD class
      n: number of samples
    """
    self.ksd = ksd
    self.ksd_hat = None
    self.ksd_star = None
    self.alpha = None
    self.multinom = tfp.distributions.Multinomial(float(n), probs=tf.ones(n)/n)
  
  def sample_multinomial(self, n: int, num_boot: int):
    """
    Inputs:
      n: number of samples
      num_boot: number of bootstrap samples to be drawn
    
    Returns:
      w \sim 1/n * Multi(n; 1/n, \ldots, 1/n)
    """
    # n = float(n)
    w = self.multinom.sample(num_boot) # num_boot x n
    return w

  def compute_test_statistic(self, X: tf.Tensor, **kwargs):
    n = X.shape[0]

    # compute u_p(xi, xj) using the U-statistic (required for goodness-of-fit tests)
    if "log_noise_std" in kwargs and "u" not in kwargs:
      u_p = self.ksd.eval(X=X, Y=tf.identity(X), output_dim=2, **kwargs).numpy() # n x n
    elif "log_noise_std" in kwargs and "u" in kwargs:
      u_p = self.ksd.eval_mat(X=X, Y=tf.identity(X), output_dim=2, **kwargs).numpy() # n x n
    else:
      u_p = self.ksd(X=X, Y=tf.identity(X), output_dim=2, **kwargs).numpy() # n x n

    u_p_nodiag = u_p - tf.linalg.diag(tf.linalg.diag_part(u_p)) # n x n

    # compute test statistic
    self.ksd_hat = tf.reduce_sum(u_p_nodiag).numpy() / (n*(n - 1))

    return u_p

  def compute_bootstrap(
    self, 
    num_boot: int, 
    u_p: tf.Tensor,
    multinom_samples: tf.Tensor=None
  ):
    """
    Inputs:
      u_p: u_p(xi, xj). Shape: (n, n)
      ksd_hat: estimated KSD
    """
    n = u_p.shape[0]
    u_p = tf.expand_dims(u_p, axis=0) # 1 x n x n
    self.u_p = u_p
    # check for numerical overflow
    _ = tf.debugging.assert_all_finite(u_p, "Stein kernel u_p")

    # draw multinomial samples
    if multinom_samples is None:
      w = self.sample_multinomial(n, num_boot) # num_boot x n
    else:
      w = multinom_samples
    # scale to [0, 1]
    w /= n # num_boot x n
    # center
    w -= 1/float(n)

    # compute outerproduct
    w_outer = tf.expand_dims(w, axis=2) * tf.expand_dims(w, axis=1) # num_boot x n x n
    # remove diagonal
    w_outer = tf.linalg.set_diag(w_outer, tf.zeros(w_outer.shape[0:-1])) # num_boot x n x n
    # compute bootstrap samples
    self.ksd_star = tf.reduce_sum(w_outer * u_p, [1, 2]) # num_boot
    
    return w_outer * u_p

  def _test_once(self, alpha):
    """Utility function that performs bootstrap test once"""
    critical_val = np.quantile(self.ksd_star.numpy(), 1-alpha)
    reject = 1 if self.ksd_hat > critical_val else 0
    p_val = np.count_nonzero(self.ksd_star.numpy() > self.ksd_hat) / (self.ksd_star.shape[0] + 1)
    return reject, critical_val, p_val

  def test_once(
    self, 
    alpha: float, 
    num_boot: int, 
    X: tf.Tensor=None,
    multinom_samples: tf.Tensor=None,
    **kwargs
  ):
    """
    Perform bootstrap test once and return summary
    Inputs:
      alpha: significance level of test

    Returns:
      reject: 1 if test is rejected; 0 otherwise
    """
    u_p = self.compute_test_statistic(X, **kwargs)

    self.compute_bootstrap(num_boot=num_boot, u_p=u_p, multinom_samples=multinom_samples)

    reject, critical_val, p_val = self._test_once(alpha)
    conclusion = "Rejected" if reject else "NOT rejected"
    self.test_summary = "Significance\t: {} \nCritical value\t: {:.5f} \np-value\t: {:.5f} \nTest statistic\t: {:.5f} \nTest result\t: {:s}".format(
      alpha, critical_val, p_val, self.ksd_hat, conclusion)
    return reject, p_val

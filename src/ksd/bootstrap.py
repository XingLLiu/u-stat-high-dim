import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import trange

class Bootstrap:
  def __init__(self, ksd):
    """
    Inputs:
      ksd: instance of KSD class
    """
    self.ksd = ksd
    self.ksd_hat = None
    self.ksd_star = None
    self.alpha = None
  
  def sample_multinomial(self, n: int, num_boot: int):
    """
    Inputs:
      n: number of samples
      num_boot: number of bootstrap samples to be drawn
    
    Returns:
      w \sim 1/n * Multi(n; 1/n, \ldots, 1/n)
    """
    n = float(n)
    multinom = tfp.distributions.Multinomial(n, probs=tf.ones(int(n))/n)
    w = multinom.sample(num_boot) # num_boot x n
    w /= n # num_boot x n
    return w

  def compute_test_statistic(self, X: tf.Tensor, **kwargs):
    n = X.shape[0]
    # compute u_p(xi, xj) using the U-statistic (required for goodness-of-fit tests)
    u_p = self.ksd(X=X, Y=tf.identity(X), output_dim=2, **kwargs).numpy() # n x n
    u_p_nodiag = u_p - tf.linalg.diag(tf.linalg.diag_part(u_p)) # n x n
    # u_p = tf.expand_dims(u_p, axis=0) # 1 x n x n

    # compute test statistic
    self.ksd_hat = tf.reduce_sum(u_p_nodiag).numpy() / (n*(n - 1))

    return u_p

  def compute_bootstrap(
    self, 
    num_boot: int, 
    u_p: tf.Tensor
  ):
    """
    Inputs:
      u_p: u_p(xi, xj). Shape: (n, n)
      ksd_hat: estimated KSD
    """
    n = u_p.shape[0]
    u_p = tf.expand_dims(u_p, axis=0) # 1 x n x n

    # draw multinomial samples
    w = self.sample_multinomial(n, num_boot) # num_boot x n
    # center
    w -= 1/float(n)

    # compute outerproduct
    w_outer = tf.expand_dims(w, axis=2) * tf.expand_dims(w, axis=1) # num_boot x n x n
    # remove diagonal
    w_outer = w_outer - tf.linalg.diag(tf.linalg.diag_part(w_outer)) # num_boot x n x n
    # compute bootstrap samples
    self.ksd_star = tf.reduce_sum(w_outer * u_p, [1, 2]) # num_book
    return w_outer * u_p

  def _test_once(self, alpha):
    # critical_val = tfp.stats.percentile(self.ksd_star, 100*(1-alpha)).numpy()
    critical_val = np.quantile(self.ksd_star.numpy(), 1-alpha)
    reject = True if self.ksd_hat > critical_val else False
    return reject, critical_val

  def test_once(
    self, 
    alpha: float, 
    num_boost: int, 
    X: tf.Tensor=None,

    **kwargs
  ):
    """
    Inputs:
      alpha: significance level of test

    Returns:
      reject: 1 if test is rejected; 0 otherwise
    """
    u_p = self.compute_test_statistic(X, **kwargs)
    self.compute_bootstrap(num_boot=num_boost, u_p=u_p)

    reject, critical_val = self._test_once(alpha)
    conclusion = "Rejected" if reject else "NOT rejected"
    self.test_summary = "Significance\t: {} \nCritical value\t: {:.5f} \nTest statistic\t: {:.5f} \nTest result\t: {:s}".format(
      alpha, critical_val, self.ksd_hat, conclusion)
    return reject

  def test(
    self, 
    alpha: float, 
    num_test: int, 
    num_boost: int, 
    X: tf.Tensor, 
    verbose: bool=True,
    **kwargs
  ):
    """
    Repeat bootstrap tests for num_test times for computing error rate
    Inputs:
      alpha: significance level of test
      num_test: number of tests to repeat

    Returns:
      list of bootstrap samples
      list of test results
    """
    test_res = [-1] * num_test
    ksd_star = [-1] * num_test
    critical_val = [-1] * num_test
    iterator = trange(num_test) # if verbose else range(num_test)
    # compute u_p (only need to do so once)
    u_p = self.compute_test_statistic(X, **kwargs)

    for i in iterator:
      self.compute_bootstrap(num_boot=num_boost, u_p=u_p)
      ksd_star[i] = self.ksd_star
      test_res[i], critical_val[i] = self._test_once(alpha=alpha)
      iterator.set_description(f"Repeating tests: {i+1} of {num_test}")

    return ksd_star, test_res, critical_val






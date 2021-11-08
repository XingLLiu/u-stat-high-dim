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

  def compute_bootstrap(
    self, 
    num_boot: int, 
    X: tf.Tensor, 
    **kwargs
  ):
    """
    """
    n = X.shape[0]
    # compute u_p(xi, xj)
    u_p = self.ksd(X=X, Y=tf.identity(X), output_dim=2, **kwargs).numpy() # n x n
    u_p = tf.expand_dims(u_p, axis=0) # 1 x n x n

    # compute test statistic
    self.ksd_hat = tf.reduce_sum(u_p).numpy() / n**2

    # draw multinomial samples
    w = self.sample_multinomial(n, num_boot) # num_boot x n

    # compute outerproduct
    w_outer = tf.expand_dims(w, axis=2) * tf.expand_dims(w, axis=1) # num_boot x n x n
    # remove diagonal
    w_outer = w_outer - tf.linalg.diag(tf.linalg.diag_part(w_outer)) # num_boot x n x n
    # compute bootstrap samples
    self.ksd_star = tf.reduce_sum(w_outer * u_p, [1, 2]) # num_book

  def test(self, alpha: float):
    """
    Inputs:
      alpha: significance level of test

    Returns:
      reject: 1 if test is rejected; 0 otherwise
    """
    critical_val = tfp.stats.percentile(self.ksd_star, 1-alpha).numpy()
    reject = True if self.ksd_hat > critical_val else False
    conclusion = "Rejected" if reject else "NOT rejected"
    self.test_summary = "Significance\t: {} \nCritical value\t: {:.5f} \nTest statistic\t: {:.5f} \nTest result\t: {:s}".format(
      alpha, critical_val, self.ksd_hat, conclusion)
    return reject

  def test_repeated(
    self, 
    alpha: float, 
    num_test: int, 
    num_boost: int, 
    X: tf.Tensor, 
    **kwargs
  ):
    """
    Inputs:
      alpha: significance level of test
      num_test: number of tests to repeat

    Returns:
      list of bootstrap samples
      list of test results
    """
    test_res = [-1] * num_test
    ksd_star = [-1] * num_test
    for i in trange(num_test):
      self.compute_bootstrap(num_boot=num_boost, X=X, **kwargs)
      ksd_star[i] = self.ksd_star
      test_res[i] = self.test(alpha=alpha)

    return ksd_star, test_res






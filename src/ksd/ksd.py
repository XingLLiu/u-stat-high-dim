import tensorflow as tf
import tensorflow_probability as tfp

class KSD:
  def __init__(
    self,
    kernel: tf.Module,
    target: tfp.distributions.Distribution=None,
    log_prob: callable=None
  ):
    """
    Inputs:
        target (tfp.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
        kernel (tf.nn.Module): [description]
        optimizer (tf.optim.Optimizer): [description]
    """
    if target is not None:
      self.p = target
      self.log_prob = target.log_prob
    else:
      self.p = None
      self.log_prob = log_prob
    self.k = kernel

  def u_p(self, X: tf.Tensor, Y: tf.Tensor, output_dim: int=1):
    """
    Inputs:
      X: n x dim
      Y: m x dim
    """
    # copy data for score computation
    X_cp = tf.identity(X)
    Y_cp = tf.identity(Y)

    ## calculate scores using autodiff
    if not hasattr(self.p, "grad_log"):
      with tf.GradientTape() as g:
        g.watch(X_cp)
        log_prob_X = self.log_prob(X_cp)
      score_X = g.gradient(log_prob_X, X_cp) # n x dim
      with tf.GradientTape() as g:
        g.watch(Y_cp)
        log_prob_Y = self.log_prob(Y_cp) # m x dim
      score_Y = g.gradient(log_prob_Y, Y_cp)
    else:
      score_X = self.p.grad_log(X_cp) # n x dim
      score_Y = self.p.grad_log(Y_cp) # m x dim
      assert score_X.shape == X_cp.shape

    # median heuristic
    if self.k.med_heuristic:
      self.k.bandwidth(X, Y)

    # kernel
    K_XY = self.k(X, Y) # n x m
    dim = X.shape[-1]

    # term 1
    term1_mat = tf.linalg.matmul(score_X, score_Y, transpose_b=True) * K_XY # n x m
    # term 2
    if dim <= 10:
      grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
      term2_mat = tf.expand_dims(score_X, -2) * grad_K_Y # n x m x dim
      term2_mat = tf.reduce_sum(term2_mat, axis=-1)
    else:  
      term2_mat = self.k.grad_second_prod(X, Y, score_X)

    # term3
    if dim <= 10:
      grad_K_X = self.k.grad_first(X, Y) # n x m x dim
      term3_mat = tf.expand_dims(score_Y, -3) * grad_K_X # n x m x dim
      term3_mat = tf.reduce_sum(term3_mat, axis=-1)
    else:
      term3_mat = self.k.grad_first_prod(X, Y, score_Y)

    # term4
    term4_mat = self.k.gradgrad(X, Y) # n x m

    u_p = term1_mat + term2_mat + term3_mat + term4_mat
    u_p_nodiag = tf.linalg.set_diag(u_p, tf.zeros(u_p.shape[:-1]))

    if output_dim == 1:
      ksd = tf.reduce_sum(
        u_p_nodiag,
        axis=[-1, -2]
      ) / (X.shape[-2] * (Y.shape[-2]-1))
      return ksd
    elif output_dim == 2:
      assert term1_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
      assert term2_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
      assert term3_mat.shape[-2:] == (X.shape[-2], Y.shape[-2])
      assert term4_mat.shape[-2:] == (X.shape[-2], Y.shape[-2]), term4_mat.shape
      return u_p_nodiag

  def u_p_moment(self, X: tf.Tensor, Y: tf.Tensor, k: int):
    u_p = self.u_p(X, Y, output_dim=2)
    u_p_sq = u_p**k
    u_p_sq = tf.math.reduce_sum(
      u_p_sq * (1 - tf.eye(u_p_sq.shape[-2]))
    ) / (X.shape[-2] * (X.shape[-2] - 1))
    return u_p_sq

  def abs_cond_central_moment(self, X: tf.Tensor, Y: tf.Tensor, k: int):
    u_p = self.u_p(X, Y, output_dim=2)
    g = tf.math.reduce_sum(u_p, axis=-1) / X.shape[-2]
    ksd = tf.math.reduce_sum(u_p) / (X.shape[-2]**2) # /n^2 as diagnal is included
    
    mk = tf.math.reduce_sum(
      tf.math.abs(g - ksd)**k
    ) / X.shape[-2]
    return mk

  def abs_full_central_moment(self, X: tf.Tensor, Y: tf.Tensor, k: int):
    u_p = self.u_p(X, Y, output_dim=2)
    ksd = tf.math.reduce_sum(u_p) / (X.shape[-2]**2) # /n^2 as diagnal is included

    Mk = tf.math.reduce_sum(
      tf.math.abs(u_p - ksd)**k
    ) / X.shape[-2]**2
    return Mk

  def beta_k(self, X: tf.Tensor, Y: tf.Tensor, k: int):
    u_mat = self.__call__(X, Y, output_dim=2) # n x n
    n = X.shape[-2]
    witness = tf.reduce_sum(u_mat, axis=1) / n # n
    term1 = tf.reduce_sum(witness**k) / n
    return term1

  def __call__(self, X: tf.Tensor, Y: tf.Tensor, output_dim: int=1):
    """
    Inputs:
      X: n x dim
      Y: m x dim
    """
    return self.u_p(X, Y, output_dim)

  def h1_var(self, return_scaled_ksd: bool=False, **kwargs):
    """Compute the variance of the asymtotic Gaussian distribution under H_1
    Args:
      return_scaled_ksd: if True, return KSD / (\sigma_{H_1} + jitter), where 
        \sigma_{H_1}^2 is the asymptotic variance of the KSD estimate under H1
    """
    u_mat = self.__call__(output_dim=2, **kwargs) # n x n
    n = kwargs["X"].shape[-2]
    witness = tf.reduce_sum(u_mat, axis=1) # n
    term1 = tf.reduce_sum(witness**2) * 4 / n**3
    term2 = tf.reduce_sum(u_mat)**2 * 4 / n**4
    var = term1 - term2 + 1e-12
    if not return_scaled_ksd:
      return var
    else:
      ksd = tf.reduce_sum(u_mat) / n**2
      ksd_scaled = ksd / tf.math.sqrt(var)
      return var, ksd_scaled

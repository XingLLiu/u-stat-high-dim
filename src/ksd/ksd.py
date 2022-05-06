import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tqdm import trange

class KSD:
  def __init__(
    self,
    target: tfp.distributions.Distribution,
    kernel: tf.Module,
  ):
    """
    Inputs:
        target (tf.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
        kernel (tf.nn.Module): [description]
        optimizer (tf.optim.Optimizer): [description]
    """
    self.p = target
    self.k = kernel

  def __call__(self, X: tf.Tensor, Y: tf.Tensor, output_dim: int=1):
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
        log_prob_X = self.p.log_prob(X_cp)
      score_X = g.gradient(log_prob_X, X_cp) # n x dim
      with tf.GradientTape() as g:
        g.watch(Y_cp)
        log_prob_Y = self.p.log_prob(Y_cp) # m x dim
      score_Y = g.gradient(log_prob_Y, Y_cp)
    else:
      score_X = self.p.grad_log(X_cp) # n x dim
      score_Y = self.p.grad_log(Y_cp) # m x dim
      assert score_X.shape == X_cp.shape

    # median heuristic #TODO using pre-specified bandwidth
    if self.k.med_heuristic:
      self.k.bandwidth(X, Y)
    
    # kernel
    K_XY = self.k(X, Y) # n x m
    
    # kernel grad
    grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
    grad_K_X = self.k.grad_first(X, Y) # n x m x dim
    
    # term 1
    term1_mat = tf.linalg.matmul(score_X, score_Y, transpose_b=True) * K_XY # n x m
    term1 = tf.reduce_sum(term1_mat)
    # term 2
    term2_mat = tf.expand_dims(score_X, 1) * grad_K_Y # n x m x dim
    term2_mat = tf.reduce_sum(term2_mat, axis=-1)
    term2 = tf.reduce_sum(term2_mat)
    # term3
    term3_mat = tf.expand_dims(score_Y, 0) * grad_K_X # n x m x dim
    term3_mat = tf.reduce_sum(term3_mat, axis=-1)
    term3 = tf.reduce_sum(term3_mat)
    # term4
    gradgrad_K = self.k.gradgrad(X, Y) # n x m x dim x dim
    term4_mat = tf.experimental.numpy.diagonal(gradgrad_K, axis1=2, axis2=3) # n x m x dim
    term4_mat = tf.reduce_sum(term4_mat, axis=2) # n x m
    term4 = tf.reduce_sum(term4_mat)

    if output_dim == 1:
      ksd = (term1 + term2 + term3 + term4) / (X.shape[0] * Y.shape[0])
      return ksd
    elif output_dim == 2:
      assert term1_mat.shape == (X.shape[0], Y.shape[0])
      assert term2_mat.shape == (X.shape[0], Y.shape[0])
      assert term3_mat.shape == (X.shape[0], Y.shape[0])
      assert term4_mat.shape == (X.shape[0], Y.shape[0]), term4_mat.shape
      return term1_mat + term2_mat + term3_mat + term4_mat

  def h1_var(self, return_scaled_ksd: bool=False, **kwargs):
    """Compute the variance of the asymtotic Gaussian distribution under H_1
    Args:
      return_scaled_ksd: if True, return KSD / (\sigma_{H_1} + jitter), where 
        \sigma_{H_1}^2 is the asymptotic variance of the KSD estimate under H1
    """
    u_mat = self.__call__(output_dim=2, **kwargs) # n x n
    n = kwargs["X"].shape[0]
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

class ConvolvedKSD:
  def __init__(
    self,
    target: tfp.distributions.Distribution,
    kernel: tf.Module,
  ):
    """
    Inputs:
        target (tf.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
        kernel (tf.nn.Module): [description]
        optimizer (tf.optim.Optimizer): [description]
    """
    self.p = target
    self.k = kernel

  def __call__(self, X: tf.Tensor, Y: tf.Tensor, conv_samples: tf.Tensor, output_dim: int=1):
    """
    Inputs:
      X: n x dim
      Y: m x dim
      output_dim: dim of output. If 1, then KSD_hat is returned. If 2, then 
        the matrix [ u_p(xi, xj) ]_{ij} is returned
    """
    ## copy data for score computation
    X_cp = tf.expand_dims(tf.identity(X), axis=0) # 1 x n x dim
    Y_cp = tf.expand_dims(tf.identity(Y), axis=0) # 1 x m x dim

    ## estimate score for convolution
    Z = tf.expand_dims(conv_samples, axis=1) # l x 1 x dim

    with tf.GradientTape() as g:
      g.watch(X_cp)
      diff_1 = X_cp - Z # l x n x dim #TODO broadcasting is potentially causing problems
      prob_1 = self.p.prob(diff_1) # l x n
    grad_1 = g.gradient(prob_1, X_cp) # 1 x n x dim
    grad_1 = tf.squeeze(grad_1, axis=0) # n x dim
    score_X = grad_1 / tf.expand_dims(
      tf.math.reduce_sum(prob_1, axis=0), axis=1) # n x dim

    with tf.GradientTape() as g:
      g.watch(Y_cp)
      diff_2 = Y_cp - tf.identity(Z)
      prob_2 = self.p.prob(diff_2) # m x dim
    grad_2 = g.gradient(prob_2, Y_cp)
    grad_2 = tf.squeeze(grad_2, axis=0) # m x dim
    score_Y = grad_2 / tf.expand_dims(
      tf.math.reduce_sum(prob_2, axis=0), axis=1) # m x dim

    # median heuristic
    self.k.bandwidth(X, Y)
    
    # kernel
    K_XY = self.k(X, Y) # n x m
    
    # kernel grad
    grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
    grad_K_X = self.k.grad_first(X, Y) # n x m x dim

    # term 1
    term1_mat = tf.linalg.matmul(score_X, score_Y, transpose_b=True) * K_XY # n x m
    term1 = tf.reduce_sum(term1_mat)
    # term 2
    term2_mat = tf.expand_dims(score_X, 1) * grad_K_Y # n x m x dim
    term2_mat = tf.reduce_sum(term2_mat, axis=-1)
    term2 = tf.reduce_sum(term2_mat)
    # term3
    term3_mat = tf.expand_dims(score_Y, 0) * grad_K_X # n x m x dim
    term3_mat = tf.reduce_sum(term3_mat, axis=-1)
    term3 = tf.reduce_sum(term3_mat)
    # term4
    gradgrad_K = self.k.gradgrad(X, Y) # n x m x dim x dim
    term4_mat = tf.experimental.numpy.diagonal(gradgrad_K, axis1=2, axis2=3) # n x m x dim
    term4_mat = tf.reduce_sum(term4_mat, axis=2) # n x m
    term4 = tf.reduce_sum(term4_mat)

    if output_dim == 1:
      ksd = (term1 + term2 + term3 + term4) / (X.shape[0] * Y.shape[0])
      return ksd
    elif output_dim == 2:
      assert term1_mat.shape == (X.shape[0], Y.shape[0])
      assert term2_mat.shape == (X.shape[0], Y.shape[0])
      assert term3_mat.shape == (X.shape[0], Y.shape[0])
      assert term4_mat.shape == (X.shape[0], Y.shape[0])
      return term1_mat + term2_mat + term3_mat + term4_mat

  def eval(self, log_noise_std: float, X: tf.Tensor, Y: tf.Tensor, conv_samples_full: tf.Tensor, conv_samples: tf.Tensor, output_dim: int=1):
    """
    Inputs:
      X: n x dim
      Y: m x dim
      output_dim: dim of output. If 1, then KSD_hat is returned. If 2, then 
        the matrix [ u_p(xi, xj) ]_{ij} is returned
    """
    noise_sd = tf.exp(log_noise_std)
    
    assert conv_samples.shape == X.shape
    ## add noise to samples
    X += conv_samples * noise_sd
    Y += conv_samples * noise_sd
    
    ## copy data for score computation
    X_cp = tf.expand_dims(tf.identity(X), axis=0) # 1 x n x dim
    Y_cp = tf.expand_dims(tf.identity(Y), axis=0) # 1 x m x dim

    ## estimate score for convolution
    Z = tf.expand_dims(conv_samples_full, axis=1) # l x 1 x dim

    with tf.GradientTape() as g:
      g.watch(X_cp)
      input_1 = X_cp - Z * noise_sd # l x n x dim #TODO broadcasting is potentially causing problems
      prob_1 = self.p.prob(input_1) # l x n
    grad_1 = g.gradient(prob_1, X_cp) # 1 x n x dim
    grad_1 = tf.squeeze(grad_1, axis=0) # n x dim
    score_X = grad_1 / tf.expand_dims(
      tf.math.reduce_sum(prob_1, axis=0), axis=1) # n x dim

    with tf.GradientTape() as g:
      g.watch(Y_cp)
      input_2 = Y_cp - tf.identity(Z) * noise_sd # l x m x dim
      prob_2 = self.p.prob(input_2) # m x dim
    grad_2 = g.gradient(prob_2, Y_cp)
    grad_2 = tf.squeeze(grad_2, axis=0) # m x dim
    score_Y = grad_2 / tf.expand_dims(
      tf.math.reduce_sum(prob_2, axis=0), axis=1) # m x dim

    # median heuristic
    self.k.bandwidth(X, Y)
    
    # kernel
    K_XY = self.k(X, Y) # n x m
    
    # kernel grad
    grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
    grad_K_X = self.k.grad_first(X, Y) # n x m x dim

    # term 1
    term1_mat = tf.linalg.matmul(score_X, score_Y, transpose_b=True) * K_XY # n x m
    term1 = tf.reduce_sum(term1_mat)
    # term 2
    term2_mat = tf.expand_dims(score_X, 1) * grad_K_Y # n x m x dim
    term2_mat = tf.reduce_sum(term2_mat, axis=-1)
    term2 = tf.reduce_sum(term2_mat)
    # term3
    term3_mat = tf.expand_dims(score_Y, 0) * grad_K_X # n x m x dim
    term3_mat = tf.reduce_sum(term3_mat, axis=-1)
    term3 = tf.reduce_sum(term3_mat)
    # term4
    gradgrad_K = self.k.gradgrad(X, Y) # n x m x dim x dim
    term4_mat = tf.experimental.numpy.diagonal(gradgrad_K, axis1=2, axis2=3) # n x m x dim
    term4_mat = tf.reduce_sum(term4_mat, axis=2) # n x m
    term4 = tf.reduce_sum(term4_mat)

    if output_dim == 1:
      ksd = (term1 + term2 + term3 + term4) / (X.shape[0] * Y.shape[0])
      return ksd
    elif output_dim == 2:
      assert term1_mat.shape == (X.shape[0], Y.shape[0])
      assert term2_mat.shape == (X.shape[0], Y.shape[0])
      assert term3_mat.shape == (X.shape[0], Y.shape[0])
      assert term4_mat.shape == (X.shape[0], Y.shape[0])
      return term1_mat + term2_mat + term3_mat + term4_mat

  def h1_var(self, return_scaled_ksd: bool=False, **kwargs):
    """Compute the variance of the asymtotic Gaussian distribution under H_1
    Args:
      return_scaled_ksd: if True, return KSD / (\sigma_{H_1} + jitter), where 
        \sigma_{H_1}^2 is the asymptotic variance of the KSD estimate under H1
    """
    u_mat = self.eval_mat(output_dim=2, **kwargs) # n x n
    n = kwargs["X"].shape[0]

    witness = tf.reduce_sum(u_mat, axis=1) # n
    term1 = tf.reduce_sum(witness**2) * 4 / n**3
    term2 = tf.reduce_sum(u_mat)**2 * 4 / n**4
    var = term1 - term2 + 1e-12
    if not return_scaled_ksd:
      return var
    else:
      ksd = tf.reduce_sum(u_mat) / n**2
      ksd_scaled = ksd / tf.math.sqrt(var)
      return ksd_scaled

  def eval_mat(self, log_noise_std: float, u: tf.Tensor, X: tf.Tensor, Y: tf.Tensor, conv_samples_full: tf.Tensor, conv_samples: tf.Tensor, output_dim: int=1):
    """
    Inputs:
      X: n x dim
      Y: m x dim
      u: (1, dim) unit vector of the direction for noise
      conv_samples: noise samples of shape (n, 1)
      output_dim: dim of output. If 1, then KSD_hat is returned. If 2, then 
        the matrix [ u_p(xi, xj) ]_{ij} is returned
    """
    noise_sd = tf.exp(log_noise_std)
    assert u.shape == (1, X.shape[1]), u.shape
    
    Lmat = noise_sd * u # 1 x dim
    assert Lmat.shape[1] == X.shape[1]
    
    ## add noise to samples
    X += conv_samples @ Lmat # n x dim
    Y += conv_samples @ Lmat # m x dim
    
    ## copy data for score computation
    X_cp = tf.expand_dims(tf.identity(X), axis=0) # 1 x n x dim
    Y_cp = tf.expand_dims(tf.identity(Y), axis=0) # 1 x m x dim

    ## estimate score for convolution
    Z = tf.expand_dims(conv_samples_full, axis=1) # l x 1 x dim

    with tf.GradientTape() as g:
      g.watch(X_cp)
      input_1 = X_cp - Z @ Lmat # l x n x dim
      prob_1 = self.p.prob(input_1) # l x n
    grad_1 = g.gradient(prob_1, X_cp) # 1 x n x dim
    grad_1 = tf.squeeze(grad_1, axis=0) # n x dim
    score_X = grad_1 / tf.expand_dims(
      tf.math.reduce_sum(prob_1, axis=0), axis=1) # n x dim

    with tf.GradientTape() as g:
      g.watch(Y_cp)
      input_2 = Y_cp - tf.identity(Z) @ Lmat # l x m x dim
      prob_2 = self.p.prob(input_2) # l x m
    grad_2 = g.gradient(prob_2, Y_cp) # 1 x n x dim
    grad_2 = tf.squeeze(grad_2, axis=0) # m x dim
    score_Y = grad_2 / tf.expand_dims(
      tf.math.reduce_sum(prob_2, axis=0), axis=1) # m x dim

    # median heuristic
    self.k.bandwidth(X, Y)
    
    # kernel
    K_XY = self.k(X, Y) # n x m
    
    # kernel grad
    grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
    grad_K_X = self.k.grad_first(X, Y) # n x m x dim

    # term 1
    term1_mat = tf.linalg.matmul(score_X, score_Y, transpose_b=True) * K_XY # n x m
    term1 = tf.reduce_sum(term1_mat)
    # term 2
    term2_mat = tf.expand_dims(score_X, 1) * grad_K_Y # n x m x dim
    term2_mat = tf.reduce_sum(term2_mat, axis=-1)
    term2 = tf.reduce_sum(term2_mat)
    # term3
    term3_mat = tf.expand_dims(score_Y, 0) * grad_K_X # n x m x dim
    term3_mat = tf.reduce_sum(term3_mat, axis=-1)
    term3 = tf.reduce_sum(term3_mat)
    # term4
    gradgrad_K = self.k.gradgrad(X, Y) # n x m x dim x dim
    term4_mat = tf.experimental.numpy.diagonal(gradgrad_K, axis1=2, axis2=3) # n x m x dim
    term4_mat = tf.reduce_sum(term4_mat, axis=2) # n x m
    term4 = tf.reduce_sum(term4_mat)

    if output_dim == 1:
      ksd = (term1 + term2 + term3 + term4) / (X.shape[0] * Y.shape[0])
      return ksd
    elif output_dim == 2:
      assert term1_mat.shape == (X.shape[0], Y.shape[0])
      assert term2_mat.shape == (X.shape[0], Y.shape[0])
      assert term3_mat.shape == (X.shape[0], Y.shape[0])
      assert term4_mat.shape == (X.shape[0], Y.shape[0])
      return term1_mat + term2_mat + term3_mat + term4_mat
  
  def optim(self, nsteps: int, optimizer: tf.optimizers, param: tf.Tensor, verbose: bool=False, 
    desc=None, **kwargs):
    """
    Inputs:
      allparams: whether to optimise for the entire cov matrix
    """
    self.losses = []
    self.params = []
    assert param.shape == (1, kwargs["X"].shape[1]+1)
    
    iterator = trange(nsteps) if verbose else range(nsteps)

    # define loss function (-ksd)
    def loss_fn(param):
      res = self.h1_var(
          return_scaled_ksd=True,
          log_noise_std=param[0, 0],
          X=kwargs["X"],
          Y=tf.identity(kwargs["Y"]),
          conv_samples_full=kwargs["conv_samples_full"],
          conv_samples=kwargs["conv_samples"],
          u=param[:1, 1:])
      return -res
    
    # minimise
    for i in iterator:
      if desc:
        desc.set_description(f"Gradient step [{i+1} / {nsteps}]")
      # open a GradientTape
      with tf.GradientTape() as tape:
          # forward pass.
          # u_n = param[:1, 1:] / (tf.math.sqrt(tf.reduce_sum(param[:1, 1:]**2, axis=1)) + 1e-12) # 1 x dim
          # param_n = tf.concat([param[:1, :1], u_n], axis=1) # 1 x (dim+1)
          # loss_value = loss_fn(param_n)
          loss_value = loss_fn(param)

      # get gradients of loss wrt noise params
      gradients = tape.gradient(loss_value, param)
      
      # store results
      # u_n = param[:1, 1:] / (tf.math.sqrt(tf.reduce_sum(param[:1, 1:]**2, axis=1)) + 1e-12) # 1 x dim
      # param_n = tf.concat([param[:1, :1], u_n], axis=1) # 1 x (dim+1)
      self.losses.append(loss_value)
      self.params.append(tf.constant(param))

      # update noise params
      optimizer.apply_gradients(zip([gradients], [param]))

  def optim_var(self, nsteps: int, optimizer: tf.optimizers, param: tf.Tensor, u_vec: tf.Tensor,
    verbose: bool=False, **kwargs):
    """Optimise for log std of noise only
    Inputs:
      allparams: whether to optimise for the entire cov matrix
    """
    self.losses = []
    self.params = []
    assert u_vec.shape == (1, kwargs["X"].shape[1])
    
    iterator = trange(nsteps) if verbose else range(nsteps)

    # define loss function (-ksd)
    def loss_fn(param):
      res = self.h1_var(
          return_scaled_ksd=True,
          log_noise_std=param,
          X=kwargs["X"],
          Y=tf.identity(kwargs["Y"]),
          conv_samples_full=kwargs["conv_samples_full"],
          conv_samples=kwargs["conv_samples"],
          u=u_vec)
      return -res
    
    # minimise
    for i in iterator:
      # open a GradientTape
      with tf.GradientTape() as tape:
          # forward pass.
          loss_value = loss_fn(param)

      # get gradients of loss wrt noise params
      gradients = tape.gradient(loss_value, param)
      
      # store results
      self.losses.append(loss_value)
      self.params.append(tf.constant(param))

      # update noise params
      optimizer.apply_gradients(zip([gradients], [param]))


class SDEKSD:
  def __init__(
    self,
    target: tfp.distributions.Distribution,
    kernel: tf.Module,
  ):
    """
    Inputs:
        target (tf.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
        kernel (tf.nn.Module): [description]
        optimizer (tf.optim.Optimizer): [description]
    """
    self.p = target
    self.k = kernel

  def __call__(self, X: tf.Tensor, Y: tf.Tensor, conv_samples: tf.Tensor, alpha_t: float, output_dim: int=1):
    """
    Inputs:
      X: n x dim
      Y: m x dim
      output_dim: dim of output. If 1, then KSD_hat is returned. If 2, then 
        the matrix [ u_p(xi, xj) ]_{ij} is returned
    """
    ## copy data for score computation
    X_cp = tf.expand_dims(tf.identity(X), axis=0) # 1 x n x dim
    Y_cp = tf.expand_dims(tf.identity(Y), axis=0) # 1 x m x dim

    ## estimate score for convolution
    Z = tf.expand_dims(conv_samples, axis=1) # l x 1 x dim

    with tf.GradientTape() as g:
      g.watch(X_cp)
      diff_1 = (X_cp - tf.math.sqrt(1 - alpha_t) * Z) / tf.math.sqrt(alpha_t) # l x n x dim
      prob_1 = self.p.prob(diff_1) # l x n
    grad_1 = g.gradient(prob_1, X_cp) # 1 x n x dim
    grad_1 = tf.squeeze(grad_1, axis=0) # n x dim
    score_X = grad_1 / tf.expand_dims(
      tf.math.reduce_sum(prob_1, axis=0), axis=1) # n x dim
    ## prevents division by 0
    score_X = tf.where(tf.math.is_nan(score_X), 0., score_X) # n x dim
    _ = tf.debugging.assert_all_finite(grad_1, "grad_1")
    _ = tf.debugging.assert_all_finite(prob_1, "prob_1")
    _ = tf.debugging.assert_all_finite(score_X, "score")

    with tf.GradientTape() as g:
      g.watch(Y_cp)
      diff_2 = (Y_cp - tf.math.sqrt(1 - alpha_t) * tf.identity(Z)) / tf.math.sqrt(alpha_t)
      prob_2 = self.p.prob(diff_2) # m x dim
    grad_2 = g.gradient(prob_2, Y_cp)
    grad_2 = tf.squeeze(grad_2, axis=0) # m x dim
    score_Y = grad_2 / tf.expand_dims(
      tf.math.reduce_sum(prob_2, axis=0), axis=1) # m x dim
    ## prevents division by 0
    score_Y = tf.where(tf.math.is_nan(score_Y), 0., score_Y) # n x dim

    # median heuristic
    self.k.bandwidth(X, Y)
    
    # kernel
    K_XY = self.k(X, Y) # n x m
    
    # kernel grad
    grad_K_Y = self.k.grad_second(X, Y) # n x m x dim
    grad_K_X = self.k.grad_first(X, Y) # n x m x dim

    # term 1
    term1_mat = tf.linalg.matmul(score_X, score_Y, transpose_b=True) * K_XY # n x m
    term1 = tf.reduce_sum(term1_mat)
    # term 2
    term2_mat = tf.expand_dims(score_X, 1) * grad_K_Y # n x m x dim
    term2_mat = tf.reduce_sum(term2_mat, axis=-1)
    term2 = tf.reduce_sum(term2_mat)
    # term3
    term3_mat = tf.expand_dims(score_Y, 0) * grad_K_X # n x m x dim
    term3_mat = tf.reduce_sum(term3_mat, axis=-1)
    term3 = tf.reduce_sum(term3_mat)
    # term4
    gradgrad_K = self.k.gradgrad(X, Y) # n x m x dim x dim
    term4_mat = tf.experimental.numpy.diagonal(gradgrad_K, axis1=2, axis2=3) # n x m x dim
    term4_mat = tf.reduce_sum(term4_mat, axis=2) # n x m
    term4 = tf.reduce_sum(term4_mat)
    _ = tf.debugging.assert_all_finite(term1, "term1")
    _ = tf.debugging.assert_all_finite(term2, "term2")
    _ = tf.debugging.assert_all_finite(term3, "term3")
    _ = tf.debugging.assert_all_finite(term4, "term4")

    if output_dim == 1:
      ksd = (term1 + term2 + term3 + term4) / (X.shape[0] * Y.shape[0])
      return ksd
    elif output_dim == 2:
      assert term1_mat.shape == (X.shape[0], Y.shape[0])
      assert term2_mat.shape == (X.shape[0], Y.shape[0])
      assert term3_mat.shape == (X.shape[0], Y.shape[0])
      assert term4_mat.shape == (X.shape[0], Y.shape[0])
      return term1_mat + term2_mat + term3_mat + term4_mat

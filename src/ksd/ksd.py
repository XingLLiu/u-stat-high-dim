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
    with tf.GradientTape() as g:
      g.watch(X_cp)
      log_prob_X = self.p.log_prob(X_cp)
    score_X = g.gradient(log_prob_X, X_cp) # n x dim
    with tf.GradientTape() as g:
      g.watch(Y_cp)
      log_prob_Y = self.p.log_prob(Y_cp) # m x dim
    score_Y = g.gradient(log_prob_Y, Y_cp)

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
      # print(term1.numpy(), term2.numpy(), term3.numpy(), term4.numpy())
      ksd = (term1 + term2 + term3 + term4) / (X.shape[0] * Y.shape[0])
      return ksd
    elif output_dim == 2:
      assert term1_mat.shape == (X.shape[0], Y.shape[0])
      assert term2_mat.shape == (X.shape[0], Y.shape[0])
      assert term3_mat.shape == (X.shape[0], Y.shape[0])
      assert term4_mat.shape == (X.shape[0], Y.shape[0]), term4_mat.shape
      return term1_mat + term2_mat + term3_mat + term4_mat


class ConvolvedKSD:
  def __init__(
    self,
    target: tfp.distributions.Distribution,
    kernel: tf.Module,
    conv_kernel: tfp.distributions.Distribution,
  ):
    """
    Inputs:
        target (tf.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
        kernel (tf.nn.Module): [description]
        optimizer (tf.optim.Optimizer): [description]
    """
    self.p = target
    self.k = kernel
    self.conv_kernel = conv_kernel

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

  def eval_mat(self, log_noise_std: float, X: tf.Tensor, Y: tf.Tensor, conv_samples_full: tf.Tensor, conv_samples: tf.Tensor, output_dim: int=1, u: tf.Tensor=None):
    """
    Inputs:
      X: n x dim
      Y: m x dim
      u: noise is of the form \sigma * u @ u^T Z, Z \sim N(0, I_d)
      conv_samples: noise samples of shape (n, 1)
      output_dim: dim of output. If 1, then KSD_hat is returned. If 2, then 
        the matrix [ u_p(xi, xj) ]_{ij} is returned
    """
    noise_sd = tf.exp(log_noise_std)
    if u is None:
      noise_mat = noise_sd * tf.eye(X.shape[1]) # dim x dim
    else:
      noise_mat = noise_sd * tf.reshape(u, (1, -1)) # 1 x dim

    ## add noise to samples
    X += conv_samples * noise_mat
    Y += conv_samples * noise_mat
    
    ## copy data for score computation
    X_cp = tf.expand_dims(tf.identity(X), axis=0) # 1 x n x dim
    Y_cp = tf.expand_dims(tf.identity(Y), axis=0) # 1 x m x dim

    ## estimate score for convolution
    Z = tf.expand_dims(conv_samples_full, axis=1) # l x 1 x dim

    with tf.GradientTape() as g:
      g.watch(X_cp)
      input_1 = X_cp - Z * noise_mat # l x n x dim #TODO broadcasting is potentially causing problems
      prob_1 = self.p.prob(input_1) # l x n
    grad_1 = g.gradient(prob_1, X_cp) # 1 x n x dim
    grad_1 = tf.squeeze(grad_1, axis=0) # n x dim
    score_X = grad_1 / tf.expand_dims(
      tf.math.reduce_sum(prob_1, axis=0), axis=1) # n x dim

    with tf.GradientTape() as g:
      g.watch(Y_cp)
      input_2 = Y_cp - tf.identity(Z) * noise_mat # l x m x dim
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

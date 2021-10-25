
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class KSD:
  def __init__(
    self,
    target: tfp.distributions,
    kernel: tf.Module,
    device: tf.device="cpu",
  ):
    """
    Inputs:
        target (tf.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
        kernel (tf.nn.Module): [description]
        optimizer (tf.optim.Optimizer): [description]
    """
    self.p = target
    self.k = kernel
    self.device = device

  def __call__(self, X: tf.Tensor, Y: tf.Tensor):
    """
    Inputs:
      X: n x dim
      Y: m x dim
    """
    # copy data for score computation
    X_cp = tf.identity(X)
    Y_cp = tf.identity(Y)

    with tf.GradientTape() as g:
      g.watch(X_cp)
      log_prob_X = self.p.log_prob(X_cp)
    score_X = g.gradient(log_prob_X, X_cp) # n x dim
    with tf.GradientTape() as g:
      g.watch(Y_cp)
      log_prob_Y = self.p.log_prob(Y_cp) # m x dim
    score_Y = g.gradient(log_prob_Y, Y_cp)

    # median heuristic
    self.k.bandwidth(X, Y)
    
    # kernel, grad and hessian
    with tf.GradientTape() as g:
      g.watch(X)
      with tf.GradientTape() as tapeY, tf.GradientTape() as tapeX:
        tapeY.watch(Y)
        tapeX.watch(X)
        
        K_XY = self.k(X, Y) # n x m
        K_XY_Y = tf.reduce_sum(K_XY, axis=1)
        K_XY_X = tf.reduce_sum(K_XY, axis=0)

      # grad_k_Y
      grad_K_Y = tapeY.jacobian(K_XY_Y, Y) # n x m x dim
      grad_K_Y_sum = tf.reduce_sum(grad_K_Y, axis=0)
      
      # grad_k_X
      grad_K_X = tapeX.jacobian(K_XY_X, X) # n x m x dim

    gradgrad_K = g.jacobian(grad_K_Y_sum, X) # n x dim x m x dim
    gradgrad_K = tf.transpose(gradgrad_K, (0, 2, 1, 3)) # n x m x dim x dim

    # term 1
    term1 = tf.linalg.matmul(score_X, score_Y, transpose_b=True) * K_XY # n x m
    term1 = tf.reduce_sum(term1)
    # term 2
    term2 = tf.expand_dims(score_X, 1) * grad_K_Y # n x m x dim
    term2 = tf.reduce_sum(term2)
    # term3
    term3 = tf.expand_dims(score_Y, 0) * grad_K_X # n x m x dim
    term3 = tf.reduce_sum(term3)
    # term4
    diag_gradgrad_K = tf.linalg.diag_part(gradgrad_K) # n x m
    print(diag_gradgrad_K.shape)
    term4 = tf.reduce_sum(diag_gradgrad_K)

    ksd = (term1 + term2 + term3 + term4) / (X.shape[0] * Y.shape[0])
    # print("terms:", tf.reduce_sum(term1), tf.reduce_sum(term2), tf.reduce_sum(term3), tf.reduce_sum(term4))
    return ksd
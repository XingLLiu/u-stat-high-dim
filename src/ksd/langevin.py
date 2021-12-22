import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class Langevin:
  def __init__(self, log_prob: callable) -> None:
    self.log_prob = log_prob
    self.x = None
    self.noise_dist = tfd.Normal(0., 1.)

  def run(self, steps: int, step_size: float, x_init: tf.Tensor):
    n, dim = x_init.shape
    self.x = tf.Variable(-1 * tf.ones((steps, n, dim))) # nsteps x n x dim
    self.x[0, :, :].assign(x_init)

    self.noise = self.noise_dist.sample((steps, n, dim)) # nsteps x n x dim
    scale = tf.math.sqrt(2 * step_size)

    for t in range(steps-1):
      x_t = tf.identity(self.x[t, :, :])
      # calculate scores using autodiff
      with tf.GradientTape() as g:
        g.watch(x_t)
        log_prob_x = self.log_prob(x_t) # n x dim
      score = g.gradient(log_prob_x, x_t) # n x dim
      assert score.shape == (n, dim)
      
      # compute langevin update
      drive = step_size * score # n x dim
      x_next = self.x[t, :, :] + drive + scale * self.noise[t, :, :] # n x dim
      self.x[t+1, :, :].assign(x_next)


class RandomWalk:
  def __init__(self, log_prob: callable) -> None:
    self.log_prob = log_prob
    self.x = None
    self.noise_dist = tfd.Normal(0., 1.)

  def run(self, steps: int, step_size: float, x_init: tf.Tensor):
    n, dim = x_init.shape
    self.x = tf.Variable(-1 * tf.ones((steps, n, dim))) # nsteps x n x dim
    self.x[0, :, :].assign(x_init)

    self.noise = self.noise_dist.sample((steps, n, dim)) # nsteps x n x dim
    scale = tf.math.sqrt(2 * step_size)

    for t in range(steps-1):
      x_t = tf.identity(self.x[t, :, :])
      # calculate scores using autodiff
      with tf.GradientTape() as g:
        g.watch(x_t)
        log_prob_x = self.log_prob(x_t) # n x dim
      score = g.gradient(log_prob_x, x_t) # n x dim
      assert score.shape == (n, dim)
      
      # compute langevin update
      drive = step_size * score # n x dim
      x_next = self.x[t, :, :] + drive + scale * self.noise[t, :, :] # n x dim
      self.x[t+1, :, :].assign(x_next)
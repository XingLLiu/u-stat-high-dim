import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import trange

class MCMC:
  def __init__(self, log_prob: callable) -> None:
    self.log_prob = log_prob
    self.x = None
    self.noise_dist = tfd.Normal(0., 1.)

  def log_transition_kernel(self, xp: tf.Tensor, x: tf.Tensor):
    '''Compute log k(x'| x), where k is the transition kernel 

    Args:
      x_proposed: n x dim. Proposed samples
      x_current: n x dim. Samples at time t-1
      score: n x dim. Score evaluated at x_current

    Output:
      log_kernel: n. k(x', x)
    '''
    raise NotImplementedError(
      "Sampler class '{:s}' does not provide a 'transition kernel'".format(self._get_class_name())
    )

  def compute_accept_prob(self, x_proposed: tf.Tensor, x_current: tf.Tensor,
    log_det_jacobian: tf.Tensor, **kwargs):
    '''Compute log acceptance prob in the Metropolis step,
    log( \min(1, k(x, x') * p(x') / (k(x', x) * p(x))) )
    '''
    log_numerator = self.log_prob(x_proposed) + self.log_transition_kernel(
      xp=x_current, x=x_proposed, **kwargs
    ) + log_det_jacobian # n
    log_denominator = self.log_prob(x_current) + self.log_transition_kernel(
      xp=x_proposed, x=x_current, **kwargs
    ) # n

    log_prob = tf.math.minimum(0., log_numerator - log_denominator)
    return tf.math.exp(log_prob)

  def metropolis(self, x_proposed: tf.Tensor, x_current: tf.Tensor, accept_prob: tf.Tensor):
    '''Metropolis move according to accept_prob

    Args:
      accept_prob: n. Tensor of alpha_i, i = 1, ..., n.

    Output:
      x_next: n x dim. new particles
    '''
    unif = tf.random.uniform(shape=accept_prob.shape) # n
    # print("x_curr", x_current[:6, ])
    # print("x_prop", x_proposed[:6, ])
    # print("accept_prob", accept_prob[:6, ])
    cond = tf.expand_dims(unif < accept_prob, axis=-1) # n x 1
    x_next = tf.where(cond, x_proposed, x_current) # n x dim
    # print("x_next", x_next[:6, ])
    assert x_next.shape == x_proposed.shape
    assert tf.experimental.numpy.all(tf.where(cond, x_next == x_proposed, x_next == x_current))
    return x_next, tf.cast(cond, dtype=tf.int64)


class MALA(MCMC):
  def __init__(self, log_prob: callable) -> None:
      super().__init__(log_prob)
      self.log_prob = log_prob
      self.x = None
      self.noise_dist = tfd.Normal(0., 1.)

  def run(self, steps: int, step_size: float, x_init: tf.Tensor, verbose: bool=False):
    n, dim = x_init.shape
    self.x = tf.Variable(-1 * tf.ones((steps, n, dim))) # nsteps x n x dim
    self.x[0, :, :].assign(x_init)

    self.noise = self.noise_dist.sample((steps, n, dim)) # nsteps x n x dim
    scale = tf.math.sqrt(2 * step_size)

    # accept_thresholds = tfd.Uniform().sample((steps, n)) # nsteps x n

    self.accept_prob = 0. # n x 1
    
    iterator = trange(steps-1) if verbose else range(steps-1)

    for t in iterator:
      x_t = tf.identity(self.x[t, :, :])
      # calculate score of current samples
      with tf.GradientTape() as g:
        g.watch(x_t)
        log_prob_x = self.log_prob(x_t) # n x dim
      score = g.gradient(log_prob_x, x_t) # n x dim
      assert score.shape == (n, dim)
      
      # compute langevin update
      drive = step_size * score # n x dim
      xp_next = self.x[t, :, :] + drive + scale * self.noise[t, :, :] # n x dim

      # calculate score of proposed samples
      with tf.GradientTape() as g:
        g.watch(xp_next)
        log_prob_xp = self.log_prob(xp_next) # n x dim
      score_xp = g.gradient(log_prob_xp, xp_next) # n x dim
      assert score_xp.shape == (n, dim)

      # compute acceptance prob
      accept_prob = self.compute_accept_prob(
        x_proposed=xp_next,
        x_current=self.x[t, :, :],
        score_proposed=score_xp,
        score_current=score,
        step_size=step_size
      ) # n

      # move
      x_next, if_accept = self.metropolis(x_proposed=xp_next, x_current=self.x[t, :, :], accept_prob=accept_prob) # n x dim, n x 1
      self.accept_prob += if_accept / steps # n x 1

      # store next samples
      self.x[t+1, :, :].assign(x_next)

  def log_transition_kernel(self, xp: tf.Tensor, x: tf.Tensor, score_x: tf.Tensor, step_size: float):
    '''Compute log k(x'| x), where k is the transition kernel 
    k(x' | x) \propto \exp(- 1 / (4 * \epsilon) \| x' - x - \epsilon * score(x) \|_2^2)
    
    Args:
      x_proposed: n x dim. Proposed samples
      x_current: n x dim. Samples at time t-1
      score: n x dim. Score evaluated at x_current

    Output:
      log_kernel: n. k(x', x)
    '''
    mean = xp - x - step_size * score_x # n x dim
    mean_norm_sq = tf.reduce_sum(mean**2, axis=1) # n
    log_kernel = - 1 / (4. * step_size) * mean_norm_sq # n
    return log_kernel


class RandomWalkMH(MCMC):
  def __init__(self, log_prob: callable) -> None:
    self.log_prob = log_prob
    self.x = None
    self.dir_dist = tfp.distributions.Bernoulli(probs=0.5) # tfd.Normal(0., 1.) #! delete

  def run(self, steps: int, x_init: tf.Tensor, verbose: bool=False, **kwargs):
    n, dim = x_init.shape
    self.x = tf.Variable(-1 * tf.ones((steps, n, dim))) # nsteps x n x dim
    self.x[0, :, :].assign(x_init)

    self.directions = self.dir_dist.sample((steps-1, n, 1)) # steps-1 x n

    self.accept_prob = tf.Variable(tf.zeros((steps-1, n))) # (steps-1) x n

    iterator = trange(steps-1) if verbose else range(steps-1)

    for t in iterator: 
      # propose MH update
      x_current = self.x[t, :, :]
      xp_next, log_det_jacobian = self.proposal(x_current=x_current, noise=self.directions[t, :], **kwargs) # n x dim #! delete

      # compute acceptance prob
      accept_prob = self.compute_accept_prob(
        x_proposed=xp_next,
        x_current=x_current,
        log_det_jacobian=log_det_jacobian,
        **kwargs
      ) # n
      self.accept_prob[t, :].assign(accept_prob)
      
      # move
      x_next, _ = self.metropolis(x_proposed=xp_next, x_current=x_current, accept_prob=accept_prob) # n x dim, n x 1

      # store next samples
      self.x[t+1, :, :].assign(x_next)

  def log_transition_kernel(self, xp: tf.Tensor, x: tf.Tensor, std: float, **kwargs):
    '''Compute log k(x'| x), where k is the transition kernel 
    k(x' | x) \propto N(x' | x, std**2)
    
    Args:
      x_proposed: n x dim. Proposed samples
      x_current: n x dim. Samples at time t-1
      score: n x dim. Score evaluated at x_current

    Output:
      log_kernel: n. k(x', x)
    '''
    log_prob = tf.math.log(0.5) * tf.ones(xp.shape[0])

    return log_prob

  def proposal(self, x_current, **kwargs):
    """Default proposal:

    x_current: n x dim
    """
    noise = kwargs["noise"] # n x dim
    std = kwargs["std"]

    if "dir_vec" in kwargs:
      dir_vec = kwargs["dir_vec"] # dim
      indicator = tf.cast(noise[:, :1] > 0, dtype=tf.float32)*2 - 1. # n x 1
      xp_next = x_current + std * indicator * dir_vec # n x dim
    elif "mode1" in kwargs:
      mode1, mode2 = kwargs["mode1"], kwargs["mode2"] # dim
      root_cov_1, inv_root_cov_1 = kwargs["hess1_sqrt"], kwargs["hess1_inv_sqrt"] # dim x dim 
      root_cov_2, inv_root_cov_2 = kwargs["hess2_sqrt"], kwargs["hess2_inv_sqrt"] # dim x dim
      root_cov_1_det, inv_root_cov_1_det = kwargs["hess1_sqrt_det"], kwargs["hess1_inv_sqrt_det"]
      root_cov_2_det, inv_root_cov_2_det = kwargs["hess2_sqrt_det"], kwargs["hess2_inv_sqrt_det"]

      xp_next1 = (x_current - mode1) @ inv_root_cov_1 @ root_cov_2 + std * mode2 # n x dim
      xp_next2 = (x_current - std * mode2) @ inv_root_cov_2 @ root_cov_1 + mode1 # n x dim 

      which_next = noise == 1
      xp_next = tf.where(which_next, xp_next1, xp_next2) # n x dim
      
      n = xp_next.shape[0]
      det_next1 = tf.ones(n) * (inv_root_cov_1_det * root_cov_2_det) # n
      det_next2 = tf.ones(n) * (inv_root_cov_2_det * root_cov_1_det) # n
      det_jacobian = tf.math.abs(
        tf.where(tf.reshape(which_next, (n,)), det_next1, det_next2)
      ) # n
      log_det_jacobian = tf.math.log(det_jacobian) # n

    return xp_next, log_det_jacobian

class RandomWalkBarker(RandomWalkMH):
  def __init__(self, log_prob: callable) -> None:
    super().__init__(log_prob)

  def compute_accept_prob(self, x_proposed: tf.Tensor, x_current: tf.Tensor,
    log_det_jacobian: tf.Tensor, **kwargs):
    '''Compute log acceptance prob using the Barker's rule, 
    log( k(x, x') * p(x') / (k(x', x) * p(x) + k(x, x') * p(x')) )
    '''
    term_xp = tf.exp(
      self.log_prob(x_proposed) + self.log_transition_kernel(
        xp=x_current, x=x_proposed, **kwargs
      ) + log_det_jacobian
    ) # n
    term_x = tf.exp(
      self.log_prob(x_current) + self.log_transition_kernel(
        xp=x_proposed, x=x_current, **kwargs
      )
    ) # n
    prob = term_xp / (term_xp + term_x) # n
    return prob
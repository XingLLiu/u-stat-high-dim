import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import trange

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

    for t in trange(steps-1):
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

  def compute_accept_prob(self, x_proposed: tf.Tensor, x_current: tf.Tensor, **kwargs):
    '''Compute log acceptance prob in the Metropolis step, log( k(x, x') * p(x') / (k(x', x) * p(x)) )
    '''
    log_numerator = self.log_prob(x_proposed) + self.log_transition_kernel(
      xp=x_current, x=x_proposed, **kwargs
    )
    log_denominator = self.log_prob(x_current) + self.log_transition_kernel(
      xp=x_proposed, x=x_current, **kwargs
    )

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
    cond = tf.expand_dims(unif < accept_prob, axis=-1) # n x 1
    x_next = tf.where(cond, x_proposed, x_current) # n x dim
    assert x_next.shape == x_proposed.shape
    assert tf.experimental.numpy.all(tf.where(cond, x_next == x_proposed, x_next == x_current))
    return x_next, tf.cast(cond, dtype=tf.int64)


class MALA(Langevin):
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


class RandomWalkMH(Langevin):
  def __init__(self, log_prob: callable, proposal: callable=None) -> None:
    self.log_prob = log_prob
    self.x = None
    self.noise_dist = tfd.Normal(0., 1.)
    self.proposal = self._proposal if proposal is None else proposal

  def run(self, steps: int, x_init: tf.Tensor, verbose: bool=False, **kwargs):
    n, dim = x_init.shape
    self.x = tf.Variable(-1 * tf.ones((steps, n, dim))) # nsteps x n x dim
    self.x[0, :, :].assign(x_init)

    self.noise = self.noise_dist.sample((steps, n, dim)) # nsteps x n x dim

    self.accept_prob = tf.Variable(tf.zeros((steps-1, n))) # (steps-1) x n
    self.accept_proportion = 0. # (steps-1)

    iterator = trange(steps-1) if verbose else range(steps-1)

    for t in iterator: 
      # propose MH update
      xp_next = self.proposal(x_current=self.x[t, :, :], noise=self.noise[t, :, :], **kwargs) # n x dim

      # compute acceptance prob
      accept_prob = self.compute_accept_prob(
        x_proposed=xp_next,
        x_current=self.x[t, :, :],
        **kwargs
      ) # n

      # move
      x_next, if_accept = self.metropolis(x_proposed=xp_next, x_current=self.x[t, :, :], accept_prob=accept_prob) # n x dim, n x 1
      self.accept_proportion += if_accept / steps # n x 1
      self.accept_prob[t, :].assign(accept_prob)

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
    log_prob = tf.math.log(tf.ones(xp.shape[0]) * 0.5)
    return log_prob

  def _proposal(self, x_current, **kwargs):
    """Default proposal: x' = x + \sigma * Z, Z \sim N(0, 1)
    x_current: n x dim
    """
    noise = kwargs["noise"] # n x dim
    std = kwargs["std"]
    dim = x_current.shape[1]

    # xp_next = x_current + std * noise # n x dim

    #!
    # std_vec = tf.concat([tf.constant([std]), tf.ones(dim-1)], axis=0) # dim
    # xp_next = x_current[t, :, :] + std_vec * noise[t, :, :] # n x dim #! gaussian with var in 1st dim
    #!
    # std_vec = tf.concat([tf.constant([std]), tf.zeros(dim-1)], axis=0) # dim
    # xp_next = x_current + std_vec * (tf.cast(noise > 0, dtype=tf.float32)*2 - 1.) # n x dim #! discrete jump
    #!
    if "dir_vec" in kwargs:
      dir_vec = kwargs["dir_vec"] # dim
      indicator = tf.cast(noise[:, :1] > 0, dtype=tf.float32)*2 - 1. # n x 1
      xp_next = x_current + std * indicator * dir_vec # n x dim #! discrete jump with dir
    elif "mode1" in kwargs:
      mode1, mode2 = kwargs["mode1"], kwargs["mode2"] # dim
      hess1_sqrt, hess1_inv_sqrt = kwargs["hess1_sqrt"], kwargs["hess1_inv_sqrt"] # dim x dim 
      hess2_sqrt, hess2_inv_sqrt = kwargs["hess2_sqrt"], kwargs["hess2_inv_sqrt"] # dim x dim
      
      xp_next1 = (x_current - mode1) @ hess1_inv_sqrt @ hess2_sqrt + std * mode2 # n x dim
      xp_next2 = (x_current - std * mode2) @ hess2_inv_sqrt @ hess1_sqrt + mode1 # n x dim
      xp_next = tf.where(noise[:, :1] > 0, xp_next1, xp_next2) # n x dim
    #!
    # std_vec = tf.concat([tf.ones(2), tf.zeros(dim-2)], axis=0) # dim
    # std_vec = std * std_vec / tf.math.sqrt(tf.reduce_sum(std_vec**2))
    # xp_next = x_current[t, :, :] + std_vec * (tf.cast(noise[t, :, :] > 0, dtype=tf.float32)*2 - 1.) # n x dim #! discrete diagnal jump
    return xp_next


import autograd.numpy as anp

def create_mixture_gaussian_kdim_logprobb(dim, k, delta, ratio=0.5, shift=0.):
    """
    Evaluate the log density at the points (rows) in X 
    of the standard isotropic Gaussian.
    Note that the density is NOT normalized. 
    
    X: n x d nd-array
    return a length-n array
    """
    a = [1. if x < k else 0. for x in range(dim)]
    a = anp.array(a)
    multiplier = delta / anp.sqrt(float(k))
    mean1 = anp.zeros(dim) + shift
    mean2 = multiplier * a + shift

    log_ratio1 = anp.log(ratio)
    log_ratio2 = anp.log(1-ratio)
    
    variance = 1

    def log_prob_fn(X):
      exp1 = -0.5 * anp.sum((X-mean1)**2, axis=-1) / variance + log_ratio1
      exp2 = -0.5 * anp.sum((X-mean2)**2, axis=-1) / variance + log_ratio2
      unden = anp.logaddexp(exp1, exp2) # n
      return unden

    return log_prob_fn
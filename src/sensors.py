import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def norm2_sq(loca, locb):
    diff = loca - locb # batch x n x dim
    return tf.reduce_sum(diff**2, axis=-1) # batch x n

def dnorm(x, mean, sd, log=False):
    lkhd = (2*np.pi)**(-0.5) * sd**(-1) * tf.exp(-0.5 * (x - mean)**2 * sd**(-2))
    if not log:
        return lkhd
    else:
        return tf.math.log(lkhd)

def dnorm_log(x, mean, sd):
    ll = -0.5 * tf.math.log(2*np.pi) - tf.math.log(sd) - 0.5 * (x - mean)**2 * sd**(-2)
    return ll

def sqrt_stable(x, axis):
    res = []
    for i in range(x.shape[axis]):
        sqrt = tf.math.sqrt(tf.gather(x, i, axis=axis))
        res.append(sqrt)
    return tf.stack(res, axis=axis)
    
    
class Sensor:
    def __init__(self, Ob, Os, Xb, Xs, Yb, Ys, R=0.3, sigma=0.02):
        self.Ob = Ob
        self.Os = Os
        self.Xb = Xb
        self.Xs = Xs
        self.Yb = Yb
        self.Ys = Ys
        self.R = R
        self.sigma = sigma
        self.nb = Xb.shape[0]
        self.ns = Xs.shape[0]
        self.shift = 25. # for numerical stability

    def _loglkhd(self, loc):
        term1 = 0.
        for i in range(self.nb):
            for j in range(self.ns):
                # location term
                norm_sq = norm2_sq(self.Xb[i, :], tf.gather(loc, range(2*j, 2 * j +2), axis=-1)) # batch x n
                tt1 = -norm_sq / (2*self.R**2) * self.Ob[j, i] # batch x n
                tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2))*(1 - self.Ob[j, i]) # batch x n
                term1 += tt1 + tt2

                # distance term
                norm = tf.math.sqrt(norm_sq)
                tt = dnorm_log(
                    self.Yb[j, i], 
                    mean=norm,
                    sd=self.sigma) * self.Ob[j, i] # batch x n
                term1 += tt

        term2 = 0.
        for i in range(self.ns):
            for j in range(i+1, self.ns):
                # location term
                norm_sq = norm2_sq(
                    tf.gather(loc, range(2*i, 2*i+2), axis=-1),
                    tf.gather(loc, range(2*j, 2*j+2), axis=-1)) # batch x n
                tt1 = -norm_sq / (2*self.R**2) * self.Os[i, j] # batch x n
                tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2))*(1 - self.Os[i, j]) # batch x n
                term2 += tt1 + tt2

                # distance term
                tt = dnorm_log(
                    self.Ys[i, j],
                    mean=tf.math.sqrt(norm_sq),
                    sd=self.sigma) * self.Os[i, j] # batch x n
                term2 += tt

        loglkhd = term1 + term2
        return loglkhd

    def log_prob(self, loc):
        loglkhd = self._loglkhd(loc)
        prior = tf.reduce_sum(
            dnorm_log(loc, mean=tf.zeros((8,)), sd=10.*tf.ones((8,))),
            axis=-1) # batch x n
        return loglkhd + prior + self.shift # batch x n

class SensorImproper(Sensor):

    def __init__(self, Ob, Os, Xb, Xs, Yb, Ys, R=0.3, sigma=0.02):
        super().__init__(Ob, Os, Xb, Xs, Yb, Ys, R, sigma)

    def log_prob(self, loc):
        loglkhd = self._loglkhd(loc)
        return loglkhd + self.shift



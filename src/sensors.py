import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def norm2(loca, locb):
    diff = loca - locb # batch x n x dim
    return tf.math.sqrt(tf.reduce_sum(diff**2, axis=-1)) # batch x n

def dnorm(x, mean, sd, log=False):
    lkhd = (2*np.pi)**(-0.5) * sd**(-1) * tf.exp(-0.5 * (x - mean)**2 * sd**(-2))
    if not log:
        return lkhd
    else:
        return tf.math.log(lkhd)

def dnorm_log(x, mean, sd):
    ll = -0.5 * tf.math.log(2*np.pi) - tf.math.log(sd) - 0.5 * (x - mean)**2 * sd**(-2)
    return ll
    
    
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
        self.shift = 25.

    def _loglkhd(self, loc):
        ## observed sensors
        loc_exp = tf.reshape(loc, loc.shape[:-1] + [self.ns, 1, 2]) # batch x n x ns x 1 x 2
        # observation term
        norm = norm2(
            tf.expand_dims(self.Xb, axis=0), # 1 x nb x 2
            loc_exp
        ) # batch x n x nb x ns
        norm_sq = norm**2 # batch x n x ns x nb
        tt1 = -norm_sq / (2*self.R**2) * self.Ob # batch x n x ns x nb
        tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2)) * (1 - self.Ob) # batch x n x ns x nb
        tt_obs = tt1 + tt2

        # distance term
        tt_dist = dnorm_log(
            self.Yb, 
            mean=norm,
            sd=self.sigma) * self.Ob # batch x n x ns x nb
        
        term1_all = tf.reshape(tt_obs + tt_dist, tt_obs.shape[:-2] + [self.ns*self.nb]) # batch x n x (ns * nb)
        
        ## non-observed sensors
        loc_exp1 = tf.reshape(loc, loc.shape[:-1] + [1, self.ns, 2]) # batch x n x 1 x ns x 2
        loc_exp2 = tf.reshape(loc, loc.shape[:-1] + [self.ns, 1, 2]) # batch x n x ns x 1 x 2
        norm = norm2(loc_exp1, loc_exp2) # batch x n x ns x ns
        norm_sq = norm**2 # batch x n x ns x ns
        # jitter; does not contribute to computation since only entries in upper triangular
        # area is used
        norm_sq += 1e-18 * tf.eye(self.ns)

        # observation term
        tt1 = -norm_sq / (2*self.R**2) * self.Os # batch x n x ns x ns
        tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2)) * (1 - self.Os) # batch x n x ns x ns
        term2 = tt1 + tt2
        assert term2.shape == loc.shape[:-1] + [self.ns, self.ns]

        # distance term
        tt_dist = dnorm_log(
            self.Ys,
            mean=norm,
            sd=self.sigma) * self.Os # batch x n x ns x ns
        assert tt_dist.shape == loc.shape[:-1] + [self.ns, self.ns]

        # select only (i,j)-entries where 1 <= i < j <= ns
        pad = tf.experimental.numpy.triu(tf.ones(tt_dist.shape), k=1) # batch x n x ns x ns
        term2_all = tf.reshape((term2 + tt_dist) * pad, term2.shape[:-2] + [self.ns**2]) # batch x n x ns**2
        
        # combine
        terms_concat = tf.concat([term1_all, term2_all], axis=-1) # batch x n x (ns**2 + ns * nb)

        loglkhd = tf.reduce_sum(terms_concat, axis=-1) # batch x n
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



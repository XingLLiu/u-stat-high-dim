import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def norm2(loca, locb):
    diff = loca - locb # n x dim
    return tf.math.sqrt(tf.reduce_sum(diff**2, axis=-1))

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

    def log_prob(self, loc):
        n = loc.shape[0]

        term1 = []
        for i in range(2):
            for j in range(4):
                norm_sq = norm2(self.Xb[i, :], loc[:, (2 * j) : (2 * j +2)])**2
                tt1 = -norm_sq / (2*self.R**2) * self.Ob[j, i]
                tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2))*(1 - self.Ob[j, i])
                term1.append(tt1 + tt2)

        term2 = []
        for i in range(3):
            for j in range(i+1, 4):
                norm_sq = norm2(
                    loc[:, (2 * i):(2 * i +2)], loc[:, (2 * j) : (2 * j +2)])**2
                tt1 = -norm_sq / (2*self.R**2) * self.Os[i, j]
                tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2))*(1 - self.Os[i, j])
                term2.append(tt1 + tt2)

        term1_obs = []
        for i in range(2):
            for j in range(4):
                tt = dnorm_log(self.Yb[j, i], 
                           mean = norm2(self.Xb[i, ], loc[:, (2 * j):(2 * j +2)]),
                           sd = self.sigma)*self.Ob[j, i]
                term1_obs.append(tt)

        term2_obs = []
        for i in range(3):
            for j in range(i+1, 4):
                tt = dnorm_log(self.Ys[i, j],
                           mean = norm2(loc[:, (2 * i):(2 * i +2)], loc[:, (2 * j) : (2 * j +2)]),
                           sd = self.sigma)*self.Os[i, j]
                term2_obs.append(tt)
                # print(self.Ys[i, j], norm2(loc[:, (2 * i):(2 * i +2)], loc[:, (2 * j) : (2 * j +2)]), self.Os[i, j], tt)

        term1 = tf.stack(term1, axis=1) # n x 8
        term2 = tf.stack(term2, axis=1) # n x 6
        term1_obs = tf.stack(term1_obs, axis=1) # n x 8
        term2_obs = tf.stack(term2_obs, axis=1) # n x 6
        # print(term2_obs)

        terms_concat = tf.concat([term1, term2, term1_obs, term2_obs], axis=-1) # n x 28

        loglkhd = tf.reduce_sum(terms_concat, axis=-1)
        prior = tf.reduce_sum(
            dnorm_log(loc, mean = tf.zeros((8,)), sd = 10.*tf.ones((8,))),
            axis=-1)
        # print("prior", prior)
        post = loglkhd + prior
        return post + 25.
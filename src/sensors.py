import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# def norm2(loca, locb):
#     diff = loca - locb # batch x n x dim
#     print("normeq", tf.reduce_sum(diff**2, axis=-1))
#     return tf.math.sqrt(tf.reduce_sum(diff**2, axis=-1)) # batch x n

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
        self.shift = 25.

    def _loglkhd(self, loc):
        ## observed sensors
        loc_exp = tf.reshape(loc, list(loc.shape[:-1]) + [self.ns, 1, 2]) # batch x n x ns x 1 x 2
        # observation term
        # norm = norm2(
        #     tf.expand_dims(self.Xb, axis=0), # 1 x nb x 2
        #     loc_exp
        # ) # batch x n x nb x ns
        # norm_sq = norm**2 # batch x n x ns x nb
        norm_sq = norm2_sq(
            tf.expand_dims(self.Xb, axis=0), # 1 x nb x 2
            loc_exp
        ) # batch x n x ns x nb
        norm = sqrt_stable(norm_sq, axis=-1)
        # print("norm_sq", norm_sq)
        # print("norm", norm)
        tt1 = -norm_sq / (2*self.R**2) * self.Ob # batch x n x ns x nb
        tt2 = tfp.substrates.numpy.math.log1mexp(norm_sq / (2*self.R**2)) * (1 - self.Ob) # batch x n x ns x nb
        # print("inner", norm_sq / (2*self.R**2))
        # print("tt1", tt1, "tt2", tt2)
        # print("tt2", tt2)
        # print("manual", tfp.math.log1mexp((norm_sq / (2*self.R**2))[:, -1, -1]))
        tt_obs = tt1 + tt2
        # print("term1_both", tf.reshape(tf.transpose(tt_obs), (1, 8)))

        # distance term
        tt_dist = dnorm_log(
            self.Yb, 
            mean=norm,
            sd=self.sigma) * self.Ob # batch x n x ns x nb
        
        term1_all = tf.reshape(tt_obs + tt_dist, tt_obs.shape[:-2] + [self.ns*self.nb]) # batch x n x (ns * nb)
        # print("term1", tf.reshape(tf.transpose(tt_dist), tt_obs.shape[:-2] + [self.ns*self.nb])[-1])
        # print("term1", tf.reduce_sum(term1_all, -1))
        # print("term1", tf.reduce_sum(tf.reshape(tt_obs, tt_obs.shape[:-2] + [self.ns*self.nb]), -1))
        
        ## non-observed sensors
        loc_exp1 = tf.reshape(loc, list(loc.shape[:-1]) + [1, self.ns, 2]) # batch x n x 1 x ns x 2
        loc_exp2 = tf.reshape(loc, list(loc.shape[:-1]) + [self.ns, 1, 2]) # batch x n x ns x 1 x 2
        # norm = norm2(loc_exp1, loc_exp2) # batch x n x ns x ns
        # norm_sq = norm**2 # batch x n x ns x ns
        norm_sq = norm2_sq(loc_exp1, loc_exp2) # batch x n x ns x ns
        norm = sqrt_stable(norm_sq, axis=-2) # batch x n x ns x ns
        # jitter; does not contribute to computation since only entries in upper triangular
        # area is used
        norm_sq += 1 * tf.eye(self.ns)

        # observation term
        tt1 = -norm_sq / (2*self.R**2) * self.Os # batch x n x ns x ns
        tt2 = tfp.substrates.numpy.math.log1mexp(norm_sq / (2*self.R**2)) * (1 - self.Os) # batch x n x ns x ns
        term2 = tt1 + tt2
        assert term2.shape == list(loc.shape[:-1]) + [self.ns, self.ns]

        # distance term
        tt_dist = dnorm_log(
            self.Ys,
            mean=norm,
            sd=self.sigma) * self.Os # batch x n x ns x ns
        assert tt_dist.shape == list(loc.shape[:-1]) + [self.ns, self.ns]

        # select only (i,j)-entries where 1 <= i < j <= ns
        pad = tf.experimental.numpy.triu(tf.ones(tt_dist.shape), k=1) # batch x n x ns x ns
        term2_all = tf.reshape((term2 + tt_dist) * pad, term2.shape[:-2] + [self.ns**2]) # batch x n x ns**2
        
        # combine
        terms_concat = tf.concat([term1_all, term2_all], axis=-1) # batch x n x (ns**2 + ns * nb)

        loglkhd = tf.reduce_sum(terms_concat, axis=-1) # batch x n
        return loglkhd



        # ## observed sensors
        # loc_exp = tf.reshape(loc, list(loc.shape[:-1]) + [self.ns, 1, 2]) # batch x n x ns x 1 x 2
        # term1_all = 0.
        # # observation term
        # norm_sq = norm2_sq(
        #     tf.expand_dims(self.Xb, axis=0), # 1 x nb x 2
        #     loc_exp
        # ) # batch x n x ns x nb
        # # norm_sq = tf.reshape(
        # #     tf.transpose(norm_sq), 
        # #     list(norm_sq.shape[:-2]) + [self.nb * self.ns]) # batch x n x (nb*ns)
        # tt1 = -norm_sq / (2*self.R**2) * self.Ob # batch x n x ns x nb
        # tt2 = tfp.substrates.numpy.math.log1mexp(norm_sq / (2*self.R**2)) * (1 - self.Ob) # batch x n x ns x nb
        # tt_obs = tt1 + tt2
        # tt_obs = tf.reduce_sum(tt_obs, axis=-1) # batch x n x (ns * nb)

        # tt_dist = 0.
        # for i in range(self.nb):
        #     # distance term
        #     norm = tf.math.sqrt(tf.gather(norm_sq, i, axis=-1))
        #     # print("norm", norm)
        #     tt3 = dnorm_log(
        #         self.Yb[:, i], 
        #         mean=norm,
        #         sd=self.sigma) * self.Ob[:, i] # batch x n x ns
        #     tt_dist += tt3
        #     # print("tt3", tt3)
            
        # term1_all = tf.concat([tt_obs, tt_dist], axis=-1) # batch x n x ...
        # term1_all = tf.reduce_sum(term1_all, axis=-1) # batch x n
        # # print("term1_both", term1_all)
        
        # ## non-observed sensors
        # loc_exp1 = tf.reshape(loc, list(loc.shape[:-1]) + [1, self.ns, 2]) # batch x n x 1 x ns x 2
        # loc_exp2 = tf.reshape(loc, list(loc.shape[:-1]) + [self.ns, 1, 2]) # batch x n x ns x 1 x 2
        # # norm = norm2(loc_exp1, loc_exp2) # batch x n x ns x ns
        # # norm_sq = norm**2 # batch x n x ns x ns
        # norm_sq = norm2_sq(loc_exp1, loc_exp2) # batch x n x ns x ns
        # # jitter; does not contribute to computation since only entries in upper triangular
        # # area is used
        # norm_sq += 1 * tf.eye(self.ns)

        # # observation term
        # tt1 = -norm_sq / (2*self.R**2) * self.Os # batch x n x ns x ns
        # tt2 = tfp.substrates.numpy.math.log1mexp(norm_sq / (2*self.R**2)) * (1 - self.Os) # batch x n x ns x ns
        # # select only (i,j)-entries where 1 <= i < j <= ns
        # pad = tf.experimental.numpy.triu(tf.ones((self.ns, self.ns)), k=1) # batch x n x ns x ns
        # term2 = (tt1 + tt2) * pad
        # # print("term2", term2)
        # term2 = tf.reshape(term2, list(term2.shape[:-2]) + [self.ns * self.ns])
        # # assert term2.shape == list(loc.shape[:-1]) + [self.ns, self.ns]

        # # distance term
        # tt_dist = []
        # for i in range(self.ns):
        #     norm = tf.math.sqrt(tf.gather(norm_sq, i, axis=-1)) # batch x n x ns
        #     tt3 = dnorm_log(
        #         self.Ys[i],
        #         mean=norm,
        #         sd=self.sigma) * self.Os[i] # batch x n x ns
        #     tt_dist.append(tf.gather(tt3, range(i+1, self.ns), axis=-1))

        # term2_dist = tf.concat(tt_dist, axis=-1) # batch x n x ((ns-1)*ns/2)
        # # print("term2", term2_dist)
        
        # term2_all = tf.concat([term2, term2_dist], axis=-1) # batch x n x ...
        # term2_all = tf.reduce_sum(term2_all, axis=-1) # batch x n
        # # print("term2_both", term2_all)

        # # combine
        # loglkhd = term1_all + term2_all # batch x n
        # return loglkhd

    # def _loglkhd(self, loc):
    #     term1 = []
    #     term1_obs = []
    #     term1 = 0.
    #     for i in range(self.nb):
    #         for j in range(self.ns):
    #             norm_sq = norm2_sq(self.Xb[i, :], tf.gather(loc, range(2*j, 2 * j +2), axis=-1)) # batch x n
    #             tt1 = -norm_sq / (2*self.R**2) * self.Ob[j, i] # batch x n
    #             tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2))*(1 - self.Ob[j, i]) # batch x n
    #             # term1.append(tt1 + tt2)
    #             term1 += tt1 + tt2

    #             norm = tf.math.sqrt(norm_sq)
    #             tt = dnorm_log(
    #                 self.Yb[j, i], 
    #                 mean=norm,
    #                 sd=self.sigma) * self.Ob[j, i] # batch x n
    #             # term1_obs.append(tt)
    #             term1 += tt

    #     term2 = []
    #     term2_obs = []
    #     term2 = 0.
    #     for i in range(self.ns):
    #         for j in range(i+1, self.ns):
    #             norm_sq = norm2_sq(
    #                 tf.gather(loc, range(2*i, 2*i+2), axis=-1),
    #                 tf.gather(loc, range(2*j, 2*j+2), axis=-1)) # batch x n
    #             tt1 = -norm_sq / (2*self.R**2) * self.Os[i, j] # batch x n
    #             tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2))*(1 - self.Os[i, j]) # batch x n
    #             # term2.append(tt1 + tt2)
    #             term2 += tt1 + tt2

    #             tt = dnorm_log(
    #                 self.Ys[i, j],
    #                 mean=tf.math.sqrt(norm_sq),
    #                 sd=self.sigma) * self.Os[i, j] # batch x n
    #             # term2_obs.append(tt)
    #             term2 += tt

    #     # term1 = tf.stack(term1, axis=-1) # batch x n x 8
    #     # term2 = tf.stack(term2, axis=-1) # batch x n x 6
    #     # term1_obs = tf.stack(term1_obs, axis=-1) # batch x n x 8
    #     # term2_obs = tf.stack(term2_obs, axis=-1) # batch x n x 6
    #     # terms_concat = tf.concat([term1, term2, term1_obs, term2_obs], axis=-1) # batch x n x 28

    #     # loglkhd = tf.reduce_sum(terms_concat, axis=-1) # batch x n
    #     loglkhd = term1 + term2
    #     return loglkhd

    def log_prob2(self, loc):
        loglkhd = self._loglkhd(loc)
        prior = tf.reduce_sum(
            dnorm_log(loc, mean=tf.zeros((8,)), sd=10.*tf.ones((8,))),
            axis=-1) # batch x n
        return loglkhd + prior + self.shift # batch x n

    def log_prob(self, loc):
        term1 = []
        for i in range(2):
            for j in range(4):
                # norm_sq = norm2(self.Xb[i, :], tf.gather(loc, range(2*j, 2 * j +2), axis=-1))**2 # batch x n
                norm_sq = norm2_sq(self.Xb[i, :], tf.gather(loc, range(2*j, 2 * j +2), axis=-1)) # batch x n
                # print(j, i, "norm_sq", norm_sq)
                # print(j, i, "norm", )
                tt1 = -norm_sq / (2*self.R**2) * self.Ob[j, i] # batch x n
                tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2))*(1 - self.Ob[j, i]) # batch x n
                term1.append(tt1 + tt2)
                # print(j, i, "inner", norm_sq / (2*self.R**2))
                # print(j, i, "tt1", tt1, "tt2", tt2)
                # print(j, i, "tt2", tt2)

        term2 = []
        for i in range(3):
            for j in range(i+1, 4):
                # norm_sq = norm2(
                #     tf.gather(loc, range(2*i, 2*i+2), axis=-1),
                #     tf.gather(loc, range(2*j, 2*j+2), axis=-1))**2 # batch x n
                norm_sq = norm2_sq(
                    tf.gather(loc, range(2*i, 2*i+2), axis=-1),
                    tf.gather(loc, range(2*j, 2*j+2), axis=-1)) # batch x n
                tt1 = -norm_sq / (2*self.R**2) * self.Os[i, j] # batch x n
                tt2 = tfp.math.log1mexp(norm_sq / (2*self.R**2))*(1 - self.Os[i, j]) # batch x n
                term2.append(tt1 + tt2)

        term1_obs = []
        for i in range(2):
            for j in range(4):
                norm = tf.math.sqrt(norm2_sq(self.Xb[i], tf.gather(loc, range(2*j, 2*j+2), axis=-1)))
                # print("norm", norm)
                tt = dnorm_log(
                    self.Yb[j, i], 
                    mean=norm,
                    sd=self.sigma) * self.Ob[j, i] # batch x n
                # print("tt3", tt)
                term1_obs.append(tt)

        term2_obs = []
        for i in range(3):
            for j in range(i+1, 4):
                tt = dnorm_log(
                    self.Ys[i, j],
                    mean=tf.math.sqrt(norm2_sq(
                        tf.gather(loc, range(2*i, 2*i+2), axis=-1),
                        tf.gather(loc, range(2*j, 2*j+2), axis=-1))),
                    sd=self.sigma) * self.Os[i, j] # batch x n
                term2_obs.append(tt)


        term1 = tf.stack(term1, axis=-1) # batch x n x 8
        term2 = tf.stack(term2, axis=-1) # batch x n x 6
        term1_obs = tf.stack(term1_obs, axis=-1) # batch x n x 8
        term2_obs = tf.stack(term2_obs, axis=-1) # batch x n x 6

        terms_concat = tf.concat([term1, term2, term1_obs, term2_obs], axis=-1) # batch x n x 28
        # print("term1", term1_obs)
        # print("term1_both", term1 + term1_obs)
        # print("term1_both", tf.reduce_sum(tf.concat([term1, term1_obs], axis=-1), axis=-1))

        # print("term2", term2)
        # print("term2", term2_obs)
        # print("term2_both", tf.reduce_sum(tf.concat([term2, term2_obs], axis=-1), axis=-1))

        loglkhd = tf.reduce_sum(terms_concat, axis=-1) # batch x n
        prior = tf.reduce_sum(
            dnorm_log(loc, mean=tf.zeros((8,)), sd=10.*tf.ones((8,))),
            axis=-1) # batch x n
        post = loglkhd + prior # batch x n
        # print("loglkhd", loglkhd)
        # print("prior", prior)
        return post + self.shift

class SensorImproper(Sensor):

    def __init__(self, Ob, Os, Xb, Xs, Yb, Ys, R=0.3, sigma=0.02):
        super().__init__(Ob, Os, Xb, Xs, Yb, Ys, R, sigma)

    def log_prob(self, loc):
        loglkhd = self._loglkhd(loc)
        return loglkhd + self.shift



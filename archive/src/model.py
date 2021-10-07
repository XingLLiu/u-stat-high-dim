import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np

class Model:
    def __init__(self, log_gamma=None, proposal=None, tag=0, update=True):
        self.children = []
        self.log_gamma = log_gamma
        self.proposal = proposal
        self.seq = [tag]
        self.tag = tag
        self.update = update
        
    def build_child(self, log_gamma, proposal, tag, update=True):
        self.children.append(Model(log_gamma, proposal, tag, update))
        self.seq.append(tag)
        
    def __str__(self):
        return "nodes:" + str(self.seq)

    def get_successor_indices(self):
        ls = [self.tag]
        if not self.children:
            return ls
        else:
            for child in self.children:
                # ls += [child.tag]
                ls += child.get_successor_indices()
            return ls


#! matrix multiplication is wrong!
class LinearGaussian:
    def __init__(self, sigma0, A, B, C, D):
        self.sigma0 = sigma0
        self.mu0 = np.array(0.)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
    def sample(self, nsamples, T):
        """sample x_i for i=1:(n+1) and y_i for i=1:n
        """
        x_vec = np.zeros((T+1, 1))
        y_vec = np.zeros((T, nsamples))
        
        # sample x0 from prior
        x_vec[0, :] = tfd.Normal(loc=self.mu0, scale=self.sigma0).sample(1)
        # sample iteratively
        for i in range(T):
            x_new, y_new = self.sample_once(x_vec[i, :], nsamples, i)
            x_vec[i+1, :] = x_new
            y_vec[i, :] = y_new
        
        return x_vec, y_vec
    
    def update_sample(self, x_vec, y_vec, T):
        """sample n more data
        """
        T_old, nsamples = y_vec.shape
        x_vec = np.append(x_vec, np.zeros((T, 1)), axis=0)
        y_vec = np.append(y_vec, np.zeros((T, nsamples)), axis=0)
        for i in range(T):
            x_new, y_new = self.sample_once(x_vec[i+T_old, :], nsamples, i)
            x_vec[i+1+T_old, :] = x_new
            y_vec[i+T_old, :] = y_new
        return x_vec, y_vec
            
    def sample_once(self, xt, nsamples, t):
        """sample for one time step
        """
        x_new = tf.math.multiply(self.A, xt) + \
            tf.math.multiply(self.B, tfd.Normal(loc=0, scale=1).sample(1)).numpy()
        y_new = tf.math.multiply(self.C, x_new) + \
            tf.math.multiply(self.D, tfd.Normal(loc=0, scale=1).sample(nsamples)).numpy()
        return x_new, y_new
            
    def log_gamma(self, x_vec, y_vec):
        """unnormalized log posterior
        """
        if x_vec.shape[0] == 1:
            log_p = tfd.Normal(loc=self.mu0, scale=self.sigma0).log_prob(x_vec[0, :])
            return log_p
        else:
            log_gamma = 0
            T, nsamples = y_vec.shape

            latent_mean = tf.math.multiply(self.C, x_vec[1:, :])[:, np.newaxis, :]
            log_lik = 0
            for t in range(T):
                log_lik += tfd.Normal(
                    loc=latent_mean[t, :], 
                    scale=self.D
                ).log_prob(y_vec[t, :].reshape(-1, 1))

            log_p = tfd.Normal(loc=self.mu0, scale=self.sigma0).log_prob(x_vec[0, :])
            for i in range(1, T):
                log_p += tfd.Normal(
                    loc=tf.math.multiply(self.A, x_vec[i-1, :]), 
                    scale=self.B
                ).log_prob(x_vec[i, :])

            return log_p + tf.reduce_sum(log_lik, axis=0)


#! matrix multiplication is wrong!
class NonLinearGaussian(LinearGaussian):
    def __init__(self, sigma0, mu0, sigma_v, sigma_w):
        self.sigma0 = sigma0
        self.mu0 = mu0
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
            
    def _latent_mean(self, xt, t):
        return 0.5 * xt + 25 * xt / (1 + xt**2) + 8 * np.cos(1.2 * t)
    
    def sample_once(self, xt, nsample, t):
        x_new = self._latent_mean(xt, t) + \
            tfd.Normal(loc=0, scale=self.sigma_v).sample(1).numpy()
        y_new = 1/20 * x_new**2 + \
            tfd.Normal(loc=0, scale=self.sigma_w).sample(nsample).numpy()
        y_new = y_new[:, 0, 0]
        return x_new, y_new
            
    def log_gamma(self, x_vec, y_vec):
        """unnormalized log posterior
        """
        if x_vec.shape[0] == 1:
            log_p = tfd.Normal(loc=self.mu0, scale=self.sigma0).log_prob(x_vec[0, :])
            return log_p
        else:
            log_gamma = 0
            T, nsamples = y_vec.shape
            
        log_lik = 0
        for t in range(T):
            log_lik += tfd.Normal(
                loc=1/20 * x_vec[t+1, :]**2, 
                scale=self.sigma_w
            ).log_prob(y_vec[t, :].reshape(-1, 1))
            
        log_p = tfd.Normal(loc=self.mu0, scale=self.sigma0).log_prob(x_vec[0, :])
        for i in range(1, T):
            log_p += tfd.Normal(
                loc=self._latent_mean(x_vec[i-1, :], i), 
                scale=self.sigma_v
            ).log_prob(x_vec[i, :])
        return log_p + tf.reduce_sum(log_lik, axis=0)
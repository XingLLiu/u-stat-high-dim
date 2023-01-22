import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()

import sys
sys.path.append(".")
from src.utils import resample

def dc_smc(samples, weights, model):
    
    if not model.children:
        if model.update:
            xt = model.proposal.sample((samples.shape[0], 1))
        else:
            xt = samples[:, model.tag]
        
        # update samples
        samples[:, model.tag].assign(xt.reshape((-1,)))
        # compute weight
        log_w = model.log_gamma(samples) - model.proposal.log_prob(samples[:, model.tag].reshape((-1, 1)))
    
        return samples, log_w
    
    else:
        for child in model.children:
            samples, log_w = dc_smc(samples, weights, child)
            xc = samples[:, child.tag]

            # resample if needed
            # if child.update:
            #     xc_resamp = resample(log_w, xc)
            # else: 
            #     xc_resamp = xc

            # update samples
            samples[:, child.tag].assign(xc.reshape((-1,)))

            # #? resample whenever
            # xc_resamp = resample(log_w, xc)
            successor_indices = child.get_successor_indices()
            xc_resampled = resample(log_w, tf.gather(samples, successor_indices, axis=1))
            for j, c_ind in enumerate(successor_indices):
                samples[:, c_ind].assign(xc_resampled[:, j])
            
        
        # calculate mean based on values on children nodes
        children_mean = 0
        for child in model.children:
            children_mean += samples[:, child.tag] / len(model.children)

        # draw samples from proposal
        if model.update:
            # xt = model.proposal.sample((samples.shape[0], 1), samples)
            xt = model.proposal.sample((samples.shape[0], 1), children_mean)
            # xt = model.proposal.sample((samples.shape[0], 1))
        else:
            xt = samples[:, model.tag]

        samples[:, model.tag].assign(xt.reshape((-1,)))

        # compute weights
        log_w = model.log_gamma(samples)
        for c in model.children:
            log_w -= c.log_gamma(samples)

        #! argument to proposal.log_prob is hard-coded
        # log_w -= model.proposal.log_prob(tf.reshape(xt, (-1, 1)), samples)
        log_w -= model.proposal.log_prob(xt.reshape((-1, 1)), children_mean.reshape(-1, 1))
        # log_w -= model.proposal.log_prob(xt.reshape((-1, 1)))

        return samples, log_w
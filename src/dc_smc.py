import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import sys
sys.path.append(".")
from src.utils import resample

# def dc_smc(samples, weights, gamma_curr, gamma_child, proposal, leaf_flag=0):
    
#     if leaf_flag == 1:
#         xt = proposal.sample(weights.shape, None)
#         log_w = gamma_curr(xt) - proposal.log_prob(xt, None)
#         return xt, log_w
#     else:
#         # resample
#         mu_pos_resampled = resample(log_alpha, mu_pos)
#         # draw samples from proposal
#         xt = proposal.sample(weights.shape, samples)
#         xt_new = tf.concat([samples, xt], axis=1)
#         log_w = gamma_curr(xt) - gamma_child(samples) - proposal.log_prob(xt, samples)
#         return xt, log_w

def dc_smc(samples, weights, model):
    
    if model.left is None:
        xt = model.proposal.sample(samples.shape, None)
        log_w = model.log_gamma(xt) - model.proposal.log_prob(xt, None)
        return xt, log_w
    else:
        xc_left, log_w_left = dc_smc(samples, weights, model.left)
        xc_right, log_w_right = dc_smc(samples, weights, model.right)
        
        # resample
        xc_left_resamp = resample(
            tf.reshape(log_w_left[:, model.left.tag], [-1, 1]), 
            tf.reshape(xc_left[:, model.left.tag], [-1, 1])
        )
        xc_right_resamp = resample(
            tf.reshape(log_w_right[:, model.right.tag], [-1, 1]), 
            tf.reshape(xc_right[:, model.right.tag], [-1, 1])
        )
        
        # update samples
        samples = samples.numpy()
        samples[:, model.left.tag] = tf.reshape(xc_left_resamp, (-1,))
        samples[:, model.right.tag] = tf.reshape(xc_right_resamp, (-1,))
        samples = tf.convert_to_tensor(samples)
        
        
        # draw samples from proposal
        xt = model.proposal.sample(samples.shape, samples)[:, model.tag]
        samples = samples.numpy()
        samples[:, model.tag] = xt
        samples = tf.convert_to_tensor(samples)
#         xt_new = tf.concat([samples, xt], axis=1)
        log_w = model.log_gamma(samples)
        log_w -= model.left.log_gamma(samples)
        log_w -= model.right.log_gamma(samples) 
        log_w -= model.proposal.log_prob(xt, samples)
        return samples, log_w
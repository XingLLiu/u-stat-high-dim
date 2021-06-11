import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import sys
sys.path.append(".")
from src.utils import resample


# def dc_smc(samples, weights, model):
    
#     if model.left is None:
#         if model.update:
#             xt = model.proposal.sample((samples.shape[0], 1), None)
#         else:
#             xt = tf.gather(samples, [model.tag], axis=1)
#         # update samples
#         samples = samples.numpy()
#         samples[:, model.tag] = tf.reshape(xt, (-1,))
#         samples = tf.convert_to_tensor(samples)
#         # compute weight
#         log_w = model.log_gamma(samples) - model.proposal.log_prob(tf.reshape(samples[:, model.tag], (-1, 1)), None)
#         return samples, log_w
#     else:
#         samples, log_w_left = dc_smc(samples, weights, model.left)
#         xc_left = tf.gather(samples, [model.left.tag], axis=1)
#         samples, log_w_right = dc_smc(samples, weights, model.right)
#         xc_right = tf.gather(samples, [model.right.tag], axis=1)
        
#         # resample if needed
#         if model.left.update:
#             xc_left_resamp = resample(log_w_left, xc_left)
#         else: 
#             xc_left_resamp = xc_left

#         if model.right.update:
#             xc_right_resamp = resample(log_w_right, xc_right)
#         else:
#             xc_right_resamp = xc_right
        
#         # update samples
#         samples = samples.numpy()
#         samples[:, model.left.tag] = tf.reshape(xc_left_resamp, (-1,))
#         samples[:, model.right.tag] = tf.reshape(xc_right_resamp, (-1,))
#         samples = tf.convert_to_tensor(samples)
        
#         if model.update:
#             # draw samples from proposal
#             xt = model.proposal.sample((samples.shape[0], 1), samples)
#         else:
#             xt = tf.gather(samples, [model.tag], axis=1)
#         samples = samples.numpy()
#         samples[:, model.tag] = tf.reshape(xt, (-1,))
#         samples = tf.convert_to_tensor(samples)

#         # compute weights
#         log_w = model.log_gamma(samples)
#         log_w -= model.left.log_gamma(samples)
#         log_w -= model.right.log_gamma(samples) 
#         log_w -= model.proposal.log_prob(tf.reshape(xt, (-1, 1)), samples)
#         return samples, log_w

def dc_smc(samples, weights, model):
    
    if not model.children:
        if model.update:
            xt = model.proposal.sample((samples.shape[0], 1), None)
        else:
            xt = tf.gather(samples, [model.tag], axis=1)
        # update samples
        samples = samples.numpy()
        samples[:, model.tag] = tf.reshape(xt, (-1,))
        samples = tf.convert_to_tensor(samples)
        # compute weight
        log_w = model.log_gamma(samples) - model.proposal.log_prob(tf.reshape(samples[:, model.tag], (-1, 1)), None)
        return samples, log_w
    else:
        for child in model.children:
            samples, log_w = dc_smc(samples, weights, child)
            xc = tf.gather(samples, [child.tag], axis=1)

            # resample if needed
            if child.update:
                xc_resamp = resample(log_w, xc)
            else: 
                xc_resamp = xc

            # update samples
            samples = samples.numpy()
            samples[:, child.tag] = tf.reshape(xc_resamp, (-1,))
            samples = tf.convert_to_tensor(samples)
        
        if model.update:
            # draw samples from proposal
            xt = model.proposal.sample((samples.shape[0], 1), samples)
        else:
            xt = tf.gather(samples, [model.tag], axis=1)
        samples = samples.numpy()
        samples[:, model.tag] = tf.reshape(xt, (-1,))
        samples = tf.convert_to_tensor(samples)

        # compute weights
        log_w = model.log_gamma(samples)
        for child in model.children:
            log_w -= child.log_gamma(samples)
        log_w -= model.proposal.log_prob(tf.reshape(xt, (-1, 1)), samples)
        return samples, log_w
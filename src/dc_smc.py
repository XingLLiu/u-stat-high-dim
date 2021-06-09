import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import sys
sys.path.append(".")
from src.utils import resample

# def dc_smc(samples, weights, model):
    
#     if model.left is None:
#         xt = model.proposal.sample(samples.shape, None)
#         log_w = model.log_gamma(xt) - model.proposal.log_prob(xt, None)
#         return xt, log_w
#     else:
#         xc_left, log_w_left = dc_smc(samples, weights, model.left)
#         xc_right, log_w_right = dc_smc(samples, weights, model.right)
        
#         # resample
#         xc_left_resamp = resample(
#             tf.reshape(log_w_left[:, model.left.tag], [-1, 1]), 
#             tf.reshape(xc_left[:, model.left.tag], [-1, 1])
#         )
#         xc_right_resamp = resample(
#             tf.reshape(log_w_right[:, model.right.tag], [-1, 1]), 
#             tf.reshape(xc_right[:, model.right.tag], [-1, 1])
#         )
        
#         # update samples
#         samples = samples.numpy()
#         samples[:, model.left.tag] = tf.reshape(xc_left_resamp, (-1,))
#         samples[:, model.right.tag] = tf.reshape(xc_right_resamp, (-1,))
#         samples = tf.convert_to_tensor(samples)
        
        
#         # draw samples from proposal
#         xt = model.proposal.sample(samples.shape, samples)[:, model.tag]
#         samples = samples.numpy()
#         samples[:, model.tag] = xt
#         samples = tf.convert_to_tensor(samples)
#         print(model.tag, samples[:5])
# #         xt_new = tf.concat([samples, xt], axis=1)
#         log_w = model.log_gamma(samples)
#         log_w -= model.left.log_gamma(samples)
#         log_w -= model.right.log_gamma(samples) 
#         log_w -= model.proposal.log_prob(xt, samples)
#         return samples, log_w



def dc_smc(samples, weights, model):
    
    if model.left is None:
        xt = model.proposal.sample((samples.shape[0], 1), None)
        # print(model.tag, xt.shape)
        # update samples
        samples = samples.numpy()
        samples[:, model.tag] = tf.reshape(xt, (-1,))
        samples = tf.convert_to_tensor(samples)
        # print(model.tag, samples.shape)
        # compute weight
        log_w = model.log_gamma(samples) - model.proposal.log_prob(tf.reshape(samples[:, model.tag], (-1, 1)), None)
        return samples, log_w
    else:
        xc_left, log_w_left = dc_smc(samples, weights, model.left)
        xc_left = tf.gather(samples, [model.left.tag], axis=1)
        # print(model.tag, xc_left.shape, log_w_left.shape)
        #! here's wrong! need to update samples here?
        xc_right, log_w_right = dc_smc(samples, weights, model.right)
        xc_right = tf.gather(samples, [model.right.tag], axis=1)
        
        # resample
        xc_left_resamp = resample(log_w_left, xc_left)
        xc_right_resamp = resample(log_w_right, xc_right)
        
        # update samples
        print(model.left.tag, xc_left_resamp)
        print(model.right.tag, xc_right_resamp)
        samples = samples.numpy()
        samples[:, model.left.tag] = tf.reshape(xc_left_resamp, (-1,))
        samples[:, model.right.tag] = tf.reshape(xc_right_resamp, (-1,))
        samples = tf.convert_to_tensor(samples)
        print(model.tag, model.left.tag, model.right.tag, samples)
        
        
        # draw samples from proposal
        xt = model.proposal.sample((samples.shape[0], 1), samples)
        samples = samples.numpy()
        samples[:, model.tag] = tf.reshape(xt, (-1,))
        samples = tf.convert_to_tensor(samples)

        # compute weights
        log_w = model.log_gamma(samples)
        log_w -= model.left.log_gamma(samples)
        log_w -= model.right.log_gamma(samples) 
        log_w -= model.proposal.log_prob(tf.reshape(xt, (-1, 1)), samples)
        return samples, log_w
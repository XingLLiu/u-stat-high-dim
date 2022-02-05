import tensorflow as tf
import tensorflow_probability as tfp

def pairwise_mahalanobis(inv_hessian, x):
    """Compute Mahalanobis distance
    inv_hessian: dim x dim
    x: dim
    """
    x_h = tf.expand_dims(x, axis=0) # 1 x dim
    res = x_h @ inv_hessian @ tf.transpose(x_h) # 1,
    return res[0, 0]

def merge_modes(inv_hessians: tf.Tensor, end_pts: tf.Tensor, threshold: float, log_prob: callable):
    """Merge modes according to pairwise mahalanobis distance
    end_pts: m x dim
    inv_hessians: m x dim x dim
    """
    M = end_pts.shape[0]
    mode_list = [end_pts[0, :]]
    inv_hessians_list = [inv_hessians[0, :, :]]
    
    for i in range(1, M):
        maha_dist_list = []
        end_pt_i = end_pts[i, :]
        inv_hess_i = inv_hessians[i]
        
        # compute pairwise Mahalanobis dist with the existing modes
        for j in range(len(mode_list)):
            inv_hess = 0.5 * (inv_hessians_list[j] + inv_hess_i)
            diff = mode_list[j] - end_pt_i
            maha_dist = pairwise_mahalanobis(inv_hess, diff)
            maha_dist_list.append(maha_dist)

        # find the mode with the closest distance
        argmin_i = tf.math.argmin(maha_dist_list).numpy()
        min_maha_dist = maha_dist_list[argmin_i]
        
        if min_maha_dist < threshold:
            # classify into closest mode
            closest_mode = mode_list[argmin_i]

            if log_prob(closest_mode) < log_prob(end_pt_i):
                # if current pt is better than local mode, swap
                mode_list[argmin_i] = end_pt_i
                inv_hessians_list[argmin_i] = inv_hess_i
                
        else:
            # store current pt as a new mode
            mode_list.append(end_pt_i)
            inv_hessians_list.append(inv_hess_i)
    
    return mode_list, inv_hessians_list

def run_bfgs(start_pts: tf.Tensor, log_prob_fn: callable, **kwargs):
    """Run BFGS algorithm
    start_pts: M x dim
    """
    # define objective
    def nll_and_grad(x):
        return tfp.math.value_and_gradient(
            lambda x: -log_prob_fn(x), # minus as we want to minimise
            x)

    optim_results = tfp.optimizer.bfgs_minimize(nll_and_grad, initial_position=start_pts, **kwargs)

    if_converged = tf.experimental.numpy.all(optim_results.converged).numpy() # should return true if all converged

    if not if_converged:
        nstart_pts = start_pts.shape[0]
        not_conv = nstart_pts - tf.reduce_sum(tf.cast(optim_results.converged, dtype=tf.int32))
        print(Warning(f"{not_conv} of {nstart_pts} BFGS optim chains did not converge"))

    return optim_results

def find_modes(start_pts, log_prob_fn, threshold, **kwargs):
    """Run run_bfgs and merge_modes"""
    # run BFGS to find modes
    bfgs = run_bfgs(start_pts, log_prob_fn)
    end_pts = bfgs.position
    inverse_hessian_estimate = bfgs.inverse_hessian_estimate

    # merge modes
    mode_list, inv_hess_list = merge_modes(inverse_hessian_estimate, end_pts, threshold, log_prob_fn)

    return mode_list, inv_hess_list

def pairwise_directions(modes):
    """Compute v_{ij} = \mu_i - \mu_j for all 1 \leq i < j \leq len(modes). 
    Order does not matter for symmetric kernels
    modes: list of mode vectors. Must have length >= 2
    """
    n = len(modes)
    dir_list = []
    for i in range(n-1):
        for j in range(i+1, n):
            dir = modes[i] - modes[j]
            dir_list.append(dir)
    
    return dir_list
from operator import index
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
import argparse
import pickle

from src.ksd.ksd import KSD
from src.ksd.kernel import RBF, IMQ
from src.ksd.bootstrap import Bootstrap
import src.ksd.models as models
from src.ksd.langevin import RandomWalkMH
from src.ksd.find_modes import find_modes, pairwise_directions

def log_trans_q(x, log_q_fn, log_p_fn, std, dir_vec, jitter=1e-12):
    """Density of q after 1 step of Markov transition"""
    term1, term2 = 0., 0.
    p_x = tf.exp(log_p_fn(x)) + jitter # n
    q_x = tf.exp(log_q_fn(x)) # n

    for xp in [x + std*dir_vec, x - std*dir_vec]:
        p_xp = tf.exp(log_p_fn(xp)) + jitter
        q_xp = tf.exp(log_q_fn(xp))

        term1 += q_xp * tf.math.minimum(1, p_x / p_xp)
        term2 += q_x * (1 - tf.math.minimum(1, p_xp / p_x))
    
    den = tf.math.log(term1 + term2) # n
    return den

def score_den(x, log_trans_q_fn):
    """Compute gradient of log_trans_q_fn at x"""
    with tf.GradientTape() as g:
        g.watch(x)
        log_prob = log_trans_q_fn(x=x) # n

    score = g.gradient(log_prob, x) # n x dim
    assert score.shape == x.shape

    return score

def run_bootstrap_experiment(nrep, target, proposal_on, log_prob_fn, proposal_off, kernel, alpha, num_boot, T, std_ls, **kwargs):
    """compute KSD and repeat for nrep times"""
    ksd = KSD(target=target, kernel=kernel)
    
    n = kwargs["n"]
    ntrain = int(n*0.5)
    dim = proposal_off.event_shape[0]

    ksd_df = pd.DataFrame(columns=["n", "best_std", "p_value", "seed", "type"])
    iterator = trange(nrep)
    bootstrap = Bootstrap(ksd, n-ntrain)
    multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x ntest

    bootstrap_nopert = Bootstrap(ksd, n)
    multinom_samples_nopert = bootstrap_nopert.multinom.sample((nrep, num_boot)) # nrep x num_boot x n

    names = ["off-target", "target", "off-target no pert", "target no pert"]
    proposals = [proposal_off, proposal_on, proposal_off, proposal_on]

    # # generate points for finding modes #TODO
    # unif_dist = tfp.distributions.Uniform(low=-tf.ones((dim,)), high=tf.ones((dim,)))
    # start_pts_all = 20. * unif_dist.sample((nrep, kwargs["nstart_pts"])) # change range

    iterator.set_description(f"Running with sample size {n}")
    for seed in iterator:
        # # get initial points for finding modes #TODO
        # start_pts = start_pts_all[seed, :, :]
        # # merge modes
        # mode_list, _ = find_modes(start_pts, log_prob_fn, **kwargs)
        # # find between-modes dir
        # if len(mode_list) == 1:
        #     dir_vec_list = [mode_list[0]]
        # else:
        #     dir_vec_list = pairwise_directions(mode_list)

        for name, proposal in zip(names, proposals):
            sample_init = proposal.sample(n)

            if "no pert" not in name:
                sample_init_train, sample_init_test = sample_init[:ntrain, ], sample_init[ntrain:, ]
                
                #TODO start optim from samples
                start_pts = sample_init_train
                # merge modes
                mode_list, hess_list = find_modes(start_pts, log_prob_fn, **kwargs)
                
                # find between-modes dir
                if len(mode_list) == 1:
                    dir_vec_list, ind_pair_list = [mode_list[0]], [(0, 0)]
                else:
                    dir_vec_list, ind_pair_list = pairwise_directions(mode_list, return_index=True)

                # find v_{i^*, j^*}, \sigma^*_{i^*, j^*}
                best_ksd = 0.
                best_dir_vec = dir_vec_list[0]
                for j, dir_vec in enumerate(dir_vec_list):
                    # loop through directional vecs

                    # find estimated hessians
                    ind1, ind2 = ind_pair_list[j]
                    mode1, mode2 = mode_list[ind1], mode_list[ind2]
                    # hess1, hess2 = hess_list[ind1], hess_list[ind2]
                    hess1, hess2 = tf.eye(dim), tf.eye(dim) #! delete
                    hess1_sqrt = tf.linalg.sqrtm(hess1)
                    hess2_sqrt = tf.linalg.sqrtm(hess2)
                    hess1_inv_sqrt = tf.linalg.inv(hess1_sqrt)
                    hess2_inv_sqrt = tf.linalg.inv(hess2_sqrt)
                    hess_dict = {"mode1": mode1, "mode2": mode2, "hess1_sqrt": hess1_sqrt, "hess2_sqrt": hess2_sqrt,
                        "hess1_inv_sqrt": hess1_inv_sqrt, "hess2_inv_sqrt": hess2_inv_sqrt}

                    for i, std in enumerate(std_ls):
                        # loop through jump scales
                        iterator.set_description(f"Jump scale [{i+1} / {len(std_ls)}] of dir vector [{j+1} / {len(dir_vec_list)}]")

                        # run dynamic for T steps
                        mh = RandomWalkMH(log_prob=log_prob_fn)
                        # mh.run(steps=T, std=std, x_init=sample_init_train, dir_vec=dir_vec)
                        mh.run(steps=T, std=std, x_init=sample_init_train, **hess_dict)

                        # compute ksd
                        x_t = mh.x[-1, :, :].numpy()
                        # ksd_val = ksd(x_t, tf.identity(x_t)).numpy()
                        _, ksd_val = ksd.h1_var(X=x_t, Y=tf.identity(x_t), return_scaled_ksd=True)
                        ksd_val = ksd_val.numpy()
                            
                        # update if ksd is larger
                        if ksd_val > best_ksd:
                            best_std = std
                            best_dir_vec = dir_vec
                            best_ksd = ksd_val
                            best_hess_dict = hess_dict

                # run dynamic for T steps with test data and optimal params
                mh = RandomWalkMH(log_prob=log_prob_fn)
                # mh.run(steps=T, x_init=sample_init_test, std=best_std, dir_vec=best_dir_vec)
                mh.run(steps=T, x_init=sample_init_test, std=best_std, **best_hess_dict)

                # get perturbed samples
                x_t = mh.x[-1, :, :].numpy()

                # get multinomial sample
                multinom_one_sample = multinom_samples[seed, :] # nrep x num_boost x ntest
            
            else:
                x_t = sample_init
                multinom_one_sample = multinom_samples_nopert[seed, :] # nrep x num_boost x n

            # compute p-value
            _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=x_t, multinom_samples=multinom_one_sample)
            ksd_df.loc[len(ksd_df)] = [n, best_std, p_val, seed, name]

    return ksd_df, best_dir_vec, best_hess_dict


num_boot = 800 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
mode_threshold = 1. # threshold for merging modes
sigma_list = np.linspace(0.5, 1.5, 21).tolist() # std for discrete jump proposal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="", help="path to pre-saved results")
    parser.add_argument("--model", type=str, default="bimodal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--n", type=int, default=500, help="sample size")
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--nrep", type=int, default=50)
    parser.add_argument("--ratio_t", type=float, default=0.5, help="max num of steps")
    parser.add_argument("--ratio_s", type=float, default=1.)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--nmodes", type=int, default=10)
    parser.add_argument("--nbanana", type=int, default=2)
    args = parser.parse_args()
    model = args.model
    seed = args.seed
    T = args.T
    n = args.n
    dim = args.dim
    nstart_pts = 20 * dim # num of starting points for finding modes
    nrep = args.nrep
    ratio_target = args.ratio_t
    ratio_sample = args.ratio_s
    k = args.k
    delta_list = [4.0] if args.model != "bimodal" else [1.0, 2.0, 4.0, 6.0]

    tf.random.set_seed(seed)

    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 15))
    subfigs = fig.subfigures(1, len(delta_list))

    for ind, delta in tqdm(enumerate(delta_list)):
        print(f"Running with delta = {delta}")
        test_imq_df = None
        
        # set model
        if model == "bimodal":
            model_name = f"{model}_steps{T}_ratio{ratio_target}_{ratio_sample}_k{k}_seed{seed}"
            create_target_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, return_logprob=True, ratio=ratio_target)
            create_sample_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, return_logprob=True, ratio=ratio_sample)

        elif model == "t-banana":
            nmodes = args.nmodes
            nbanana = args.nbanana
            model_name = f"{model}_steps{T}_nmodes{nmodes}_diag_seed{seed}"
            ratio_target = [1/nmodes] * nmodes
            random_weights = tfp.distributions.Uniform(low=0., high=1.).sample(nmodes)
            # ratio_sample = random_weights / tf.reduce_sum(random_weights)
            ratio_sample = [0.3, 0.7] #TODO
            loc = tfp.distributions.Uniform(low=-tf.ones((dim,))*10, high=tf.ones((dim,))*20).sample(nmodes) # uniform in [-20, 20]^d

            loc = tf.constant([[4.1144876, 6.538103, 4.1443825, 4.311162, -9.23945],
                [-4.1144876, -6., 0.08298874, -1.2335253, 7.6566525],
                [-4.977256, 16.026318, 8.160973, 12.019077, -6.351266],
                [ 5.583577, 10.599161, 13.44928, 1.0351582, 3.1638136 ]])
            # loc = tf.constant([[4.1144876, 6.538103, 4.1443825, 4.311162, -9.23945],
            #     [4.1144876, -6., 0.08298874, -1.2335253, 7.6566525],
            #     [-4.977256, 16.026318, 8.160973, 12.019077, -6.351266],
            #     [ 5.583577, 10.599161, 13.44928, 1.0351582, 3.1638136 ]])
            # loc = tf.constant([[40.1144876, 6.538103, 4.1443825, 4.311162, -9.23945],
            #     [-40.1144876, 6., 0.08298874, -1.2335253, 7.6566525],
            #     [-4.977256, 16.026318, 8.160973, 12.019077, -6.351266],
            #     [ 5.583577, 10.599161, 13.44928, 1.0351582, 3.1638136 ]])
            loc = loc[:nmodes, :]

            b = 0.003 # 0.03
            create_target_model = models.create_mixture_t_banana(dim=dim, ratio=ratio_target, loc=loc, b=b,
                nbanana=nbanana, return_logprob=True)
            create_sample_model = models.create_mixture_t_banana(dim=dim, ratio=ratio_sample, loc=loc, b=b,
                nbanana=nbanana, return_logprob=True)

        elif model == "gaussianmix":
            nmodes = args.nmodes
            model_name = f"{model}{nmodes}_steps{T}_seed{seed}_delta2.0_newproposal_flipped"
            means = tfp.distributions.Uniform(low=-tf.ones((dim,))*20, high=tf.ones((dim,))*20).sample(nmodes) # uniform in [-5, 5]^d
            # means = tf.constant([[4.1144876, 6.538103, 4.1443825, 4.311162, -9.23945],
            #     [-4.1144876, -6., 0.08298874, -1.2335253, 7.6566525],
            #     [-4.977256, 16.026318, 8.160973, 12.019077, -6.351266],
            #     [ 5.583577, 10.599161, 13.44928, 1.0351582, 3.1638136 ]])
            means = tf.constant([[-15.692321, -1.1227798, -15.462275, -7.772279, 0.45660973],
                [ 13.653507, 12.54893875, -16.00213, -17.85245, -19.995266],
                [  10.38526535, -5.5862007, 2.0345726, -12.112765, -19.61071],
                [  25.736, 25.068437, -16.550621, -9.046288, -18.059994]])
            means = means[:nmodes, :]

            indicator = tf.cast(tfp.distributions.Bernoulli(probs=0.5).sample(nmodes-1), dtype=bool)
            indicator = tf.concat([tf.constant([True]), indicator], axis=0)
            random_weights = tfp.distributions.Uniform(low=0., high=1.).sample(nmodes)
            random_weights = tf.where(indicator, random_weights, 0.)
            # ratio_sample = random_weights / tf.reduce_sum(random_weights)
            ratio_sample = [0.19549179, 0.1501031, 0.50429535, 0.15010974] #! delete

            ratio_target = 0.5
            delta = 2.

            create_target_model = models.create_mixture_20_gaussian(means, ratio=ratio_target, scale=delta, return_logprob=True)
            create_sample_model = models.create_mixture_20_gaussian(means, ratio=ratio_sample, scale=delta, return_logprob=True)

        elif model == "gauss-scaled":
            model_name = f"{model}_1d_steps{T}_with_hessian_seed{seed}"
            create_target_model = models.create_mixture_gaussian_scaled(ratio=ratio_target, return_logprob=True)
            create_sample_model = models.create_mixture_gaussian_scaled(ratio=ratio_sample, return_logprob=True)

        # target distribution
        target, log_prob_fn = create_target_model

        # on-target proposal distribution
        proposal_on, log_prob_on_fn = create_target_model
        
        # off-target proposal distribution
        proposal_off, log_prob_off_fn = create_sample_model
        
        # check if log_prob is correct
        models.check_log_prob(target, log_prob_fn)
        models.check_log_prob(proposal_off, log_prob_off_fn)

        if len(args.load) > 0 :
            try:
                test_imq_df = pd.read_csv(args.load + f"/{model_name}_delta{delta}.csv")
                print(f"Loaded pre-saved data for steps = {T}")
            except:
                print(f"Pre-saved data for steps = {T}, delta = {delta} not found. Running from scratch now.")

        if test_imq_df is None:
            # with IMQ
            imq = IMQ(med_heuristic=True)
            test_imq_df, best_dir_vec, best_hess_dict = run_bootstrap_experiment(
                nrep, target, proposal_on, log_prob_fn, proposal_off, imq, 
                alpha, num_boot, T, sigma_list, threshold=mode_threshold, nstart_pts=nstart_pts, n=n)

            # save res
            test_imq_df.to_csv(f"res/bootstrap/{model_name}_delta{delta}.csv", index=False)
            np.save(f"res/bootstrap/{model_name}_delta{delta}.npy", best_dir_vec.numpy())
            pickle.dump(best_hess_dict, open(f"res/bootstrap/{model_name}_delta{delta}.pkl", "wb"))


        # between-modes vector (only for plotting)
        dir_vec = tf.constant(np.load(f"res/bootstrap/{model_name}_delta{delta}.npy"))

        # for plotting dist of mh-perturbed samples`
        off_sample = proposal_off.sample(1000)
        best_std_off = test_imq_df.loc[(test_imq_df.type == "off-target"), "best_std"].median()
        mh_off = RandomWalkMH(log_prob=log_prob_fn)
        mh_off.run(steps=T, std=best_std_off, x_init=off_sample, dir_vec=dir_vec)

        on_sample = proposal_on.sample(1000)
        best_std_on = test_imq_df.loc[(test_imq_df.type == "target"), "best_std"].median()
        mh_on = RandomWalkMH(log_prob=log_prob_fn)
        mh_on.run(steps=T, std=best_std_on, x_init=on_sample, dir_vec=dir_vec)

        # plot
        subfig = subfigs.flat[ind] if len(delta_list) > 1 else subfigs
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(5, 1) # (8, 1)
        axs = axs.flat
        xlim, ylim = 30, 30
        samples_df_off_target = pd.DataFrame({
            "x1": off_sample.numpy()[:, 0], 
            "x2": off_sample.numpy()[:, 1],
            "type": "off-target"})
        samples_df_off_perturbed = pd.DataFrame({
            "x1": mh_off.x[-1, :, :].numpy()[:, 0], 
            "x2": mh_off.x[-1, :, :].numpy()[:, 1],
            "type": "perturbed off-target"})
        samples_df_off = pd.concat([samples_df_off_target, samples_df_off_perturbed], ignore_index=True)

        # sns.ecdfplot(ax=axs[0], data=samples_df_off, x="x1", hue="type")
        # sns.kdeplot(ax=axs[0], data=samples_df_off, x="x1", y="x2", hue="type", alpha=0.8)
        sns.scatterplot(ax=axs[0], data=samples_df_off, x="x1", y="x2", hue="type", alpha=0.8)
        axs[0].axis(xmin=-xlim, xmax=xlim, ymin=-ylim, ymax=ylim)
        axs[0].set_ylabel("CDF")
        if ind != len(delta_list) - 1: axs[0].legend([],[], frameon=False)

        samples_df_target = pd.DataFrame({
            "x1": on_sample.numpy()[:, 0],
            "x2": on_sample.numpy()[:, 1],
            "type": "target"})
        samples_df_perturbed = pd.DataFrame({
            "x1": mh_on.x[-1, :, :].numpy()[:, 0],
            "x2": mh_on.x[-1, :, :].numpy()[:, 1],
            "type": "perturbed target"})
        samples_df = pd.concat([samples_df_target, samples_df_perturbed], ignore_index=True)
        
        # sns.ecdfplot(ax=axs[1], data=samples_df, x="x1", hue="type")
        # sns.kdeplot(ax=axs[1], data=samples_df, x="x1", y="x2", hue="type", alpha=0.8)
        sns.scatterplot(ax=axs[1], data=samples_df, x="x1", y="x2", hue="type", alpha=0.8)
        axs[1].axis(xmin=-xlim, xmax=xlim, ymin=-ylim, ymax=ylim)
        axs[1].set_ylabel("CDF")
        if ind != len(delta_list) - 1: axs[1].legend([],[], frameon=False)

        err2 = (test_imq_df.loc[(test_imq_df.type == "off-target"), "p_value"] > alpha).mean()
        err2_nopert = (test_imq_df.loc[(test_imq_df.type == "off-target no pert"), "p_value"] > alpha).mean()
        sns.ecdfplot(ax=axs[2], data=test_imq_df.loc[test_imq_df.type.isin(["off-target", "off-target no pert"])], x="p_value", hue="type")
        axs[2].plot([0, 1], [0, 1], transform=axs[2].transAxes, color="grey", linestyle="dashed")
        axs[2].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        axs[2].set_title(f"type II error = {err2}, no pert. = {err2_nopert}")
        axs[2].set_xlabel("p-value")
        if ind != len(delta_list) - 1: axs[2].legend([],[], frameon=False)

        err1 = (test_imq_df.loc[(test_imq_df.type == "target"), "p_value"] <= alpha).mean()
        err1_nopert = (test_imq_df.loc[(test_imq_df.type == "target no pert"), "p_value"] <= alpha).mean()
        sns.ecdfplot(ax=axs[3], data=test_imq_df.loc[test_imq_df.type.isin(["target", "target no pert"])], x="p_value", hue="type")
        axs[3].plot([0, 1], [0, 1], transform=axs[3].transAxes, color="grey", linestyle="dashed")
        axs[3].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        axs[3].set_title(f"type I error = {err1}, no pert. = {err1_nopert}")
        axs[3].set_xlabel("p-value")
        if ind != len(delta_list) - 1: axs[3].legend([],[], frameon=False)
        
        sns.ecdfplot(ax=axs[4], data=test_imq_df, x="best_std", hue="type")
        axs[4].axis(xmin=0.4, xmax=1.6, ymin=0, ymax=1.01)
        axs[4].set_title("median of estimated best std = {:.2f} (off), {:.2f} (on)".format(best_std_off, best_std_on))
        axs[4].set_xlabel("p-value")
        if ind != len(delta_list) - 1: axs[4].legend([],[], frameon=False)

    fig.savefig(f"figs/bootstrap/bootstrap_{model_name}.png")

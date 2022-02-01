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

from src.ksd.ksd import KSD
from src.ksd.kernel import RBF, IMQ
from src.ksd.bootstrap import Bootstrap
from src.ksd.models import create_mixture_gaussian
from src.ksd.langevin import RandomWalkMH
from src.ksd.find_modes import find_modes

tf.random.set_seed(0)

def log_trans_q(x, log_q_fn, log_p_fn, std, dir_vec):
    """Density of q after 1 step of Markov transition"""
    term1, term2 = 0., 0.
    p_x = tf.exp(log_p_fn(x)) # n
    q_x = tf.exp(log_q_fn(x)) # n

    for xp in [x + std*dir_vec, x - std*dir_vec]:
        p_xp = tf.exp(log_p_fn(xp))
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
    
    n = 500
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

    iterator.set_description(f"Running with sample size {n}")
    for seed in iterator:
        # generate points for finding modes
        unif_dist = tfp.distributions.Uniform(low=-tf.ones((dim,)), high=tf.ones((dim,)))
        start_pts = 5. * unif_dist.sample(kwargs["nstart_pts"])

        # merge modes
        mode_list, _ = find_modes(start_pts, log_prob_fn, **kwargs)

        # find between-modes dir #TODO this assumes two modes are found
        if len(mode_list) == 1:
            dir_vec = mode_list[0]
        else:
            dir_vec = mode_list[0] - mode_list[1] # order does not matter as the proposal is symmetric

        for name, proposal in zip(names, proposals):
            iterator.set_description(f"Repetition: {seed+1} of {nrep}")
            sample_init = proposal.sample(n)

            if "no pert" not in name:
                sample_init_train, sample_init_test = sample_init[:ntrain, ], sample_init[ntrain:, ]

                best_ksd = 0.
                for i, std in enumerate(std_ls):
                    iterator.set_description(f"Jump dist after BFGS {i+1} of {len(std_ls)}")

                    # run dynamic for T steps
                    mh = RandomWalkMH(log_prob=log_prob_fn)
                    mh.run(steps=T, std=std, x_init=sample_init_train, dir_vec=dir_vec)
                    
                    # compute ksd
                    x_t = mh.x[-1, :, :].numpy()
                    ksd_val = ksd(x_t, tf.identity(x_t)).numpy()

                    if ksd_val > best_ksd:
                        best_std = std
                        best_ksd = ksd_val

                # run dynamic for T steps with test data
                mh = RandomWalkMH(log_prob=log_prob_fn)
                mh.run(steps=T, std=best_std, x_init=sample_init_test, dir_vec=dir_vec)

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

    return ksd_df


dim = 5
nrep = 500
num_boot = 1000 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
delta_list = [1.0, 2.0, 4.0, 6.0]
T = 2 # max num of steps
mode_threshold = 1. # threshold for merging modes
nstart_pts = 20 * dim # num of starting points for finding modes
sigma_list = np.linspace(0.5, 1.5, 26).tolist() # std for discrete jump proposal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="", help="path to pre-saved results")
    parser.add_argument("--ratio_t", type=float, default=0.5)
    parser.add_argument("--ratio_s", type=float, default=1.)
    args = parser.parse_args()
    ratio_target = args.ratio_t
    ratio_sample = args.ratio_s

    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 15))
    subfigs = fig.subfigures(1, len(delta_list))
    

    for ind, delta in tqdm(enumerate(delta_list)):
        print(f"Running with delta = {delta}")
        test_imq_df = None

        # target distribution
        target, log_prob_fn = create_mixture_gaussian(dim=dim, delta=delta, return_logprob=True, ratio=ratio_target)

        # off-target proposal distribution
        proposal_on, log_prob_on_fn = create_mixture_gaussian(dim=dim, delta=delta, return_logprob=True, ratio=ratio_target)
        
        # off-target proposal distribution
        proposal_off, log_prob_off_fn = create_mixture_gaussian(dim=dim, delta=delta, return_logprob=True, ratio=ratio_sample)
        
        if len(args.load) > 0 :
            try:
                test_imq_df = pd.read_csv(args.load + f"/mh_discrete_optim_steps{T}_delta{delta}_ratio{ratio_target}_{ratio_sample}.csv")
                print(f"Loaded pre-saved data for steps = {T}")
            except:
                print(f"Pre-saved data for steps = {T}, delta = {delta} not found. Running from scratch now.")

        if test_imq_df is None:
            # with IMQ
            imq = IMQ(med_heuristic=True)
            test_imq_df = run_bootstrap_experiment(nrep, target, proposal_on, log_prob_fn, proposal_off, imq, alpha, num_boot, T, sigma_list,
                threshold=mode_threshold, nstart_pts=nstart_pts)

            # save res
            test_imq_df.to_csv(f"res/bootstrap/mh_discrete_optim_steps{T}_delta{delta}_ratio{ratio_target}_{ratio_sample}.csv", index=False)


        # between-modes vector (only for plotting)
        dir_vec = tf.eye(dim)[:, 0] * delta * 2 

        # for plotting dist of mh-perturbed samples`
        off_sample = proposal_off.sample(1000)
        best_std_off = test_imq_df.loc[(test_imq_df.type == "off-target"), "best_std"].median()
        mh_off = RandomWalkMH(log_prob=log_prob_fn)
        mh_off.run(steps=T, std=best_std_off, x_init=off_sample, dir_vec=dir_vec)

        on_sample = proposal_on.sample(1000)
        best_std_on = test_imq_df.loc[(test_imq_df.type == "target"), "best_std"].median()
        mh_on = RandomWalkMH(log_prob=log_prob_fn)
        mh_on.run(steps=T, std=best_std_on, x_init=on_sample, dir_vec=dir_vec)

        # log prob of 1-step mh-perturbed densities        
        xx = tf.concat(
            [tf.reshape(tf.linspace(-2*delta, 2*delta, 1000), (-1, 1)), tf.zeros((1000, dim-1))],
            # [tf.reshape(tf.linspace(-1.4, -1.2, 1000), (-1, 1)), tf.zeros((1000, dim-1))],
            axis=1) # 200 x dim
        
        log_trans_q_off = lambda x: log_trans_q(x, log_prob_off_fn, log_prob_fn, best_std_off, dir_vec)
        log_prob_trans_off = log_trans_q_off(xx).numpy()
        log_prob_trans_off_df = pd.DataFrame({"y1": log_prob_trans_off, "x1": xx[:, 0].numpy(), "type": "pert. off-target"})

        log_prob_off = log_prob_off_fn(xx).numpy()
        log_prob_off_df = pd.DataFrame({"y1": log_prob_off, "x1": xx[:, 0].numpy(), "type": "off-target"})

        # log_trans_q_on = lambda x: log_trans_q(x, log_prob_on_fn, log_prob_fn, best_std_on, dir_vec)
        log_trans_q_on = log_prob_on_fn
        log_prob_trans_on = log_trans_q_on(xx).numpy()
        log_prob_trans_on_df = pd.DataFrame({"y1": log_prob_trans_on, "x1": xx[:, 0].numpy(), "type": "target"})
        
        log_prob_trans_diff_df = pd.DataFrame({"y1": np.exp(log_prob_trans_off) * np.abs(log_prob_trans_off - log_prob_trans_on), "x1": xx[:, 0].numpy(), "type": "abs diff"})
        log_prob_trans_df = pd.concat([log_prob_trans_off_df, log_prob_trans_on_df, log_prob_trans_diff_df, log_prob_off_df], ignore_index=True)

        # score functions of 1-step mh-perturbed densities
        score_trans_off = score_den(xx, log_trans_q_off).numpy()[:, 0] # n x dim
        score_trans_off_df = pd.DataFrame({"y1": score_trans_off, "x1": xx[:, 0].numpy(), "type": "pert. off-target"})

        score_off = score_den(xx, log_prob_off_fn).numpy()[:, 0] # n x dim
        score_off_df = pd.DataFrame({"y1": score_off, "x1": xx[:, 0].numpy(), "type": "off-target"})

        score_trans_on = score_den(xx, log_trans_q_on).numpy()[:, 0] # n x dim
        score_trans_on_df = pd.DataFrame({"y1": score_trans_on, "x1": xx[:, 0].numpy(), "type": "target"})

        score_trans_df = pd.concat([score_trans_off_df, score_trans_on_df, score_off_df], ignore_index=True)

        score_trans_diff_df = pd.DataFrame({"y1": np.exp(log_prob_trans_off) * np.abs(score_trans_off - score_trans_on), "x1": xx[:, 0].numpy(), "type": "abs diff pert."})
        score_diff_df = pd.DataFrame({"y1": np.exp(log_prob_off) * np.abs(score_off - score_trans_on), "x1": xx[:, 0].numpy(), "type": "abs diff"})
        # score_trans_diff_df = pd.DataFrame({"y1": np.abs(score_trans_off - score_trans_on), "x1": xx[:, 0].numpy(), "type": "abs diff pert."})
        # score_diff_df = pd.DataFrame({"y1": np.abs(score_off - score_trans_on), "x1": xx[:, 0].numpy(), "type": "abs diff"})
        score_diff_df = pd.concat([score_trans_diff_df, score_diff_df], ignore_index=True)


        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(8, 1)
        axs = axs.flat
        samples_df_off_target = pd.DataFrame({"x1": off_sample.numpy()[:, 0], "type": "off-target"})
        samples_df_off_perturbed = pd.DataFrame({"x1": mh_off.x[-1, :, :].numpy()[:, 0], "type": "perturbed off-target"})
        samples_df_off = pd.concat([samples_df_off_target, samples_df_off_perturbed], ignore_index=True)

        sns.ecdfplot(ax=axs[0], data=samples_df_off, x="x1", hue="type")
        # sns.histplot(ax=axs[0], data=samples_df_off, x="x1", hue="type", alpha=0.3)
        axs[0].axis(xmin=-3*delta, xmax=3*delta)
        axs[0].set_ylabel("CDF")
        if ind != len(delta_list) - 1: axs[0].legend([],[], frameon=False)

        samples_df_target = pd.DataFrame({"x1": on_sample.numpy()[:, 0], "type": "target"})
        samples_df_perturbed = pd.DataFrame({"x1": mh_on.x[-1, :, :].numpy()[:, 0], "type": "perturbed target"})
        samples_df = pd.concat([samples_df_target, samples_df_perturbed], ignore_index=True)
        sns.ecdfplot(ax=axs[1], data=samples_df, x="x1", hue="type")
        # sns.histplot(ax=axs[1], data=samples_df_off, x="x1", hue="type", alpha=0.3)
        axs[1].axis(xmin=-3*delta, xmax=3*delta)
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
        axs[3].plot([0, 1], [0, 1], transform=axs[2].transAxes, color="grey", linestyle="dashed")
        axs[3].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        axs[3].set_title(f"type I error = {err1}, no pert. = {err1_nopert}")
        axs[3].set_xlabel("p-value")
        if ind != len(delta_list) - 1: axs[3].legend([],[], frameon=False)
        
        sns.ecdfplot(ax=axs[4], data=test_imq_df, x="best_std", hue="type")
        axs[4].axis(xmin=0.4, xmax=1.6, ymin=0, ymax=1.01)
        axs[4].set_title("median of estimated best std = {:.2f} (off), {:.2f} (on)".format(best_std_off, best_std_on))
        axs[4].set_xlabel("p-value")
        if ind != len(delta_list) - 1: axs[4].legend([],[], frameon=False)

        sns.lineplot(ax=axs[5], data=log_prob_trans_df, x="x1", y="y1", hue="type", style="type")
        axs[5].axis(xmin=-3*delta, xmax=3*delta, ymin=-6, ymax=2)
        # axs[5].axis(xmin=np.min(xx[:, 0]), xmax=np.max(xx[:, 0]), ymin=-1, ymax=1)
        # axs[5].axis(xmin=-10., xmax=10., ymin=-6, ymax=2)
        axs[5].set_ylabel("log prob")
        if ind != len(delta_list) - 1: axs[5].legend([],[], frameon=False)

        sns.lineplot(ax=axs[6], data=score_trans_df, x="x1", y="y1", hue="type", style="type")
        axs[6].axis(xmin=-3*delta, xmax=3*delta)
        # axs[6].axis(xmin=np.min(xx[:, 0]), xmax=np.max(xx[:, 0]), ymin=-6, ymax=2)
        # axs[6].axis(xmin=-10., xmax=10., ymin=-6, ymax=2)
        axs[6].set_ylabel("score")
        if ind != len(delta_list) - 1: axs[6].legend([],[], frameon=False)

        sns.lineplot(ax=axs[7], data=score_diff_df, x="x1", y="y1", hue="type", style="type")
        axs[7].axis(xmin=-3*delta, xmax=3*delta)
        # axs[7].axis(xmin=-10., xmax=10.)
        axs[7].set_ylabel("score differences")
        if ind != len(delta_list) - 1: axs[7].legend([],[], frameon=False)


    fig.savefig(f"figs/bootstrap/bootstrap_mh_discrete_optim_steps{T}_ratio{ratio_target}_{ratio_sample}.png")

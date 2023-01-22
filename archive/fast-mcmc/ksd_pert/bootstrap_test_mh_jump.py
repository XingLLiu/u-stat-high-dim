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

tf.random.set_seed(0)

def run_bootstrap_experiment(nrep, target, proposal_on, log_prob_fn, proposal_off, kernel, alpha, num_boot, T, std_ls, dir_vec):
    """compute KSD and repeat for nrep times"""
    ksd = KSD(target=target, kernel=kernel)
    
    n = 500
    ntrain = int(n*0.5)
    # T = int(max(t_list)) + 1 # maximal number of steps

    ksd_df = pd.DataFrame(columns=["n", "best_std", "p_value", "seed", "type"])
    iterator = trange(nrep)
    bootstrap = Bootstrap(ksd, n-ntrain)
    multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x n
    iterator.set_description(f"Running with sample size {n}")
    for seed in iterator:
        for name, proposal in zip(["off-target", "target"], [proposal_off, proposal_on]):
            iterator.set_description(f"Repetition: {seed+1} of {nrep}")
            sample_init = proposal.sample(n)
            sample_init_train, sample_init_test = sample_init[:ntrain, ], sample_init[ntrain:, ]

            best_ksd = 0.
            for i, std in enumerate(std_ls):
                iterator.set_description(f"Jump dist {i+1} of {len(std_ls)}")

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

            # compute p-value
            x_t = mh.x[-1, :, :].numpy()
            _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=x_t, multinom_samples=multinom_samples[seed, :])
            ksd_df.loc[len(ksd_df)] = [n, best_std, p_val, seed, name]

    return ksd_df


dim = 5
nrep = 500
num_boot = 1000 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
delta_list = [1.0, 2.0, 4.0, 6.0]
T = 10 # max num of steps
std_list = np.linspace(0.02, 1., 25).tolist() # std for discrete jump proposal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="", help="path to pre-saved results")
    parser.add_argument("--ratio_t", type=float, default=0.5)
    parser.add_argument("--ratio_s", type=float, default=1.)
    args = parser.parse_args()
    ratio_target = args.ratio_t
    ratio_sample = args.ratio_s

    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 9))
    subfigs = fig.subfigures(1, len(delta_list))
    

    for ind, delta in tqdm(enumerate(delta_list)):
        print(f"Running with delta = {delta}")
        test_imq_df = None

        # target distribution
        target, log_prob_fn = create_mixture_gaussian(dim=dim, delta=delta, return_logprob=True, ratio=ratio_target)

        # off-target proposal distribution
        proposal_on = create_mixture_gaussian(dim=dim, delta=delta, ratio=ratio_target)
        
        # off-target proposal distribution
        proposal_off = create_mixture_gaussian(dim=dim, delta=delta, ratio=ratio_sample)

        # between-modes vector
        dir_vec = tf.eye(dim)[:, 0] * delta * 2
        
        if len(args.load) > 0 :
            try:
                test_imq_df = pd.read_csv(args.load + f"/mh_discrete_steps{T}_delta{delta}.csv")
                print(f"Loaded pre-saved data for steps = {T}")
            except:
                print(f"Pre-saved data for steps = {T}, delta = {delta} not found. Running from scratch now.")

        if test_imq_df is None:
            # with IMQ
            imq = IMQ(med_heuristic=True)
            test_imq_df = run_bootstrap_experiment(nrep, target, proposal_on, log_prob_fn, proposal_off, imq, alpha, num_boot, T, std_list, dir_vec)

            # save res
            test_imq_df.to_csv(f"res/bootstrap/mh_discrete_steps{T}_delta{delta}.csv", index=False)


        # for plotting dist of mh-perturbed samples
        off_sample = proposal_off.sample(1000)
        best_std_off = test_imq_df.loc[(test_imq_df.type == "off-target"), "best_std"].median()
        mh_off = RandomWalkMH(log_prob=log_prob_fn)
        mh_off.run(steps=T+1, std=best_std_off, x_init=off_sample, dir_vec=dir_vec)

        on_sample = proposal_on.sample(1000)
        best_std_on = test_imq_df.loc[(test_imq_df.type == "target"), "best_std"].median()
        mh_on = RandomWalkMH(log_prob=log_prob_fn)
        mh_on.run(steps=T+1, std=best_std_on, x_init=on_sample, dir_vec=dir_vec)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(4, 1)
        axs = axs.flat
        samples_df_off_target = pd.DataFrame({"x1": off_sample.numpy()[:, 0], "type": "off-target"})
        samples_df_off_perturbed = pd.DataFrame({"x1": mh_off.x[-1, :, :].numpy()[:, 0], "type": "perturbed off-target"})
        samples_df_off = pd.concat([samples_df_off_target, samples_df_off_perturbed], ignore_index=True)

        sns.ecdfplot(ax=axs[0], data=samples_df_off, x="x1", hue="type")
        axs[0].set_ylabel("CDF")

        samples_df_target = pd.DataFrame({"x1": on_sample.numpy()[:, 0], "type": "target"})
        samples_df_perturbed = pd.DataFrame({"x1": mh_on.x[-1, :, :].numpy()[:, 0], "type": "perturbed target"})
        samples_df = pd.concat([samples_df_target, samples_df_perturbed], ignore_index=True)
        sns.ecdfplot(ax=axs[1], data=samples_df, x="x1", hue="type")
        axs[1].set_ylabel("CDF")

        err2 = (test_imq_df.loc[(test_imq_df.type == "off-target"), "p_value"] > alpha).mean()
        err1 = (test_imq_df.loc[(test_imq_df.type == "target"), "p_value"] <= alpha).mean()
        sns.ecdfplot(ax=axs[2], data=test_imq_df, x="p_value", hue="type")
        axs[2].plot([0, 1], [0, 1], transform=axs[2].transAxes, color="grey", linestyle="dashed")
        axs[2].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        axs[2].set_title(f"type II error = {err2}, type I error = {err1}")
        axs[2].set_xlabel("p-value")
        
        sns.ecdfplot(ax=axs[3], data=test_imq_df, x="best_std", hue="type")
        axs[3].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        axs[3].set_title("median of estimated best std = {:.2f} (off), {:.2f} (on)".format(best_std_off, best_std_on))
        axs[3].set_xlabel("p-value")


    # plt.tight_layout()
    fig.savefig(f"figs/bootstrap/bootstrap_mh_discrete_steps{T}.png")

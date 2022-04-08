from operator import index
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import argparse

from src.ksd.ksd import ConvolvedKSD
from src.ksd.kernel import RBF, IMQ
from src.ksd.bootstrap import Bootstrap
from src.ksd.models import create_mixture_gaussian

tf.random.set_seed(0)

def run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, convolution, kernel, alpha, num_boot, num_est, log_noise_std_list):
    """compute KSD and repeat for nrep times"""
    ksd = ConvolvedKSD(target=target, kernel=kernel, conv_kernel=None) #TODO conv_kernel is not used in class
    
    n = 500
    # num train samples for finding sigma
    ntrain = int(n * 0.5)

    ksd_df = []
    iterator = trange(nrep)

    bootstrap_train = Bootstrap(ksd, ntrain)
    multinom_samples_train = bootstrap_train.multinom.sample((nrep, num_boot)) # nrep x num_boot x n_train
    bootstrap = Bootstrap(ksd, n-ntrain)
    multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x (n_test)

    print(f"Running with total sample size {n}")
    for seed in iterator:
        iterator.set_description(f"Repetition: {seed+1} of {nrep}")
        # convolution sample
        conv_sample_full = convolution.sample(num_est) # for p

        conv_ind = tf.experimental.numpy.random.randint(low=0, high=num_est, size=n)
        conv_sample = tf.gather(conv_sample_full, conv_ind, axis=0) # for q

        for dist, dist_name in zip([proposal_off, proposal_off, proposal_on, proposal_on], ["off-target", "off-target noiseless", "target", "target noiseless"]):
            # draw samples
            proposal_sample = dist.sample(n)

            # estimate optimal variance
            sample_train, sample_test = proposal_sample[:ntrain, :], proposal_sample[ntrain:, :]
            conv_sample_train, conv_sample_test = conv_sample[:ntrain, :], conv_sample[ntrain:, :]
            if "noiseless" in dist_name:
                best_log_noise_std = tf.Variable(-1e18)
            else:
                best_log_noise_std = log_noise_std_list[0]
                smalles_p_val = 1.

                for log_noise_std in log_noise_std_list:
                    # compute p-value with train set
                    _, p_val = bootstrap.test_once(
                        alpha=alpha, 
                        num_boot=num_boot, 
                        X=sample_train, 
                        multinom_samples=multinom_samples_train[seed, :], 
                        conv_samples_full=conv_sample_full,
                        conv_samples=conv_sample_train,
                        log_noise_std=log_noise_std
                    )
                    # find the var that leads to the smallest p-value
                    if p_val < smalles_p_val:
                        best_log_noise_std = log_noise_std
                        smalles_p_val = p_val

            # initialize test
            ksd = ConvolvedKSD(target=target, kernel=kernel, conv_kernel=None)
            bootstrap = Bootstrap(ksd, n)

            # perform test
            _, p_val = bootstrap.test_once(
                alpha=alpha, 
                num_boot=num_boot, 
                X=sample_test, 
                multinom_samples=multinom_samples[seed, :], 
                conv_samples_full=conv_sample_full,
                conv_samples=conv_sample_test,
                log_noise_std=best_log_noise_std
            )
            ksd_df.append([n, p_val, tf.exp(best_log_noise_std).numpy()**2, seed, dist_name])

    ksd_df = pd.DataFrame(ksd_df, columns=["n", "p_value", "var_est", "seed", "type"])
    return ksd_df

parser = argparse.ArgumentParser()
nrep = 1000
num_boot = 1000 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
delta_list = [1., 2., 4., 6.]
dim = 5
num_est = 10000 # num samples used to estimate concolved target
# grid of noise vars
noise_std_list = [float(2**x) for x in range(-2, 7)]
log_noise_std_list = [tf.math.log(x) for x in noise_std_list]

if __name__ == '__main__':
    parser.add_argument("--load", type=str, default="", help="path to pre-saved results")
    args = parser.parse_args()

    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 9))
    subfigs = fig.subfigures(1, len(delta_list))
    for ind, delta in enumerate(delta_list):
        print(f"Running with delta = {delta}")
        test_imq_df = None

        # define noise distribution
        convolution = tfd.MultivariateNormalDiag(0., tf.ones(dim))

        # target distribution
        target = create_mixture_gaussian(dim=dim, delta=delta)

        # off-target proposal distribution
        proposal_on = create_mixture_gaussian(dim=dim, delta=delta)
        
        # off-target proposal distribution
        proposal_mean = - delta * tf.eye(dim)[:, 0]
        proposal_off = tfd.MultivariateNormalDiag(proposal_mean)

        if len(args.load) > 0 :
            try:
                test_imq_df = pd.read_csv(args.load + f"/multiple_delta{delta}.csv")
                print(f"Loaded pre-saved data for delta={delta}")
            except:
                print(f"Pre-saved data for delta={delta} not found. Running from scratch now.")

        if test_imq_df is None:
            # with IMQ
            imq = IMQ(med_heuristic=True)
            test_imq_df = run_bootstrap_experiment(nrep, target, proposal_on, proposal_off, convolution, imq, alpha, num_boot, num_est, log_noise_std_list)

        # save res
        test_imq_df.to_csv(f"res/bootstrap/multiple_delta{delta}.csv", index=False)

        # plot
        subfig = subfigs.flat[ind]
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(4, 1)
        axs = axs.flat

        var = tf.constant(test_imq_df.loc[test_imq_df.type == "off-target", "var_est"].median(), dtype=tf.float32)
        convolution = tfd.MultivariateNormalDiag(0., tf.math.sqrt(var) * tf.ones(dim))
        convolution_sample = convolution.sample(10000)
        axs[0].hist((proposal_off.sample(10000) + convolution_sample).numpy()[:, 0], label="off-target", alpha=0.2)
        axs[0].hist((proposal_on.sample(10000) + convolution_sample).numpy()[:, 0], label="target", alpha=0.2)
        axs[0].legend()
        axs[0].set_title("convolved with median var over repetitions ({:.3g})".format(var.numpy()))

        sns.ecdfplot(ax=axs[1], data=test_imq_df.loc[test_imq_df.type.isin(["off-target", "off-target noiseless"])], x="p_value", hue="type")
        axs[1].plot([0, 1], [0, 1], transform=axs[1].transAxes, color="grey", linestyle="dashed")
        axs[1].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        err = (test_imq_df.loc[test_imq_df.type == "off-target", "p_value"] > alpha).mean()
        axs[1].set_title(f"off target (type II error = {err})")
        axs[1].set_xlabel("p-value")
        
        sns.ecdfplot(ax=axs[2], data=test_imq_df.loc[test_imq_df.type.isin(["target", "target noiseless"])], x="p_value", hue="type")
        axs[2].plot([0, 1], [0, 1], transform=axs[2].transAxes, color="grey", linestyle="dashed")
        axs[2].axis(xmin=-0.01, xmax=1., ymin=0, ymax=1.01)
        err = (test_imq_df.loc[test_imq_df.type == "target", "p_value"] <= alpha).mean()
        axs[2].set_title(f"On target (type I error = {err})")
        axs[2].set_xlabel("p-value")

        # sns.histplot(ax=axs[3], 
        #     data=test_imq_df.loc[test_imq_df.type.isin(["off-target", "target"])], 
        #     x="var_est", bins=20, hue="type", alpha=0.2)
        sns.ecdfplot(ax=axs[3], data=test_imq_df.loc[test_imq_df.type.isin(["off-target", "target"])], x="var_est", hue="type")
        axs[3].axis(ymin=0, ymax=1.01)
        axs[3].set_xscale("log")
        axs[3].set_title("Best var")

    fig.savefig("figs/bootstrap/bootstrap_convolved_multiple.png")

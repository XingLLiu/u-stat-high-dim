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

from src.ksd.ksd import KSD, ConvolvedKSD
from src.ksd.kernel import RBF, IMQ
from src.ksd.bootstrap import Bootstrap
import src.ksd.models as models
import src.ksd.langevin as mcmc
from src.ksd.find_modes import find_modes, pairwise_directions

def run_bootstrap_experiment(nrep, target, proposal_on, log_prob_fn, proposal_off, kernel, alpha, num_boot, T, 
    std_ls, random_start_pts=False, method="mcmc", **kwargs):
    """compute KSD and repeat for nrep times"""
    
    n = kwargs["n"]
    ntrain = int(n*0.5)
    dim = proposal_off.event_shape[0]

    if method == "mcmc":
        ksd = KSD(target=target, kernel=kernel)
        MCMCKernel = kwargs["mcmckernel"]
    elif method == "conv" or method == "convvar":
        ksd = ConvolvedKSD(target=target, kernel=kernel)

    grad_log = None # if not hasattr(target, "grad_log") else target.grad_log

    ksd_df = pd.DataFrame(columns=["n", "best_std", "p_value", "seed", "type"])
    iterator = trange(nrep)
    bootstrap = Bootstrap(ksd, n-ntrain)
    multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x ntest

    bootstrap_nopert = Bootstrap(ksd, n)
    multinom_samples_notrain = bootstrap_nopert.multinom.sample((nrep, num_boot)) # nrep x num_boot x n

    names = ["off-target", "target", "off-target no pert", "target no pert"]
    proposals = [proposal_off, proposal_on, proposal_off, proposal_on]

    if random_start_pts:
        # generate points for finding modes
        unif_dist = tfp.distributions.Uniform(low=-tf.ones((dim,)), high=tf.ones((dim,)))
        start_pts_all = 20. * unif_dist.sample((nrep, kwargs["nstart_pts"])) # change range

    if method == "conv" or method == "convvar":
        # initialise optimiser
        num_steps = 200 if method == "conv" else 50
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

        # initialise noise samples
        num_est = 10000
        convolution = tfd.MultivariateNormalDiag(0., tf.ones(dim))
        conv_samples_full_all = convolution.sample((nrep, num_est))[:, :, :1] # for p; 1D Gaussian
        
        # initialise indices for samples for Q
        conv_ind_all = tf.experimental.numpy.random.randint(low=0, high=num_est, size=(nrep, ntrain))
        conv_ind_all_notrain = tf.experimental.numpy.random.randint(low=0, high=num_est, size=(nrep, n))

    iterator.set_description(f"Running with sample size {n}")
    for seed in iterator:

        for name, proposal in zip(names, proposals):
            sample_init = proposal.sample(n)

            # use Markov kernel
            if method == "mcmc":
                if "no pert" not in name:
                    sample_init_train, sample_init_test = sample_init[:ntrain, ], sample_init[ntrain:, ]
                    
                    # start optim from either randomly initialised points or training samples
                    start_pts = start_pts_all[seed, :, :] if random_start_pts else sample_init_train
                    # merge modes
                    mode_list, inv_hess_list = find_modes(start_pts, log_prob_fn, grad_log=grad_log, **kwargs)

                    # find between-modes dir
                    if len(mode_list) == 1:
                        _, ind_pair_list = [mode_list[0]], [(0, 0)]
                    else:
                        _, ind_pair_list = pairwise_directions(mode_list, return_index=True)

                    # find v_{i^*, j^*}, \sigma^*_{i^*, j^*}
                    best_ksd = 0.
                    for j in range(len(ind_pair_list)):
                        # loop through mode pairs

                        # find estimated hessians
                        ind1, ind2 = ind_pair_list[j]
                        mode1, mode2 = mode_list[ind1], mode_list[ind2]
                        hess1_inv, hess2_inv = inv_hess_list[ind1], inv_hess_list[ind2]

                        proposal_dict = mcmc.prepare_proposal_input(
                            mode1=mode1, mode2=mode2, hess1_inv=hess1_inv, hess2_inv=hess2_inv)

                        for i, std in enumerate(std_ls):
                            # loop through jump scales
                            iterator.set_description(f"Jump scale [{i+1} / {len(std_ls)}] of dir vector [{j+1} / {len(ind_pair_list)}]")

                            # run dynamic for T steps
                            mh = MCMCKernel(log_prob=log_prob_fn)
                            mh.run(steps=T, std=std, x_init=sample_init_train, **proposal_dict)

                            # compute ksd
                            x_t = mh.x[-1, :, :].numpy()
                            _, ksd_val = ksd.h1_var(X=x_t, Y=tf.identity(x_t), return_scaled_ksd=True)
                            ksd_val = ksd_val.numpy()

                            # update if ksd is larger
                            if (ksd_val > best_ksd) or (i == 0):
                                best_std = std
                                best_ksd = ksd_val
                                best_proposal_dict = proposal_dict

                    # run dynamic for T steps with test data and optimal params
                    mh = MCMCKernel(log_prob=log_prob_fn)
                    mh.run(steps=T, x_init=sample_init_test, std=best_std, **best_proposal_dict)

                    # get perturbed samples
                    x_t = mh.x[-1, :, :].numpy()

                    # get multinomial sample
                    multinom_one_sample = multinom_samples[seed, :] # nrep x num_boost x ntest
                
                else:
                    x_t = sample_init
                    multinom_one_sample = multinom_samples_notrain[seed, :] # nrep x num_boost x n

                # compute p-value
                _, p_val = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=x_t, multinom_samples=multinom_one_sample)
                ksd_df.loc[len(ksd_df)] = [n, best_std, p_val, seed, name]

            # use convolution
            elif method == "conv":
                if "no pert" not in name:
                    sample_init_train, sample_init_test = sample_init[:ntrain, ], sample_init[ntrain:, ]
                    
                    # initialise noise variance and direction
                    log_sigma = tf.Variable([0.])
                    u_vec = tf.eye(dim)[:, 1] #TODO need to make sure this is not hacked
                    params = tf.Variable(tf.concat([log_sigma, u_vec], axis=0))

                    # get noise samples for p and Q
                    conv_samples_full = conv_samples_full_all[seed, :] # l x 1
                    conv_samples = tf.gather(conv_samples_full, conv_ind_all[seed, :], axis=0) # n x 1

                    # optimise
                    ksd.optim(
                        nsteps=num_steps,
                        optimizer=optimizer,
                        param=params,
                        X=sample_init_train,
                        Y=tf.identity(sample_init_train),
                        conv_samples_full=conv_samples_full,
                        conv_samples=conv_samples,
                        desc=iterator,
                    )
                    best_params = ksd.params[-1]
                    best_proposal_dict = {
                        "log_noise_std": best_params[0],
                        "u": best_params[1:]}

                    # get samples to compute p-value
                    x_t = sample_init_test

                    # get multinomial sample
                    multinom_one_sample = multinom_samples[seed, :] # nrep x num_boost x ntest

                else:
                    # get noise samples for p and Q
                    conv_samples_full = conv_samples_full_all[seed, :] # l x 1
                    conv_samples = tf.gather(conv_samples_full, conv_ind_all_notrain[seed, :], axis=0) # n x 1

                    x_t = sample_init
                    multinom_one_sample = multinom_samples_notrain[seed, :] # nrep x num_boost x n

                _, p_val = bootstrap.test_once(
                    alpha=0.05, 
                    num_boot=num_boot, 
                    X=x_t, 
                    multinom_samples=multinom_one_sample, 
                    conv_samples_full=conv_samples_full,
                    conv_samples=conv_samples,
                    **best_proposal_dict
                )
                best_std = tf.exp(best_proposal_dict["log_noise_std"]).numpy()
                ksd_df.loc[len(ksd_df)] = [n, best_std, p_val, seed, name]

            # use convolution and only optimise for var
            elif method == "convvar":
                if "no pert" not in name:
                    sample_init_train, sample_init_test = sample_init[:ntrain, ], sample_init[ntrain:, ]

                    # get noise samples for p and Q
                    conv_samples_full = conv_samples_full_all[seed, :] # l x 1
                    conv_samples = tf.gather(conv_samples_full, conv_ind_all[seed, :], axis=0) # n x 1

                    # start optim from either randomly initialised points or training samples
                    start_pts = start_pts_all[seed, :, :] if random_start_pts else sample_init_train
                    # merge modes
                    mode_list, inv_hess_list = find_modes(start_pts, log_prob_fn, grad_log=grad_log, **kwargs)

                    # find between-modes dir
                    if len(mode_list) == 1:
                        _, ind_pair_list = [mode_list[0]], [(0, 0)]
                    else:
                        _, ind_pair_list = pairwise_directions(mode_list, return_index=True)
                    
                    best_ksd = 0.
                    best_proposal_dict = {"log_noise_std": 0., "u": tf.zeros((1, dim))}
                    ind_pair_list_len = len(ind_pair_list)
                    for j in range(ind_pair_list_len):
                        # loop through mode pairs
                        iterator.set_description(f"Mode pair [{j+1} / {ind_pair_list_len}]")

                        # find estimated hessians
                        ind1, ind2 = ind_pair_list[j]
                        mode1, mode2 = mode_list[ind1], mode_list[ind2]
                        u_vec = tf.reshape(mode1 - mode2, (1, -1)) # 1 x dim
                        u_vec_n = u_vec / tf.sqrt(tf.reduce_sum(u_vec**2)) # 1 x dim

                        # initialise log std for noise
                        param = tf.Variable([1.])

                        # optimise
                        ksd.optim_var(
                            nsteps=num_steps,
                            optimizer=optimizer,
                            param=param,
                            u_vec=u_vec_n,
                            X=sample_init_train,
                            Y=tf.identity(sample_init_train),
                            conv_samples_full=conv_samples_full,
                            conv_samples=conv_samples,
                        )
                        if -ksd.losses[-1] > best_ksd: # -loss = ksd
                            best_proposal_dict["log_noise_std"] = ksd.params[-1][0]
                            best_proposal_dict["u"] = u_vec_n
                    
                    # get samples to compute p-value
                    x_t = sample_init_test

                    # get multinomial sample
                    multinom_one_sample = multinom_samples[seed, :] # nrep x num_boost x ntest

                else:
                    # get noise samples for p and Q
                    conv_samples_full = conv_samples_full_all[seed, :] # l x 1
                    conv_samples = tf.gather(conv_samples_full, conv_ind_all_notrain[seed, :], axis=0) # n x 1

                    x_t = sample_init
                    multinom_one_sample = multinom_samples_notrain[seed, :] # nrep x num_boost x n

                _, p_val = bootstrap.test_once(
                    alpha=0.05, 
                    num_boot=num_boot, 
                    X=x_t, 
                    multinom_samples=multinom_one_sample, 
                    conv_samples_full=conv_samples_full,
                    conv_samples=conv_samples,
                    **best_proposal_dict
                )
                best_std = tf.exp(best_proposal_dict["log_noise_std"]).numpy()
                ksd_df.loc[len(ksd_df)] = [n, best_std, p_val, seed, name]

    return ksd_df, best_proposal_dict

num_boot = 800 # number of bootstrap samples to compute critical val
alpha = 0.05 # significant level
mode_threshold = 1. # threshold for merging modes
sigma_list = np.linspace(0.5, 1.5, 21).tolist() # std for discrete jump proposal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="", help="path to pre-saved results")
    parser.add_argument("--model", type=str, default="bimodal")
    parser.add_argument("--mcmckernel", type=str, default="mh")
    parser.add_argument("--method", type=str, default="mcmc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000, help="sample size")
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--nrep", type=int, default=50)
    parser.add_argument("--ratio_t", type=float, default=0.5, help="max num of steps")
    parser.add_argument("--ratio_s", type=float, default=1.)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--nmodes", type=int, default=10)
    parser.add_argument("--nbanana", type=int, default=2)
    parser.add_argument("--shift", type=float, default=0.)
    parser.add_argument("--dh", type=int, default=10, help="dim of h for RBM")
    args = parser.parse_args()
    model = args.model
    method = args.method
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
    
    if args.mcmckernel == "mh":
        MCMCKernel = mcmc.RandomWalkMH
        mcmc_name = "mh_"
    elif args.mcmckernel == "barker":
        MCMCKernel = mcmc.RandomWalkBarker
        mcmc_name = "barker_"
    if args.method == "conv":
        mcmc_name = "conv_"
    elif args.method == "convvar":
        mcmc_name = "convvar_"

    tf.random.set_seed(seed)

    fig = plt.figure(constrained_layout=True, figsize=(5*len(delta_list), 15))
    subfigs = fig.subfigures(1, len(delta_list))

    for ind, delta in tqdm(enumerate(delta_list)):
        print(f"Running with delta = {delta}")
        test_imq_df = None
        
        # set model
        if model == "bimodal":
            model_name = f"{mcmc_name}{model}_steps{T}_ratio{ratio_target}_{ratio_sample}_k{k}_seed{seed}"
            create_target_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, return_logprob=True, ratio=ratio_target)
            create_sample_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, return_logprob=True, ratio=ratio_sample)
        
        elif model == "bimodal_shift":
            shift = args.shift
            model_name = f"{mcmc_name}{model}_steps{T}_ratio{ratio_target}_{ratio_sample}_k{k}_shift{shift}_seed{seed}"
            create_target_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, shift=shift, return_logprob=True, ratio=ratio_target)
            create_sample_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, return_logprob=True, ratio=ratio_sample)

        elif model == "t-banana":
            nmodes = args.nmodes
            nbanana = args.nbanana
            model_name = f"{mcmc_name}{model}_steps{T}_nmodes{nmodes}_seed{seed}"
            ratio_target = [1/nmodes] * nmodes
            random_weights = tfp.distributions.Uniform(low=0., high=1.).sample(nmodes)
            ratio_sample = random_weights / tf.reduce_sum(random_weights)
            # ratio_sample = [0.3, 0.7] #TODO
            print("ratio sample:", ratio_sample.numpy())
            loc = tfp.distributions.Uniform(low=-tf.ones((dim,))*10, high=tf.ones((dim,))*20).sample(nmodes) # uniform in [-20, 20]^d

            # loc = tf.constant([[4.1144876, 6.538103, 4.1443825, 4.311162, -9.23945],
            #     [-4.1144876, -6., 0.08298874, -1.2335253, 7.6566525],
            #     [-4.977256, 16.026318, 8.160973, 12.019077, -6.351266],
            #     [ 5.583577, 10.599161, 13.44928, 1.0351582, 3.1638136 ]])
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
            model_name = f"{mcmc_name}{model}{nmodes}_steps{T}_seed{seed}"
            means = tfp.distributions.Uniform(low=-tf.ones((dim,))*20, high=tf.ones((dim,))*20).sample(nmodes) # uniform in [-5, 5]^d
            # means = tf.constant([[4.1144876, 6.538103, 4.1443825, 4.311162, -9.23945],
            #     [-4.1144876, -6., 0.08298874, -1.2335253, 7.6566525],
            #     [-4.977256, 16.026318, 8.160973, 12.019077, -6.351266],
            #     [ 5.583577, 10.599161, 13.44928, 1.0351582, 3.1638136 ]])
            means = tf.constant([[-15.692321, -1.1227798, -15.462275, -7.772279, 0.45660973],
                [ 13.653507, 12.54893875, -16.00213, -17.85245, -19.995266],
                [  10.38526535, -5.5862007, 2.0345726, -12.112765, -19.61071],
                [  25.736, 25.068437, -16.550621, -9.046288, -18.059994],
                [-15., 10., 2., 8., -9.]])
            means = means[:nmodes, :]

            indicator = tf.cast(tfp.distributions.Bernoulli(probs=0.5).sample(nmodes-1), dtype=bool)
            indicator = tf.concat([tf.constant([True]), indicator], axis=0)
            random_weights = tfp.distributions.Uniform(low=0., high=1.).sample(nmodes)
            random_weights = tf.where(indicator, random_weights, 0.)
            # ratio_sample = random_weights / tf.reduce_sum(random_weights)
            ratio_sample = [0.19549179, 0.1501031, 0.50429535, 0.15010974, 0.] #! delete
            ratio_sample = ratio_sample[:nmodes] #! delete

            ratio_target = 0.5
            delta = 1. #! delete

            create_target_model = models.create_mixture_20_gaussian(means, ratio=ratio_target, scale=delta, return_logprob=True)
            create_sample_model = models.create_mixture_20_gaussian(means, ratio=ratio_sample, scale=delta, return_logprob=True)

        elif model == "gauss-scaled":
            model_name = f"{mcmc_name}{model}_steps{T}_seed{seed}"
            create_target_model = models.create_mixture_gaussian_scaled(ratio=ratio_target, return_logprob=True)
            create_sample_model = models.create_mixture_gaussian_scaled(ratio=ratio_sample, return_logprob=True)

        elif model == "rbm":
            dh = args.dh
            c_shift = args.shift
            c_off = tf.constant([c_shift, c_shift] + [0.] * (dh - 2))

            model_name = f"{mcmc_name}{model}_steps{T}_seed{seed}_dim{dim}_dh{dh}_shift{c_shift}"
            create_target_model = models.create_rbm(c=0., dx=dim, dh=dh, return_logprob=True)
            create_sample_model = models.create_rbm(c=c_off, dx=dim, dh=dh, return_logprob=True)


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
            test_imq_df, best_hess_dict = run_bootstrap_experiment(
                nrep, target, proposal_on, log_prob_fn, proposal_off, imq,
                alpha, num_boot, T, sigma_list, threshold=mode_threshold, nstart_pts=nstart_pts, n=n,
                mcmckernel=MCMCKernel, method=method)

            # save res
            test_imq_df.to_csv(f"res/bootstrap/{model_name}_delta{delta}.csv", index=False)
            pickle.dump(best_hess_dict, open(f"res/bootstrap/{model_name}_delta{delta}.pkl", "wb"))

        # between-modes vector (only for plotting)
        best_hess_dict = pickle.load(open(f"res/bootstrap/{model_name}_delta{delta}.pkl", "rb"))

        # for plotting dist of perturbed samples
        if method == "mcmc":
            off_sample = proposal_off.sample(1000)
            best_std_off = test_imq_df.loc[(test_imq_df.type == "off-target"), "best_std"].median()
            mh_off = MCMCKernel(log_prob=log_prob_fn)
            mh_off.run(steps=T, std=best_std_off, x_init=off_sample, **best_hess_dict)
            off_sample_init = off_sample.numpy()
            off_sample_pert = mh_off.x[-1, :, :].numpy()

            on_sample = proposal_on.sample(1000)
            best_std_on = test_imq_df.loc[(test_imq_df.type == "target"), "best_std"].median()
            mh_on = MCMCKernel(log_prob=log_prob_fn)
            mh_on.run(steps=T, std=best_std_on, x_init=on_sample, **best_hess_dict)
            on_sample_init = on_sample.numpy()
            on_sample_pert = mh_on.x[-1, :, :].numpy()

        elif method == "conv" or method == "convvar":
            u_vec = tf.reshape(best_hess_dict["u"], (1, -1)) # 1 x dim
            convolution = tfd.MultivariateNormalDiag(0., tf.ones(dim))
            convolution_sample = convolution.sample(1000)[:, :1] # n x 1

            off_sample = proposal_off.sample(1000)
            best_std_off = test_imq_df.loc[(test_imq_df.type == "off-target"), "best_std"].median()
            convolution_sample_off = convolution_sample @ u_vec * best_std_off
            off_sample_init = off_sample.numpy()
            off_sample_pert = (off_sample + convolution_sample_off).numpy()

            on_sample = proposal_on.sample(1000)
            best_std_on = test_imq_df.loc[(test_imq_df.type == "target"), "best_std"].median()
            convolution_sample_on = convolution_sample @ u_vec * best_std_on
            on_sample_init = on_sample.numpy()
            on_sample_pert = (on_sample + convolution_sample_on).numpy()

        # plot
        subfig = subfigs.flat[ind] if len(delta_list) > 1 else subfigs
        subfig.suptitle(f"delta = {delta}")
        axs = subfig.subplots(5, 1)
        axs = axs.flat
        xlim, ylim = 30, 30

        samples_df_off_target = pd.DataFrame({
            "x1": off_sample_init[:, 0],
            "type": "off-target"})
        samples_df_off_perturbed = pd.DataFrame({
            "x1": off_sample_pert[:, 0],
            "type": "perturbed off-target"})

        samples_df_target = pd.DataFrame({
            "x1": on_sample_init[:, 0],
            "type": "target"})
        samples_df_perturbed = pd.DataFrame({
            "x1": on_sample_pert[:, 0],
            "type": "perturbed target"})
        if on_sample.shape[-1] > 1:
            samples_df_off_target["x2"] = off_sample_init[:, 1]
            samples_df_off_perturbed["x2"] = off_sample_pert[:, 1]

            samples_df_target["x2"] = on_sample_init[:, 1]
            samples_df_perturbed["x2"] = on_sample_pert[:, 1]

        samples_df_off = pd.concat([samples_df_off_target, samples_df_off_perturbed], ignore_index=True)
        samples_df = pd.concat([samples_df_target, samples_df_perturbed], ignore_index=True)

        if on_sample.shape[-1] > 1:
            # sns.kdeplot(ax=axs[0], data=samples_df_off, x="x1", y="x2", hue="type", alpha=0.8)
            sns.scatterplot(ax=axs[0], data=samples_df_off, x="x1", y="x2", hue="type", alpha=0.5)
            axs[0].axis(xmin=-xlim, xmax=xlim, ymin=-ylim, ymax=ylim)
        else:
            sns.ecdfplot(ax=axs[0], data=samples_df_off, x="x1", hue="type")
        axs[0].set_ylabel("CDF")
        if ind != len(delta_list) - 1: axs[0].legend([],[], frameon=False)

        if on_sample.shape[-1] > 1:
            # sns.kdeplot(ax=axs[1], data=samples_df, x="x1", y="x2", hue="type", alpha=0.8)
            sns.scatterplot(ax=axs[1], data=samples_df, x="x1", y="x2", hue="type", alpha=0.5)
            axs[1].axis(xmin=-xlim, xmax=xlim, ymin=-ylim, ymax=ylim)
        else:
            sns.ecdfplot(ax=axs[1], data=samples_df, x="x1", hue="type")
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
        if method == "mcmc":
            axs[4].axis(xmin=0.4, xmax=1.6, ymin=0, ymax=1.01)
        elif method == "conv" or method == "convvar":
            axs[4].axis(ymin=0, ymax=1.01)
        axs[4].set_title("median of estimated best std = {:.2f} (off),z {:.2f} (on)".format(best_std_off, best_std_on))
        axs[4].set_xlabel("std")
        if ind != len(delta_list) - 1: axs[4].legend([],[], frameon=False)

    fig.savefig(f"figs/bootstrap/{model_name}.png")

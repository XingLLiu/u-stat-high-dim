import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import argparse

import pytorch_lightning as pl

from src.ksd.ksd import KSD, ConvolvedKSD
from src.ksd.kernel import RBF, IMQ
from src.ksd.bootstrap import Bootstrap
import src.ksd.models as models
import src.ksd.langevin as mcmc
from src.ksd.find_modes import find_modes, pairwise_directions
from src.kgof.ksdagg import ksdagg_wild_test


def run_bootstrap_experiment(nrep, target, log_prob_fn, proposal, kernel, alpha, num_boot, T, 
    jump_ls, random_start_pts=False, method="mcmc", **kwargs):
    """compute KSD and repeat for nrep times"""
    
    n = kwargs["n"]
    ntrain = int(n*0.5)
    dim = proposal.event_shape[0]

    if method == "mcmc":
        ksd = KSD(target=target, kernel=kernel)
        MCMCKernel = kwargs["mcmckernel"]
    elif method == "conv":
        ksd = ConvolvedKSD(target=target, kernel=kernel)

    grad_log = None if not hasattr(target, "grad_log") else target.grad_log

    iterator = trange(nrep)

    bootstrap = Bootstrap(ksd, n-ntrain)
    multinom_samples = bootstrap.multinom.sample((nrep, num_boot)) # nrep x num_boot x ntest

    bootstrap_nopert = Bootstrap(ksd, n)
    multinom_samples_notrain = bootstrap_nopert.multinom.sample((nrep, num_boot)) # nrep x num_boot x n

    res_df = pd.DataFrame(columns=["method", "rej", "seed"])
    
    if random_start_pts:
        print("Use randomly initialised points")
        # generate points for finding modes
        unif_dist = tfp.distributions.Uniform(low=-tf.ones((dim,)), high=tf.ones((dim,)))
        start_pts_all = 20. * unif_dist.sample((nrep, ntrain)) #TODO change range
        
        ## for NF initialisation
        # unif_dist = tfp.distributions.Uniform(low=tf.zeros((dim,)), high=255.*tf.ones((dim,)))
        # start_pts_all = unif_dist.sample((nrep, ntrain)) # change range

    if method == "conv":
        # initialise optimiser
        num_steps = 200
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

        # initialise noise samples
        num_est = 10000
        convolution = tfd.MultivariateNormalDiag(0., tf.ones(dim))
        conv_samples_full_all = convolution.sample((nrep, num_est))
        
        # initialise indices for samples for Q
        conv_ind_all = tf.experimental.numpy.random.randint(low=0, high=num_est, size=(nrep, ntrain))

    iterator.set_description(f"Running with sample size {n}")
    for iter in iterator:
        # each sample is different
        tf.random.set_seed(iter + n)
        # pl.seed_everything(iter + n)

        sample_init = proposal.sample(n)

        ## KSD
        iterator.set_description("Running KSD")
        x_t = sample_init
        multinom_one_sample = multinom_samples_notrain[iter, :] # nrep x num_boost x n

        # compute p-value
        ksd_rej, _ = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=x_t, multinom_samples=multinom_one_sample)
        res_df.loc[len(res_df)] = ["KSD", ksd_rej, iter]


        ## pKSD with MCMC kernel with uniformly chosen modes
        if method == "mcmc":
            iterator.set_description("Running pKSD mcmc")
    
            # train/test split
            sample_init_train, sample_init_test = sample_init[:ntrain, ], sample_init[ntrain:, ]
            
            # start optim from either randomly initialised points or training samples
            start_pts = start_pts_all[iter, :, :] if random_start_pts else sample_init_train
            # merge modes
            mode_list, inv_hess_list = find_modes(start_pts, log_prob_fn, grad_log=grad_log, **kwargs)

            # find between-modes dir
            if len(mode_list) == 1:
                _, ind_pair_list = [mode_list[0]], [(0, 0)]
            else:
                _, ind_pair_list = pairwise_directions(mode_list, return_index=True)
            
            proposal_dict = mcmc.prepare_proposal_input_all(mode_list=mode_list, inv_hess_list=inv_hess_list)

            best_pksd = 0.
            # find best jump scale \gamma^*
            for i, jump in enumerate(jump_ls):
                # loop through jump scales
                iterator.set_description(f"Jump scale [{i+1} / {len(jump_ls)}]]")

                # run dynamic for T steps
                mh = MCMCKernel(log_prob=log_prob_fn)
                mh.run(steps=T, std=jump, x_init=sample_init_train, ind_pair_list=ind_pair_list, **proposal_dict)

                # compute ksd
                x_t = mh.x[-1, :, :].numpy()
                _, ksd_val = ksd.h1_var(X=x_t, Y=tf.identity(x_t), return_scaled_ksd=True)
                ksd_val = ksd_val.numpy()

                # update if ksd is larger
                if (ksd_val > best_pksd) or (i == 0):
                    best_jump = jump
                    best_pksd = ksd_val
                    best_proposal_dict = proposal_dict

            # run dynamic for T steps with test data and optimal params
            mh = MCMCKernel(log_prob=log_prob_fn)
            mh.run(steps=T, x_init=sample_init_test, std=best_jump, ind_pair_list=ind_pair_list, **best_proposal_dict)

            # gather proposal params into one dict
            best_proposal_dict = {**best_proposal_dict, "ind_pair_list": ind_pair_list}

            # get perturbed samples
            x_t = mh.x[-1, :, :].numpy()

            # get multinomial sample
            multinom_one_sample = multinom_samples[iter, :] # nrep x num_boost x ntest
        
            # compute p-value
            pksd_rej, _ = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=x_t, multinom_samples=multinom_one_sample)
            res_df.loc[len(res_df)] = ["pKSD mc", pksd_rej, iter]

        
        elif method == "conv":
            iterator.set_description("Running pKSD conv")

            ## pKSD with convolution
            sample_init_train, sample_init_test = sample_init[:ntrain, ], sample_init[ntrain:, ]
            
            # initialise noise variance and direction
            log_sigma = tf.Variable([0.])
            u_vec = tf.eye(dim)[:, 1] #TODO need to make sure this is not hacked
            params = tf.Variable(tf.concat([log_sigma, u_vec], axis=0))

            # get noise samples for p and Q
            conv_samples_full = conv_samples_full_all[iter, :] # l x 1
            conv_samples = tf.gather(conv_samples_full, conv_ind_all[iter, :], axis=0) # n x 1

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
            multinom_one_sample = multinom_samples[iter, :] # nrep x num_boost x ntest

            pksd_rej, _ = bootstrap.test_once(
                alpha=0.05, 
                num_boot=num_boot, 
                X=x_t, 
                multinom_samples=multinom_one_sample, 
                conv_samples_full=conv_samples_full,
                conv_samples=conv_samples,
                **best_proposal_dict
            )
            res_df.loc[len(res_df)] = ["pKSD conv", pksd_rej, iter]


        ## KSDAGG
        iterator.set_description("Running KSDAGG")
        x_t = sample_init
        ksdagg_rej = ksdagg_wild_test(
            seed=iter + n,
            X=x_t,
            log_prob_fn=log_prob_fn,
            alpha=alpha,
            beta_imq=0.5,
            kernel_type="imq",
            weights_type="uniform",
            l_minus=0,
            l_plus=10,
            B1=num_boot,
            B2=500, # num of samples to estimate level
            B3=50, # num of bisections to estimate quantile
        )
        res_df.loc[len(res_df)] = ["KSDAGG", ksdagg_rej, iter]

    return res_df


num_boot = 800 # number of bootstrap samples to compute critical val
alpha = 0.05 # test level
mode_threshold = 1. # threshold for merging modes
jump_ls = np.linspace(0.8, 1.2, 11).tolist() # std for discrete jump proposal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="", help="path to pre-saved results")
    parser.add_argument("--model", type=str, default="bimodal")
    parser.add_argument("--mcmckernel", type=str, default="mh")
    parser.add_argument("--method", type=str, default="mcmc")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000, help="sample size")
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--nrep", type=int, default=50)
    parser.add_argument("--ratio_t", type=float, default=0.5, help="max num of steps")
    parser.add_argument("--ratio_s", type=float, default=1.)
    parser.add_argument("--delta", type=float, default=8.)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--nmodes", type=int, default=10)
    parser.add_argument("--nbanana", type=int, default=2)
    parser.add_argument("--shift", type=float, default=0.)
    parser.add_argument("--dh", type=int, default=10, help="dim of h for RBM")
    parser.add_argument("--ratio_s_var", type=float, default=0.)
    parser.add_argument("--rand_start", type=bool, default=False)
    parser.add_argument("--t_std", type=float, default=0.01)
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
    delta = args.delta # [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
    rand_start = args.rand_start
    
    if args.mcmckernel == "mh":
        MCMCKernel = mcmc.RandomWalkMH
        mcmc_name = "mh"
    elif args.mcmckernel == "barker":
        MCMCKernel = mcmc.RandomWalkBarker
        mcmc_name = "barker"

    if method == "mcmc_all":
        mcmc_name = f"{mcmc_name}_all"
    
    if method != "mcmc" and method != "mcmc_all":
        mcmc_name = args.method

    # suffix for random starting points
    rnd_st_suff = "_rnd" if rand_start else None

    # create folder for experiment
    res_root = f"res/{model}"
    fig_root = f"figs/{model}"
    tf.io.gfile.makedirs(res_root)
    tf.io.gfile.makedirs(fig_root)

    # set random seed for all
    rdg = tf.random.Generator.from_seed(seed)
    # pl.seed_everything(seed)
    
    # set model
    if model == "bimodal":
        model_name = f"{mcmc_name}_steps{T}_ratio{ratio_target}_{ratio_sample}_k{k}_dim{dim}_seed{seed}_delta{delta}_n{n}"
        create_target_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, return_logprob=True, ratio=ratio_target)
        create_sample_model = models.create_mixture_gaussian_kdim(dim=dim, k=k, delta=delta, return_logprob=True, ratio=ratio_sample)

    elif model == "t-banana":
        nmodes = args.nmodes
        nbanana = args.nbanana
        model_name = f"{mcmc_name}_steps{T}_dim{dim}_nmodes{nmodes}_nbanana{nbanana}_ratiosvar{args.ratio_s_var}_t-std{args.t_std}_n{n}_seed{seed}"
        ratio_target = [1/nmodes] * nmodes
        
        random_weights = tf.exp(rdg.normal((nmodes,)) * args.ratio_s_var)
        ratio_sample = random_weights / tf.reduce_sum(random_weights)

        loc = rdg.uniform((nmodes, dim), minval=-tf.ones((dim,))*20, maxval=tf.ones((dim,))*20) # uniform in [-20, 20]^d
        print(loc)

        b = 0.003 # 0.03
        create_target_model = models.create_mixture_t_banana(dim=dim, ratio=ratio_target, loc=loc, b=b,
            nbanana=nbanana, std=args.t_std, return_logprob=True)
        create_sample_model = models.create_mixture_t_banana(dim=dim, ratio=ratio_sample, loc=loc, b=b,
            nbanana=nbanana, std=args.t_std, return_logprob=True)

    elif model == "gauss-scaled":
        model_name = f"{mcmc_name}_steps{T}_seed{seed}"
        create_target_model = models.create_mixture_gaussian_scaled(ratio=ratio_target, return_logprob=True)
        create_sample_model = models.create_mixture_gaussian_scaled(ratio=ratio_sample, return_logprob=True)

    elif model == "rbm":
        dh = args.dh
        c_shift = args.shift
        c_off = tf.concat([tf.ones(2) * c_shift, tf.zeros(dh-2)], axis=0)

        model_name = f"{mcmc_name}_steps{T}_seed{seed}_dim{dim}_dh{dh}_shift{c_shift}"
        create_target_model = models.create_rbm(B_scale=6., c=0., dx=dim, dh=dh, burnin_number=2000, return_logprob=True)
        create_sample_model = models.create_rbm(B_scale=6., c=c_off, dx=dim, dh=dh, burnin_number=2000, return_logprob=True)

    elif model == "nf":
        model_name = f"{mcmc_name}{rnd_st_suff}_steps{T}_seed{seed}_n{n}"
        create_target_model = models.generate_nf_mnist(real_mnist=False, return_logprob=True)
        create_sample_model = models.generate_nf_mnist(real_mnist=True, return_logprob=True)

    print(f"Running {model_name}")

    # target distribution
    target, log_prob_fn = create_target_model

    # proposal distribution
    proposal, log_prob_fn_proposal = create_sample_model
    
    # check if log_prob is correct
    models.check_log_prob(target, log_prob_fn)
    models.check_log_prob(proposal, log_prob_fn_proposal)

    # run experiment
    res_df = None
    if len(args.load) > 0 :
        try:
            res_df = pd.read_csv(args.load + f"{model}/{model_name}.csv")
            print(f"Loaded pre-saved data for steps = {T}")
        except:
            print(f"Pre-saved data for steps = {T}, delta = {delta} not found. Running from scratch now.")

    if res_df is None:
        # with IMQ
        imq = IMQ(med_heuristic=True)

        res_df = run_bootstrap_experiment(
            nrep, target, log_prob_fn, proposal, imq,
            alpha, num_boot, T, jump_ls, threshold=mode_threshold, nstart_pts=nstart_pts, n=n,
            mcmckernel=MCMCKernel, method=method, random_start_pts=rand_start)

        # save res
        res_df.to_csv(f"{res_root}/{model_name}.csv", index=False)
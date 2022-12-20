from src.ksd.find_modes import find_modes, pairwise_directions
from src.ksd.langevin import RandomWalkMH, RandomWalkBarker
import src.ksd.langevin as mcmc
from src.ksd.ksd import KSD, MPKSD, SPKSD
from src.ksd.kernel import IMQ
from src.ksd.bootstrap import Bootstrap
from src.ksd.find_modes import find_modes, pairwise_directions
from tqdm import tqdm, trange
from src.kgof.ksdagg import ksdagg_wild_test

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import time
import pickle
import argparse

from src.sensors import Sensor, SensorNumpy

import autograd.numpy as anp
import kgof
import kgof.density as kgof_density
import kgof.goftest as kgof_gof

MCMCKernel = RandomWalkMH # RandomWalkBarker 

parser = argparse.ArgumentParser()
parser.add_argument("--RAM_SCALE", type=float)
parser.add_argument("--METHOD", type=str)
args = parser.parse_args()

MODEL = "modified" # "original"
T = 1000
NSAMPLE = 1000
ram_scale = args.RAM_SCALE
# RAM_SCALE_LIST = [0.1, 0.3, 0.5, 0.7, 0.9, 1.08, 1.3]
method = args.METHOD
RAM_SEED = 9
REP = 10
root = f"res/sensors"

if MODEL == "modified":
    # Observation indicators from the fifth sensor (1st column) to the first four sensors
    # and those from the sixth sensor (2nd column) to the first four sensors.
    Ob = tf.constant([1., 0, 1, 0, 1, 0, 1, 0])
    Ob = tf.transpose(tf.reshape(Ob, (2, -1)))

    # Observation indicators among the first four sensors. 
    Os = tf.constant([[0., 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [1, 1, 0, 0]])

    # Each row indicates the location of the known sensors (5th and 6th).
    Xb = tf.constant([0.5, 0.3, 0.3, 0.7])
    Xb = tf.transpose(tf.reshape(Xb, (2, -1)))

    # Each row indicates the location of the unknown sensors (1st, 2nd, 3rd, and 4th).
    Xs = tf.constant([0.5748, 0.0991, 0.2578, 0.8546, 
                0.9069, 0.3651, 0.1350, 0.0392])
    Xs = tf.transpose(tf.reshape(Xs, (2, -1)))

    # The observed distances from the fifth sensor (1st column) to the first four sensors
    # and those from the sixth sensor (2nd column) to the first four sensors.
    Yb = tf.constant([0.6103, 0, 0.2995, 0, 
                0.3631, 0, 0.5656, 0])
    Yb = tf.transpose(tf.reshape(Yb, (2, -1)))

    # Observed distances among the first four sensors.
    Ys = tf.constant([[0, 0, 0, 0.9266],
                [0, 0, 0.2970, 0.8524],
                [0, 0.2970, 0, 0],
                [0.9266, 0.8524, 0, 0]])


    loc_true = tf.constant([
        [0.57, 0.91],
        [0.10, 0.37],
        [0.26, 0.14],
        [0.85, 0.04],
    #     [0.50, 0.30],
    #     [0.30, 0.70]
    ])

    ModelClass = Sensor
    ModelClassNumpy = SensorNumpy
    model_name = "modified_ram"
    path = f"{root}/{model_name}"

tf.random.set_seed(1)
dim = Ys.shape[0] * 2
print("model class:", MODEL, "; dim:", dim)
print("loading data from", path)

target = ModelClass(Ob, Os, Xb, Xs, Yb, Ys)
log_prob_fn = target.log_prob

if method == "all" or method == "fssd":
    target_np = ModelClassNumpy(Ob, Os, Xb, Xs, Yb, Ys)
    log_prob_fn_np = target_np.log_prob
    log_prob_fn_np_den = kgof_density.from_log_den(dim, log_prob_fn_np)


def load_preprocess_sensors(path, n, ntrain):
    ## load result
    mcmc_res = pd.read_csv(path)

    ## thin sample
    ind = tf.range(start=0, limit=400000, delta=400000//n)
    # ind = tf.range(start=0, limit=800000, delta=800000//n)
    samples_off = tf.constant(mcmc_res.loc[ind].to_numpy(), dtype=tf.float32)

    ## split to train and test
    samples_init = samples_off
    samples_init = tf.random.shuffle(samples_init) # shuffle
    sample_train, sample_test = samples_init[:ntrain, ], samples_init[ntrain:, ]
    return sample_train, sample_test


def experiment(T, n, target_dist):
    # jump_ls = tf.linspace(0.8, 1.2, 51)
    jump_ls = tf.linspace(0.8, 1.2, 21)

    ntrain = n // 2
    threshold = 1e-4
    nrep = 1

    num_boot = 800
    alpha = 0.05
    
    # generate multinomial samples for bootstrap
    kernel = IMQ(med_heuristic=True)
    ksd = KSD(target=target_dist, kernel=kernel)
    bootstrap = Bootstrap(ksd, n-ntrain)

    bootstrap_nopert = Bootstrap(ksd, n)

    res = []
    # iterator = tqdm(RAM_SCALE_LIST)
    iterator = trange(REP)
    # res_samples = {}
    # for ram_scale in iterator:
    for i in iterator:
        seed = i
        tf.random.set_seed(seed)
        iterator.set_description(f"seed [{i+1} / {REP}]")

        ## get multinom sample for bootstrap
        multinom_samples = bootstrap.multinom.sample((nrep, num_boot))
        multinom_samples_notrain = bootstrap_nopert.multinom.sample((nrep, num_boot)) # nrep x num_boot x n

        ## load, schuffle, and split data
        sample_train, sample_test = load_preprocess_sensors(f"{path}{ram_scale}/seed{seed+1}.csv", n, ntrain) #! use RAM_SEED
        sample_init = tf.concat([sample_train, sample_test], axis=0)

        ## KSD and pKSD 
        start_pts = tf.random.uniform(
            shape=(ntrain//2, dim), minval=-1.5, maxval=1.5,
        ) # nrep x ntrain//2 x dim

        # 1. KSD
        if method == "all" or method == "ksd":
            iterator.set_description("Running KSD")
            multinom_t = multinom_samples_notrain[0] # num_boost x n

            ksd_rej, ksd_pval = bootstrap.test_once(
                alpha=alpha, num_boot=num_boot, X=sample_init, multinom_samples=multinom_t,
            )

            res.append(["KSD", ram_scale, ksd_rej, ksd_pval, seed])

        # 2. ospKSD
        if method == "all" or method == "ospksd" or method == "test":
            iterator.set_description("Running ospKSD")

            # instantiate ospKSD class
            ospksd = MPKSD(kernel=kernel, pert_kernel=MCMCKernel, log_prob=log_prob_fn)

            # find modes and Hessians
            tic = time.perf_counter()
            ospksd.find_modes(start_pts, threshold=threshold, max_iterations=1000)
            toc = time.perf_counter()
            print(f"Optimisation finished in {toc - tic:0.4f} seconds")
            nmodes = len(ospksd.proposal_dict["modes"])
            print(f"Num. modes found: {nmodes}")
        
            # compute test statistic and p-value
            multinom_t = multinom_samples[0]
            _, ospksd_pval = ospksd.test(
                xtrain=sample_train, 
                xtest=sample_test, 
                T=T,
                jump_ls=jump_ls, 
                num_boot=num_boot,
                multinom_samples=multinom_t,
            )
            ospksd_rej = float(ospksd_pval <= alpha)

            res.append(["ospksd", ram_scale, ospksd_rej, ospksd_pval, seed])

        # 3. spKSD
        if method == "all" or method == "spksd" or method == "test":
            iterator.set_description("Running spKSD")
        
            # instantiate pKSD class
            pksd = SPKSD(kernel=kernel, pert_kernel=MCMCKernel, log_prob=log_prob_fn)
            
            # use previously found modes and Hessians for computational speed
            pksd.ind_pair_list = ospksd.ind_pair_list
            pksd.proposal_dict = ospksd.proposal_dict

            # compute test statistic and p-value 
            multinom_t = multinom_samples_notrain[0] # num_boost x n
            _, pksd_pval = pksd.test(
                x=sample_init, 
                T=T,
                jump_ls=jump_ls, 
                num_boot=num_boot,
                multinom_samples=multinom_t, 
            )
            pksd_rej = float(pksd_pval <= alpha)

            res.append(["spksd", ram_scale, pksd_rej, pksd_pval, seed])

        ## 4. KSDAGG
        if method == "all" or method == "ksdagg":
            iterator.set_description("Running KSDAGG")
            ksdagg_rej = ksdagg_wild_test(
                seed=seed,
                X=sample_init,
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
            res.append(["KSDAGG", ram_scale, ksdagg_rej, None, iter])

        ## 5. FSSD
        if method == "all" or method == "fssd" :
            dat = anp.array(sample_init, dtype=anp.float64)
            dat = kgof.data.Data(dat)
            tr, te = dat.split_tr_te(tr_proportion=0.2, seed=seed)
            opts = {
                'reg': 1e-2, # regularization parameter in the optimization objective
                'max_iter': 200, # maximum number of gradient ascent iterations
                'tol_fun':1e-7, # termination tolerance of the objective
            }

            # J is the number of test locations (or features). Typically not larger than 10.
            J = 1

            # make sure to give tr (NOT te).
            # do the optimization with the options in opts.
            V_opt, gw_opt, opt_info = kgof_gof.GaussFSSD.optimize_auto_init(
                log_prob_fn_np_den, tr, J, **opts)

            fssd_opt = kgof_gof.GaussFSSD(log_prob_fn_np_den, gw_opt, V_opt, alpha)
            test_result = fssd_opt.perform_test(te)
            fssd_pval = test_result["pvalue"]
            fssd_rej = float(fssd_pval <= alpha)
            res.append([method, ram_scale, fssd_rej, fssd_pval, seed])

    # save results
    res_df = pd.DataFrame(res, columns=["method", "ram_scale", "rej", "pval", "seed"])
    res_df.to_csv(f"res/sensors/res_{model_name}_{ram_scale}_{method}.csv", index=False)

if __name__ == "__main__":
    tf.random.set_seed(1)
    tic = time.perf_counter()
    experiment(T, n=NSAMPLE, target_dist=target)
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.4f} seconds")
    

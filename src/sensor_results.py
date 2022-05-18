from src.ksd.find_modes import find_modes, pairwise_directions, run_bfgs
from src.ksd.langevin import RandomWalkMH, RandomWalkBarker
import src.ksd.langevin as mcmc
from src.ksd.ksd import KSD
from src.ksd.kernel import IMQ, l2norm
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

from src.sensors import Sensor, SensorImproper

MCMCKernel = RandomWalkMH # RandomWalkBarker 

MODEL = "modified" # "original"
T = 1000
NSAMPLE = 1000
RAM_SCALE_LIST = [0.1, 0.3, 0.5, 0.7, 0.9, 1.08, 1.3]
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
    model_name = "modified_ram"
    path = f"{root}/{model_name}"

tf.random.set_seed(1)

target = ModelClass(Ob, Os, Xb, Xs, Yb, Ys)
log_prob_fn = target.log_prob

dim = Ys.shape[0] * 2
print("model class:", MODEL, "; dim:", dim)
print("loading data from", path)


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
    jump_ls = tf.linspace(0.8, 1.2, 51)
    
    ntrain = n // 2
    threshold = 1e-4
    nrep = 1

    num_boot = 800
    alpha = 0.05
    
    kernel = IMQ(med_heuristic=True)
    ksd = KSD(target=target_dist, kernel=kernel)
    bootstrap = Bootstrap(ksd, n-ntrain)

    res = []
    res_ksdagg = []
    iterator = tqdm(RAM_SCALE_LIST)
    for ram_scale in iterator:
        res_samples = {}
        for i, seed in enumerate(range(REP)):
            tf.random.set_seed(seed)
            iterator.set_description(f"seed [{i+1} / {REP}]")

            ## get multinom sample for bootstrap
            multinom_samples = bootstrap.multinom.sample((nrep, num_boot))

            ## load, schuffle, and split data
            sample_train, sample_test = load_preprocess_sensors(f"{path}{ram_scale}/seed{RAM_SEED}.csv", n, ntrain)

            # ## sample initial points for finding modes
            # start_pts = tf.concat([
            #     sample_train[:(ntrain//2)], 
            #     tf.random.uniform(shape=(ntrain-ntrain//2, dim), minval=0., maxval=1.)], axis=0) # ntrain x dim
            
            # ## find modes
            # tic = time.perf_counter()
            # mode_list, inv_hess_list = find_modes(start_pts, log_prob_fn, grad_log=None, threshold=threshold, max_iterations=1000)
            # toc = time.perf_counter()
            # print(f"Optimisation finished in {toc - tic:0.4f} seconds")
            # print(f"Num. modes found: {len(mode_list)}")

            # proposal_dict = mcmc.prepare_proposal_input_all(mode_list=mode_list, inv_hess_list=inv_hess_list)
            # _, ind_pair_list = pairwise_directions(mode_list, return_index=True)

            # ## run perturbation kernel
            # print("running in parallel ...")
            # tic = time.perf_counter()
            # mh_jumps = MCMCKernel(log_prob=log_prob_fn)
            # mh_jumps.run(steps=T, std=jump_ls, x_init=sample_train, ind_pair_list=ind_pair_list, **proposal_dict)
            # toc = time.perf_counter()
            # print(f"... done in {toc - tic:0.4f} seconds")

            # ## compute approximate power
            # scaled_ksd_vals = []
            # for j in range(jump_ls.shape[0]):
            #     x_t = mh_jumps.x[j, -1, :, :]
            #     _, ksd_val = ksd.h1_var(X=x_t, Y=tf.identity(x_t), return_scaled_ksd=True)
            #     scaled_ksd_vals.append(ksd_val)
                
            # ## find best jump scale
            # best_jump = jump_ls[tf.math.argmax(scaled_ksd_vals)]

            # ## perturb test sample
            # mh = MCMCKernel(log_prob=log_prob_fn)
            # mh.run(steps=T, std=best_jump, x_init=sample_test, 
            #     ind_pair_list=ind_pair_list, **proposal_dict)
            # x_0 = mh.x[0, :, :]
            # x_t = mh.x[-1, :, :]

            # ## compute p-value
            # kernel = IMQ(med_heuristic=True)
            # ksd = KSD(target=target_dist, kernel=kernel)
            # bootstrap = Bootstrap(ksd, n-ntrain)

            # multinom_one_sample = multinom_samples[0, :]

            # # before perturbation
            # _, p_val0 = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=x_0, multinom_samples=multinom_one_sample)
            # ksd0 = bootstrap.ksd_hat

            # # after perturbation
            # _, p_valt = bootstrap.test_once(alpha=alpha, num_boot=num_boot, X=x_t, multinom_samples=multinom_one_sample)
            # ksdt = bootstrap.ksd_hat

            # res.append([ram_scale, p_val0, p_valt, best_jump.numpy(), ksd0, ksdt, seed])

            # res_samples[seed] = {"perturbed": mh, "sample_train": sample_train, "sample_test": sample_test}

            ## KSDAGG
            sample_init = tf.concat([sample_train, sample_test], axis=0)
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
            res_ksdagg.append([ksdagg_rej, seed])

        # pickle.dump(res_samples,
        #     open(f"res/sensors/sample_{model_name}_{ram_scale}.pkl", "wb"))

    # res_df = pd.DataFrame(res, columns=["ram_scale", "p_val_ksd", "p_val_pksd", "best_jump", "ksd", "pksd", "seed"])
    # res_df.to_csv(f"res/sensors/res_{model_name}.csv", index=False)

    res_df_ksdagg = pd.DataFrame(res, columns=["ksdagg_rej", "seed"])

if __name__ == "__main__":
    tf.random.set_seed(1)
    tic = time.perf_counter()
    experiment(T, n=NSAMPLE, target_dist=target)
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.4f} seconds")
    

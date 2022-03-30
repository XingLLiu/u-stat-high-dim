for ratio_s_var in 0.001 0.01 0.1 1. 
do
  CUDA_VISIBLE_DEVICES="" taskset -c 0-15 python experiments/bootstrap_test_mh_jump_optim.py \
    --model=t-banana --dim=10 --nmodes=10 --nbanana=5 --T=50 --n=1000 --ratio_s_var=$ratio_s_var --method=mcmc_all --nrep=100 &
done

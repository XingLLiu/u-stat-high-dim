# for shift in 0.25 0.5 0.75 1.0 1.25 1.5
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 1-30 python \
#     experiments/bootstrap_test_mh_jump_optim.py --model=rbm --dim=2 --dh=2 --shift=$shift --T=50 --n=1000 --nrep=100 --seed=1 --method=mcmc_all &
# done
# wait
for dh in 5 10 20 30 40
do
  CUDA_VISIBLE_DEVICES="" taskset -c 10-30 python \
    experiments/bootstrap_test_mh_jump_optim.py --model=rbm --dim=50 --dh=$dh --shift=0.5 --T=50 --n=1000 --nrep=100 --seed=1 --method=mcmc_all &
done

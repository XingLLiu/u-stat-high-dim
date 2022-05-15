for ratio_s_var in 0.01 0.05 0.1 0.5 1. 2. 5.
do
  CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python experiments/ksd_all.py \
  --model=t-banana --dim=10 --nmodes=20 --nbanana=10 --T=100 --n=1000 --ratio_s_var=$ratio_s_var --t_std=0.1 --nrep=100 --rand_start=20. &
done

# for dim in 2 5 10 15 20 25 30 35 40 45 50
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-25 python experiments/ksd_all.py \
#   --model=t-banana --dim=$dim --nmodes=20 --nbanana=10 --T=100 --n=1000 --ratio_s_var=5. --t_std=0.1 --nrep=100 --rand_start=20. & #--mcmckernel=barker 
# done

wait

for ratio_s_var in 0.01 0.05 0.1 0.5 1. 2. 5.
do
  CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python experiments/ksd_all.py \
  --model=t-banana --dim=10 --nmodes=10 --nbanana=10 --T=100 --n=1000 --ratio_s_var=$ratio_s_var --t_std=0.1 --nrep=100 --rand_start=20. &
done

wait
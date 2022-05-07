# for t_std in 0.01 0.1 0.5 1. 2.
# do
# #   CUDA_VISIBLE_DEVICES="" taskset -c 1-10 python experiments/ksd_all.py \
# #     --model=t-banana --dim=50 --nmodes=20 --nbanana=10 --T=100 --n=1000 --ratio_s_var=1. --t_std=$t_std --nrep=100 &

#   CUDA_VISIBLE_DEVICES="" taskset -c 15-20 python experiments/ksd_all.py \
#     --model=t-banana --dim=5 --nmodes=5 --nbanana=0 --T=100 --n=1000 --ratio_s_var=1. --t_std=$t_std --nrep=100 &
#   CUDA_VISIBLE_DEVICES="" taskset -c 21-25 python experiments/ksd_all.py \
#     --model=t-banana --dim=5 --nmodes=10 --nbanana=0 --T=100 --n=1000 --ratio_s_var=1. --t_std=$t_std --nrep=100 &
#   CUDA_VISIBLE_DEVICES="" taskset -c 26-30 python experiments/ksd_all.py \
#     --model=t-banana --dim=5 --nmodes=20 --nbanana=10 --T=100 --n=1000 --ratio_s_var=1. --t_std=$t_std --nrep=100 &
# done


for ratio_s_var in 0.01 0.05 0.1 0.5 1. 5. 1.
do
  CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python experiments/ksd_all.py \
  --model=t-banana --dim=10 --nmodes=20 --nbanana=10 --T=100 --n=1000 --ratio_s_var=$ratio_s_var --t_std=0.1 --nrep=100 --mcmckernel=barker --rand_start=20. &
done

# for dim in 2 5 10 15 20 25 30 35 40 45 50
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-25 python experiments/ksd_all.py \
#   --model=t-banana --dim=$dim --nmodes=20 --nbanana=10 --T=100 --n=1000 --ratio_s_var=5. --t_std=0.1 --nrep=100 --mcmckernel=barker --rand_start=20. &
# done

wait

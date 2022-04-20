# for dh in 2 5 8 10 20 30 40
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python \
#     experiments/ksd_all.py --model=rbm --dim=50 --dh=$dh --shift=0.5 --T=50 --n=1000 --nrep=100 --mcmckernel=barker &
# done

# for shift in 1. 1.5 2. #0. 0.01 0.025 0.05 0.075 0.1 0.25 0.5
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python \
#     experiments/ksd_all.py --model=rbm --dim=50 --dh=5 --shift=$shift --T=50 --n=1000 --nrep=100 --mcmckernel=barker &
# done

for n in 200 # 500 1000 1500 2000
do
  CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python \
    experiments/ksd_all.py --model=rbm --dim=50 --dh=5 --shift=0.5 --T=50 --n=$n --nrep=100 --mcmckernel=barker &
  
  CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python \
    experiments/ksd_all.py --model=rbm --dim=50 --dh=5 --shift=0. --T=50 --n=$n --nrep=100 --mcmckernel=barker &
done

wait
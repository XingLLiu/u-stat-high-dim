for dh in 2 5 8 10 20 30 40
do
  CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python \
    experiments/ksd_all.py --model=rbm --dim=50 --dh=$dh --shift=1. --T=50 --n=1000 --nrep=100 --mcmckernel=barker --rand_start=10. &
done

# for shift in 0. 0.01 0.05 0.1 0.5 1.
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 22-33 python \
#     experiments/ksd_all.py --model=rbm --dim=50 --dh=5 --shift=$shift --T=50 --n=1000 --nrep=100 --mcmckernel=barker --rand_start=10. &

#   CUDA_VISIBLE_DEVICES="" taskset -c 22-33 python \
#     experiments/ksd_all.py --model=rbm --dim=50 --dh=40 --shift=$shift --T=50 --n=1000 --nrep=100 --mcmckernel=barker --rand_start=10. &
#   doned

# for n in 200 500 1000 1500 2000
# do
#   # power test
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python \
#     experiments/ksd_all.py --model=rbm --dim=50 --dh=5 --shift=1. --T=50 --n=$n --nrep=100 --mcmckernel=barker --rand_start=10. --suffix=_mix &
  
#   # level test
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python \
#     experiments/ksd_all.py --model=rbm --dim=50 --dh=5 --shift=0. --T=50 --n=$n --nrep=100 --mcmckernel=barker --rand_start=10. --suffix=_mix &
# done

wait
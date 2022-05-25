for dh in 2 5 8 10 20 30 40
do
  CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python \
    experiments.py --model=rbm --dim=50 --dh=$dh --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. &
done


for shift in 0. 0.01 0.05 0.1 0.5 1.
do
  CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python \
    experiments.py --model=rbm --dim=50 --dh=5 --shift=$shift --T=50 --n=1000 --nrep=100 --rand_start=10. &
done
wait

# for n in 200 500 1000 1500 2000
# do
#   # power test
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=1. --T=50 --n=$n --nrep=100 --rand_start=10. & #--mcmckernel=barker
  
#   # level test
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=0. --T=50 --n=$n --nrep=100 --rand_start=10. &
# done

# wait
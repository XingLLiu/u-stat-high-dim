method=fssd

# # latent dim
# for dh in 2 5 8 10 20 30 40
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=$dh --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
# done
# wait

# # perturbation to mixing ratios
# #! some of these did not finish because log_prob and log_prob_np yield different values
# for shift in 0. 0.01 0.05 0.1 0.5 1.
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=$shift --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
# done
# wait

# #! some of these did not finish because log_prob and log_prob_np yield different values
# for n in 200 500 1000 1500 2000
# do
#   # power test
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=1. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
  
#   # level test
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=0. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
# done

# wait


# CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 \
#   experiments.py --model=rbm --dim=50 --dh=5 --shift=.5 --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method
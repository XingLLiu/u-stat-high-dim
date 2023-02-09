## ksd RBF
# CUDA_VISIBLE_DEVICES="" taskset -c 1-10 python3 high_dim/stats.py \
#     --R=1. --STAT=ksd --KERNEL=RBF --EXTRA=quad --N_EXP=10 &

# CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
#     --R=1. --STAT=ksd --KERNEL=RBF --EXTRA=sqrt --N_EXP=10 &

# # ? new result
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#     --R=1. --STAT=ksd --KERNEL=RBF --EXTRA=sqrt_large --N_EXP=10 #&
# # ?

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#     --R=1. --DELTA=2. --STAT=ksd --KERNEL=RBF --N_EXP=30 --EXTRA=cub_ld #&

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#     --R=1. --DELTA=2. --STAT=ksd --KERNEL=RBF --N_EXP=30 --EXTRA=ld #&

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/moments.py \
#     --R=1. --DELTA=2. --STAT=ksd --KERNEL=RBF --EXTRA=3000

# #? new result
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#     --R=1. --STAT=ksd --KERNEL=RBF --EXTRA=quad_ld --N_EXP=10 &

# CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
#     --R=1. --STAT=ksd --KERNEL=RBF --EXTRA=sqrt_ld --N_EXP=10

# varying gamma
DIR_GAMMA=res/high_dim/gamma_ksd
EXTRA=gammaksd

mkdir $DIR_GAMMA
for gam_scale in 0.01 0.05 0.1 0.5 1. 5. 10.
do
    CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 high_dim/moments.py \
        --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=ksd --KERNEL=RBF --EXTRA=$EXTRA --DIR=$DIR_GAMMA --NPOP=10000 #&

    CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
        --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=ksd --KERNEL=RBF --EXTRA=$EXTRA --DIR=$DIR_GAMMA

    mv $DIR_GAMMA/res_analytical_delta2.0_r0.0_ksd_RBF_$EXTRA.csv $DIR_GAMMA/res_analytical_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.csv
    mv $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_ksd_RBF_$EXTRA.p $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.p
done

## MMD RBF
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#     --R=1. --DELTA=2. --STAT=mmd --KERNEL=RBF --N_EXP=30 &

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#     --R=1. --DELTA=2. --STAT=mmd --KERNEL=RBF --N_EXP=30 --EXTRA=ld &

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/moments.py \
#     --R=1. --DELTA=2. --STAT=mmd --KERNEL=RBF 

# # use exact formula!
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/moments.py \
#     --R=1. --DELTA=2. --STAT=mmd --KERNEL=RBF --NPOP=-1 

# # varying gamma
# DIR_GAMMA="res/high_dim/gamma"
# for gam_scale in 0.1 1. 5. 7. 10. 20. 40. 50. # 60. 80. 100.
# do
#     # CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/moments.py \
#     #     --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=mmd --KERNEL=RBF --EXTRA=gamma --DIR=$DIR_GAMMA --NPOP=-1

#     CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#         --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=mmd --KERNEL=RBF --EXTRA=gamma --DIR=$DIR_GAMMA

#     # mv res/high_dim/gamma/res_analytical_delta2.0_r0.0_mmd_RBF_gamma.csv res/high_dim/gamma/res_analytical_delta2.0_r0.0_mmd_RBF_gamma_$gam_scale.csv
#     mv res/high_dim/gamma/stats_res_rep_delta2.0_r0.0_mmd_RBF_gamma.p res/high_dim/gamma/stats_res_rep_delta2.0_r0.0_mmd_RBF_gamma_$gam_scale.p
# done

## distance
# CUDA_VISIBLE_DEVICES="" taskset -c 0-5 python3 high_dim/stats.py \
#     --R=1. --DELTA=2. --STAT=mmd --KERNEL=RBF --N_EXP=30 --EXTRA=sqrt_ld #&

# CUDA_VISIBLE_DEVICES="" taskset -c 6-10 python3 high_dim/stats.py \
#     --R=1. --DELTA=2. --STAT=mmd --KERNEL=RBF --N_EXP=30 --EXTRA=linear_ld #&

# CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
#     --R=1. --DELTA=2. --STAT=mmd --KERNEL=RBF --N_EXP=30 --EXTRA=quad_ld #&

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#     --R=1. --DELTA=2. --STAT=mmd --KERNEL=RBF --N_EXP=30 --EXTRA=cub_ld

## MMD linear kernel
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#     --DELTA=10. --R=1. --STAT=mmd --KERNEL=Linear --N_EXP=30 &

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/moments.py \
#     --DELTA=10. --R=1. --STAT=mmd --KERNEL=Linear --NPOP=10000

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/stats.py \
#     --DELTA=10. --R=1. --STAT=mmd --KERNEL=Linear --N_EXP=30 --EXTRA=ld &

wait
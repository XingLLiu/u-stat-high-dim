## ksd RBF
# CUDA_VISIBLE_DEVICES="" taskset -c 1-10 python3 high_dim/stats.py \
#     --R=1. --STAT=ksd --KERNEL=RBF --EXTRA=quad --N_EXP=10

DIR_DIST=res/high_dim/dist
N_EXP=10

mkdir $DIR_DIST
CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
    --R=1. --STAT=ksd --KERNEL=RBF --EXTRA=quad --N_EXP=$N_EXP --DIR=$DIR_DIST &

CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
    --R=1. --STAT=ksd --KERNEL=RBF --EXTRA= --N_EXP=$N_EXP --DIR=$DIR_DIST &

CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
    --R=1. --STAT=ksd --KERNEL=RBF --EXTRA=sqrt --N_EXP=$N_EXP --DIR=$DIR_DIST &

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 python3 high_dim/moments.py \
#     --R=1. --DELTA=2. --STAT=ksd --KERNEL=RBF --DIR=$DIR_DIST --NPOP=-1

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

# # varying gamma
# DIR_GAMMA=res/high_dim/gamma_ksd_dim27
# EXTRA=gammaksd

# mkdir $DIR_GAMMA
# # for gam_scale in 2. 3. 4. 8. # 0.01 0.05 0.1 0.2 0.3 0.5 1. 5. 10. # dim = 8
# for gam_scale in 3.5 4. 5. 6. 7. 8. 10. 16. 20. # dim = 27
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 high_dim/moments.py \
#         --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=ksd --KERNEL=RBF --EXTRA=$EXTRA --DIR=$DIR_GAMMA --NPOP=10000 #&

#     CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
#         --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=ksd --KERNEL=RBF --EXTRA=$EXTRA --DIR=$DIR_GAMMA

#     mv $DIR_GAMMA/res_analytical_delta2.0_r0.0_ksd_RBF_$EXTRA.csv $DIR_GAMMA/res_analytical_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.csv
#     mv $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_ksd_RBF_$EXTRA.p $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.p
# done

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
# DIR_GAMMA=res/high_dim/gamma_mmd_dim27_n20
# EXTRA=gammammd

# mkdir $DIR_GAMMA
# for gam_scale in 4.6 4.8 #3. 3.2 3.4 3.6 3.8 4. 4.2 4.4 # dim = 27
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 high_dim/moments.py \
#         --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=mmd --KERNEL=RBF --EXTRA=$EXTRA --DIR=$DIR_GAMMA --NPOP=-1

#     CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 high_dim/stats.py \
#         --R=0. --GAM_SCALE=$gam_scale --DELTA=2. --STAT=mmd --KERNEL=RBF --EXTRA=$EXTRA --DIR=$DIR_GAMMA

#     mv $DIR_GAMMA/res_analytical_delta2.0_r0.0_mmd_RBF_$EXTRA.csv $DIR_GAMMA/res_analytical_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.csv
#     mv $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_mmd_RBF_$EXTRA.p $DIR_GAMMA/stats_res_rep_delta2.0_r0.0_RBF_$EXTRA\_$gam_scale.p
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
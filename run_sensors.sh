# CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python

# for jscale in 0.1 0.3 0.5 0.7 0.9 1.3
#   do
#     mkdir res/sensors/modified_ram$jscale
#   done

# for jscale in 0.1 0.3 0.5 0.7 0.9 1.3
#   do
#     CUDA_VISIBLE_DEVICES="" taskset -c 0-20 Rscript
#   done

# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 Rscript src/sensors_location.R 0.1 &
# CUDA_VISIBLE_DEVICES="" taskset -c 31-40 Rscript src/sensors_location.R 0.3 &
# wait
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 Rscript src/sensors_location.R 0.5 &
# CUDA_VISIBLE_DEVICES="" taskset -c 31-40 Rscript src/sensors_location.R 0.7 &
# wait
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 Rscript src/sensors_location.R 0.9 &
# CUDA_VISIBLE_DEVICES="" taskset -c 31-40 Rscript src/sensors_location.R 1.3 &
# wait


# CUDA_VISIBLE_DEVICES="" taskset -c 0-15 python src/sensor_results.py
CUDA_VISIBLE_DEVICES="" taskset -c 16-30 python src/sensor_results_n.py
## run RAM with different scales
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 Rscript src/sensors_location.R 0.1 &
# CUDA_VISIBLE_DEVICES="" taskset -c 31-40 Rscript src/sensors_location.R 0.3 &
# wait
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 Rscript src/sensors_location.R 0.5 &
# CUDA_VISIBLE_DEVICES="" taskset -c 31-40 Rscript src/sensors_location.R 0.7 &
# wait
# CUDA_VISIBLE_DEVICES="" taskset -c 21-30 Rscript src/sensors_location.R 0.9 &
# CUDA_VISIBLE_DEVICES="" taskset -c 31-40 Rscript src/sensors_location.R 1.3 &
# wait

## run tests with KSD, pKSD, KSDAgg
CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python src/sensor_results.py

# CUDA_VISIBLE_DEVICES="" taskset -c 0-1 python src/sensor_results_ramscale.py --RAM_SCALE=0.1 &
# CUDA_VISIBLE_DEVICES="" taskset -c 2-3 python src/sensor_results_ramscale.py --RAM_SCALE=0.3 &
# CUDA_VISIBLE_DEVICES="" taskset -c 4-5 python src/sensor_results_ramscale.py --RAM_SCALE=0.5 &
# # wait
# CUDA_VISIBLE_DEVICES="" taskset -c 6-7 python src/sensor_results_ramscale.py --RAM_SCALE=0.7 &
# CUDA_VISIBLE_DEVICES="" taskset -c 8-9 python src/sensor_results_ramscale.py --RAM_SCALE=0.9 &
# CUDA_VISIBLE_DEVICES="" taskset -c 10-11 python src/sensor_results_ramscale.py --RAM_SCALE=1.08 &
# CUDA_VISIBLE_DEVICES="" taskset -c 12-13 python src/sensor_results_ramscale.py --RAM_SCALE=1.3 &
# wait
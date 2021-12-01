for ratio in 0.5 0.7
    do
        CUDA_VISIBLE_DEVICES=6 python experiments/compare_samplers_convolved.py --ratio=$ratio
        CUDA_VISIBLE_DEVICES=6 python experiments/compare_samplers_convolved_med.py --ratio=$ratio
    done
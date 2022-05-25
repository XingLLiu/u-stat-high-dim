# Code for pKSD
See `example_gaussian_mix.ipynb` for an example of how to use pKSD tests for a given target distribution and a given sample.

```bash
.
├── src                           # Source files for pKSD and benchmarks
├── sh_scripts                    # Shell scripts to run experiments
├── res                           # Folder to store results
├── experiments.py                # Main script for generating results
├── example_gaussian_mix.ipynb    # Demonstration for how to use pKSD tests
└── README.md
```

To reproduce results in the paper, run e.g.,
```bash
# mixture of two gaussians example
sh sh_scripts/run_bimodal.sh
```
results will be stored in `res/bimodal`.

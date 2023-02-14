# Code for A High-dimensional Convergence Theorem for U-statistics with Applications to Kernel-based Testing
This repo contains the code for the experiments in the paper ([link](https://arxiv.org/abs/2302.05686)):

Huang, Kevin H. and Liu, Xing and Duncan, Andrew B. and Gandy, Axel (2023). **A High-dimensional Convergence Theorem for U-statistics with Applications to Kernel-based Testing**. arXiv preprint arXiv: 2302.05686.

## How to install?
Dependencies are listed in `setup.py`. Before running any scripts, run the following to install this repo as a package and download all dependencies:
```bash
pip install -e .
```

## Examples
To reproduce results in the paper, run the code in `run.sh`. Results will be stored in `res/`. Plots can then be made using the jupyter notebooks in `figs_code`. 

E.g., to generate the intro plot Figure 1 in the paper, first run
```bash
# generate results for the intro plot
sh run.sh
```
to generate the results, and run `figs_code/intro.ipynb` to produce the plot.

## Folder structure

```bash
.
├── figs                          # Folder for storing figures
├── figs_code                     # Jupyter notebooks for making plots
├── res                           # Folder for storing results
├── src                           # Source files for KSD, MMD and kernels
├── run.sh                        # Shell script for generating results
├── setup.py                      # Setup file for easy-install
└── README.md
```


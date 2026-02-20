import numpy as np
import scipy.io
import argparse
from turbo import Turbo1
import torch
import os, subprocess


def evaluate_floorplan(x):
    x = np.argsort(x)
    with open("permutation.txt", "w") as f:
        for i in range(len(x)):
            print(x[i].item(), end=',', file=f)
    FNULL = open(os.devnull, 'w')
    subprocess.call(['./floorplan_simulation', 'b1_floorplan.blk'], stdout=FNULL, stderr=subprocess.STDOUT)
    with open("output_floorplan.txt", "r") as f:
        fl = f.readlines()
    output = float(fl[0])
    print(f"result: {output}")
    return output


def bo_loop():
    dim = 30
    def f(x):
        return evaluate_floorplan(x)
    n_init = 20
    n_evals = 200
    lb = np.zeros(dim)
    ub = np.ones(dim)
    for nruns in range(20):
        np.random.seed(nruns)
        print(f'Input dimension {dim}')

        turbo1 = Turbo1(
            f=f,  # Handle to objective function
            lb=lb,  # Numpy array specifying lower bounds
            ub=ub,  # Numpy array specifying upper bounds
            n_init=n_init,  # Number of initial bounds from an Latin hypercube design
            max_evals=n_evals,  # Maximum number of evaluations
            batch_size=1,  # How large batch size TuRBO uses
            verbose=False,  # Print information from each batch
            use_ard=False,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
            n_training_steps=50,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
        )
        turbo1.optimize()
        torch.save({'outputs': torch.Tensor(turbo1.fX), 'train_y': torch.Tensor(turbo1.fX)},
                   'floorplan_botorch_turbo_EI_30_nrun_' + str(nruns) + '.pkl')


if __name__ == '__main__':
    bo_loop()



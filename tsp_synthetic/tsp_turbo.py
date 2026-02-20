import numpy as np
import scipy.io
import argparse
from turbo import Turbo1
import torch


def evaluate_tsp(x, benchmark_index, dim):
    x = np.argsort(x)
    A = np.asarray(scipy.io.loadmat('pcb_dim_'+str(dim) + '_'+str(benchmark_index+1)+'.mat')['A'])
    B = np.asarray(scipy.io.loadmat('pcb_dim_'+str(dim) + '_'+str(benchmark_index+1)+'.mat')['B'])
    E = np.eye(dim)

    permutation = np.array([np.arange(dim), x])

    P = np.zeros([dim, dim]) #initialize the permutation matrix

    for i in range(0,dim):
        P[:, i] = E[:, permutation[1][i]]

    result = (np.trace(P.dot(B).dot(P.T).dot(A.T)))
    print(f"Objective value: {result/10000}")
    return result/10000


def bo_loop(dim, benchmark_index):
    def f(x):
        return evaluate_tsp(x, benchmark_index, dim)
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
                   'tsp_botorch_turbo_EI_dim_' + str(dim) + 'benchmark_index_' + str(
                       benchmark_index) + '_nrun_' + str(nruns) + '.pkl')


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='Bayesian optimization over permutations (QAP)')
    parser_.add_argument('--dim', dest='dim', type=int, default=10)
    parser_.add_argument('--benchmark_index', dest='benchmark_index', type=int, default=0)
    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    bo_loop(kwag_['dim'], kwag_['benchmark_index'])
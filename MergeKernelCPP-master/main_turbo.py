from ttp_wrapper import TTPWrapper
import numpy as np
import pandas as pd
import os
from turbo import Turbo1


def item_dict_to_values(items):
    result = []
    for i in range(len(items)):
        if items['n' + str(i)] == 1:
            result.append(i + 1)
    return np.array(result)


ARGS = [
    ["instances", "a280_n837_uncorr_08.ttp", "2", "10000", "60000"],
    ["instances", "a280_n279_uncorr_08.ttp", "2", "10000", "60000"],
    ["instances", "a280_n837_bounded-strongly-corr_08.ttp", "2", "10000", "60000"]
]
METHODS = [
    'turbo'
]


def main(problem_i=0, max_iter=50, init_iter=5, trials=50):

    args = ARGS[problem_i]
    problem = TTPWrapper(args)
    num_items = problem.items
    num_nodes = problem.nodes
    lb = np.zeros(num_items + num_nodes - 1)
    ub = np.ones(num_items + num_nodes - 1)

    def f(x):
        x_items = x[:num_items]
        x_nodes = x[num_items:]
        item_index = []
        for i in range(num_items):
            if x_items[i] > 0.5:
                item_index.append(i+1)
        nodes = np.argsort(x_nodes) + 1
        nodes = np.insert(nodes, 0, 0)
        nodes = np.insert(nodes, len(nodes), 0)
        return -problem(item_index, nodes)

    problem_name = '_'.join(args)
    save_path = os.path.join('results_collection', problem_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for method in METHODS:

        if method == 'turbo':

            for n in range(trials):

                turbo1 = Turbo1(
                    f=f,  # Handle to objective function
                    lb=lb,  # Numpy array specifying lower bounds
                    ub=ub,  # Numpy array specifying upper bounds
                    n_init=init_iter,  # Number of initial bounds from an Latin hypercube design
                    max_evals=max_iter,  # Maximum number of evaluations
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

                df = pd.DataFrame(turbo1.fX, columns=['value'])
                df.to_csv(os.path.join(save_path, f'{method}_{n}.csv'))


if __name__ == "__main__":
    main(problem_i=2, trials=50, max_iter=50)

import os

from abstract_optimizer import AbstractBayesianOptimization, UtilityFunction
from ttp_wrapper import TTPWrapper
import numpy as np
import pandas as pd
from utils import get_problem_name


def item_dict_to_values(items):
    result = []
    for i in range(len(items)):
        if items['n' + str(i)] == 1:
            result.append(i + 1)
    return np.array(result)


def kernel_experiment(kernel,
                      problem_args=None,
                      max_iter=55,
                      trial_num=50,
                      random_embedding=20):

    if problem_args is None:
        problem_args = ["instances", "a280_n837_uncorr_08.ttp", "2", "10000", "60000"]
    problem = TTPWrapper(args, jar_path='./src')

    problem_name = get_problem_name(problem_args)
    if not os.path.exists(os.path.join('./results_collection', problem_name)):
        os.mkdir(os.path.join('./results_collection', problem_name))

    for n in range(trial_num):

        bo = AbstractBayesianOptimization(
            f=None,
            n_items=problem.items,
            kernel=kernel,
            permutation_length=problem.nodes - 1,
            random_embedding=random_embedding,
            random_state=n
        )

        util = UtilityFunction(value_length=problem.items,
                               kind='ucb',
                               kappa=2.576,
                               xi=0.0,
                               kappa_decay=1,
                               kappa_decay_delay=0)

        max_value = -10000000

        total_results = []
        best = []

        for i in range(max_iter):
            x_probe = bo.suggest(util)

            items = item_dict_to_values(x_probe[0])
            nodes = [i + 1 for i in x_probe[1]]
            nodes = np.array(nodes)
            nodes = np.insert(nodes, 0, 0)
            nodes = np.insert(nodes, len(nodes), 0)
            result = problem(items, nodes)

            if result > max_value:
                print('Found new best: ', result)
                max_value = result
                best.append('Yes')
            else:
                print(result)
                best.append('No')
            print('Items: ', items)
            print('City order: ', nodes)

            bo.register(x_probe[0], x_probe[1], result)
            total_results.append(result)

        df = pd.DataFrame(total_results, columns=['value'])
        df['is_best'] = best
        df.to_csv(f'{os.path.join("./results_collection", problem_name)}\\{bo.name}_result_{n}.csv')


if __name__ == '__main__':
    from dpp_mallows_gpr import DPPMallowsKernel

    # ATTENTION!
    # Jpype package only supports one java JVM run in one python run, it is impossible
    # to change problem in one script.

    # args = ["instances", "a280_n837_uncorr_08.ttp", "2", "10000", "60000"]
    # args = ["instances", "a280_n279_uncorr_08.ttp", "2", "10000", "60000"]
    args = ["instances", "a280_n837_bounded-strongly-corr_08.ttp", "2", "10000", "60000"]

    RANDOM_EMBEDDING = 20

    kernel = DPPMallowsKernel(n=RANDOM_EMBEDDING)

    kernel_experiment(kernel, problem_args=args)

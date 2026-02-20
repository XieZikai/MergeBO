from dpp_mallows_optimizer import DPPBayesianOptimization
from cpp_mallows_optimizer import UtilityFunction
from merge_optimizer import MergeOptimization

from ttp_wrapper import TTPWrapper
import numpy as np
import pandas as pd
import os


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
    'Random',
    'DPPMallows',
    'Merge'
]


def main(problem_i=0, max_iter=50, init_iter=5, trials=50, random_embedding=None):

    args = ARGS[problem_i]
    problem = TTPWrapper(args)

    problem_name = '_'.join(args)
    save_path = os.path.join('results_collection', problem_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for method in METHODS:

        if method == 'Random':

            max_value = -10000000
            total_results = []
            best = []

            for n in range(trials):
                for i in range(max_iter+init_iter):

                    item_list = np.random.rand(problem.items)
                    items = []
                    for index, item in enumerate(item_list):
                        if item > 0.5:
                            items.append(index + 1)

                    nodes = np.random.permutation(problem.nodes - 1)
                    nodes = [i + 1 for i in nodes]

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

                    total_results.append(result)

                df = pd.DataFrame(total_results, columns=['value'])
                df['is_best'] = best
                df.to_csv(os.path.join(save_path, f'{method}_{n}.csv'))

        if method == 'DPPMallows':
            for n in range(trials):
                bo = DPPBayesianOptimization(f=None, n_items=problem.items,
                                             permutation_length=problem.nodes - 1,
                                             random_embedding=random_embedding)

                util = UtilityFunction(value_length=problem.items,
                                       kind='ei',
                                       kappa=2.576,
                                       xi=0.0,
                                       kappa_decay=1,
                                       kappa_decay_delay=0)

                max_value = -10000000

                total_results = []
                best = []

                for i in range(init_iter):

                    item_list = np.random.rand(problem.items)
                    items = []
                    for index, item in enumerate(item_list):
                        if item > 0.5:
                            items.append(index + 1)

                    x_nodes = np.random.permutation(problem.nodes - 1)
                    nodes = [i + 1 for i in x_nodes]

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

                    bo.register(item_list, x_nodes, result)
                    total_results.append(result)

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
                df.to_csv(os.path.join(save_path, f'{method}_{n}.csv'))

        if method == 'Merge':
            for n in range(trials):
                bo = MergeOptimization(
                    f=None,
                    n_items=problem.items,
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

                for i in range(init_iter):

                    item_list = np.random.rand(problem.items)
                    items = []
                    for index, item in enumerate(item_list):
                        if item > 0.5:
                            items.append(index + 1)

                    x_nodes = np.random.permutation(problem.nodes - 1)
                    nodes = [i + 1 for i in x_nodes]

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

                    bo.register(item_list, x_nodes, result)
                    total_results.append(result)

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
                df.to_csv(os.path.join(save_path, f'{method}_{n}.csv'))


if __name__ == "__main__":
    main(problem_i=2, trials=50, max_iter=50)

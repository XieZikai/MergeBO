from merge_optimizer import MergeOptimization
import numpy as np
from cpp_mallows_optimizer import UtilityFunction


class SpearmanOptimization(MergeOptimization):

    def __init__(self, f, n_items, kernel=None, permutation_length=None, pbounds=(0, 1), random_state=None, verbose=2,
                 bounds_transformer=None, random_embedding=None, longer=False):
        self.permutation_length = permutation_length
        super().__init__(f=f,
                         n_items=n_items,
                         kernel=kernel,
                         permutation_length=permutation_length,
                         pbounds=pbounds,
                         random_state=random_state,
                         verbose=verbose,
                         bounds_transformer=bounds_transformer,
                         random_embedding=random_embedding)
        if random_embedding is None:
            self.random_embedding_matrix = np.eye(len(self.bounds))
            self.random_embedding_matrix_permutation = np.eye(self.permutation_length)
        else:
            self.random_embedding_matrix = np.random.randn(random_embedding, len(self.bounds))  # random embedding
            self.random_embedding_matrix_permutation = np.random.randn(random_embedding, self.permutation_length)

    def permutation_embed(self, x_permutation: np.ndarray):
        """
        Featurize the permutation value, then turn the real permutation value into the embedding(compressed) space.
        """
        if len(x_permutation.shape) == 1:
            x_permutation = x_permutation.reshape(1, -1)
            return (x_permutation @ self.random_embedding_matrix_permutation.T)[0]
        else:
            return x_permutation @ self.random_embedding_matrix_permutation.T

    def permutation_restore(self, x_permutation: np.ndarray):
        """
        Undo the embedding as well as featurization.
        """
        if len(x_permutation.shape) == 1:
            x_permutation = x_permutation.reshape(1, -1)
            x_permutation = (x_permutation @ np.linalg.pinv(self.random_embedding_matrix_permutation.T))
        else:
            x_permutation = x_permutation @ np.linalg.pinv(self.random_embedding_matrix_permutation.T)

        return np.argsort(x_permutation[0])


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
    'spearman'
]

def main(problem_i=0, max_iter=50, init_iter=5, trials=50, random_embedding=20):

    args = ARGS[problem_i]
    problem = TTPWrapper(args)

    problem_name = '_'.join(args)
    save_path = os.path.join('results_collection', problem_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for method in METHODS:

        if method == 'spearman':
            for n in range(trials):
                bo = SpearmanOptimization(
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

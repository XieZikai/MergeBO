from abstract_optimizer import AbstractBayesianOptimization
from sklearn.gaussian_process.kernels import RBF, Hyperparameter, _check_length_scale
from utils import (featurize_merge, restore_featurize_merge, merge_sort,
                   featurize_merge_longer, restore_featurize_merge_longer, merge_sort_longer)
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform


class RBFDiffusion(RBF):
    def __init__(self,
                 length_scale=1.0,
                 length_scale_bounds=(1e-5, 1e5),
                 length_scale_permutation=1.0,
                 length_scale_bounds_permutation=(1e-5, 1e5)):
        self.length_scale_bounds_permutation = length_scale_bounds_permutation
        self.length_scale_permutation = length_scale_permutation
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)

    def diag(self, X):
        return np.ones(X[0].shape[0])

    def is_stationary(self):
        return False

    @property
    def anisotropic(self):
        return False

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_length_scale_permutation(self):
        return Hyperparameter("length_scale_permutation", "numeric", self.length_scale_bounds_permutation)

    def __call__(self, X_combine, Y_combine=None, eval_gradient=False):
        assert eval_gradient is False, "Discrete kernel cannot calculate gradient!"
        X, X_permutation = X_combine
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        length_scale_permutation = _check_length_scale(X, self.length_scale_permutation)
        if Y_combine is None:
            dists = pdist(X / length_scale, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
            dists_permutation = pdist(X_permutation / length_scale_permutation, metric='sqeuclidean')
            K_permutation = np.exp(-.5 * dists_permutation)
            # convert from upper-triangular matrix to square matrix
            K_permutation = squareform(K_permutation)
            np.fill_diagonal(K_permutation, 1)
        else:
            Y, Y_permutation = Y_combine
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='sqeuclidean')
            dists_permutation = cdist(X_permutation / length_scale_permutation, Y_permutation / length_scale_permutation,
                                metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            K_permutation = np.exp(-.5 * dists_permutation)

        return K * K_permutation


class MergeOptimization(AbstractBayesianOptimization):

    def __init__(self, f, n_items, kernel=None, permutation_length=None, pbounds=(0, 1), random_state=None, verbose=2,
                 bounds_transformer=None, random_embedding=None, longer=False):

        if kernel is None:
            kernel = RBFDiffusion()
        super().__init__(f=f,
                         n_items=n_items,
                         kernel=kernel,
                         permutation_length=permutation_length,
                         pbounds=pbounds,
                         random_state=random_state,
                         verbose=verbose,
                         bounds_transformer=bounds_transformer,
                         random_embedding=random_embedding)
        self.longer = longer
        if self.longer:
            permutation_length = len(merge_sort_longer(list(range(self.permutation_length))))
        else:
            permutation_length = len(merge_sort(list(range(self.permutation_length))))

        if random_embedding is None:
            self.random_embedding_matrix = np.eye(len(self.bounds))
            self.random_embedding_matrix_permutation = np.eye(permutation_length)
        else:
            self.random_embedding_matrix = np.random.randn(random_embedding, len(self.bounds))  # random embedding
            self.random_embedding_matrix_permutation = np.random.randn(random_embedding, permutation_length)
        self.name = 'MergeOptimization' if not self.longer else 'LongerMergeOptimization'

    def permutation_embed(self, x_permutation: np.ndarray):
        """
        Featurize the permutation value, then turn the real permutation value into the embedding(compressed) space.
        """
        if len(x_permutation.shape) == 1:
            x_permutation = x_permutation.reshape(1, -1)
            if x_permutation.shape[-1] == self.permutation_length:
                if self.longer:
                    x_permutation = featurize_merge_longer(x_permutation)
                else:
                    x_permutation = featurize_merge(x_permutation)
            return (x_permutation @ self.random_embedding_matrix_permutation.T)[0]
        else:
            if x_permutation.shape[-1] == self.permutation_length:
                if self.longer:
                    x_permutation = featurize_merge_longer(x_permutation)
                else:
                    x_permutation = featurize_merge(x_permutation)
            return x_permutation @ self.random_embedding_matrix_permutation.T

    def permutation_restore(self, x_permutation: np.ndarray):
        """
        Undo the embedding as well as featurization.
        """
        if len(x_permutation.shape) == 1:
            x_permutation = x_permutation.reshape(1, -1)
            x_permutation = (x_permutation @ np.linalg.pinv(self.random_embedding_matrix_permutation.T))
            if self.longer:
                x_permutation = restore_featurize_merge_longer(x_permutation, list(range(self.permutation_length)))[0]
            else:
                x_permutation = restore_featurize_merge(x_permutation, list(range(self.permutation_length)))[0]
        else:
            x_permutation = x_permutation @ np.linalg.pinv(self.random_embedding_matrix_permutation.T)
            if self.longer:
                x_permutation = restore_featurize_merge_longer(x_permutation, list(range(self.permutation_length)))
            else:
                x_permutation = restore_featurize_merge(x_permutation, list(range(self.permutation_length)))
        return x_permutation


if __name__ == "__main__":
    import os
    from ttp_wrapper import TTPWrapper
    from abstract_optimizer import UtilityFunction
    from utils import item_dict_to_values
    import pandas as pd

    args = ["instances", "a280_n837_uncorr_08.ttp", "2", "10000", "60000"]
    # args = ["instances", "a280_n279_uncorr_08.ttp", "2", "10000", "60000"]
    # args = ["instances", "a280_n837_bounded-strongly-corr_08.ttp", "2", "10000", "60000"]

    problem_name = '_'.join(args)

    if not os.path.exists(problem_name):
        os.mkdir(problem_name)

    problem = TTPWrapper(args, jar_path='./src')

    trial_num = 50
    random_embedding = 20
    max_iter = 55

    for n in range(trial_num):

        bo = MergeOptimization(
            f=None,
            n_items=problem.items,
            permutation_length=problem.nodes - 1,
            random_embedding=random_embedding,
            random_state=n,
            longer=True
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
        df.to_csv(f'{problem_name}\\{bo.name}_result_{n}.csv')

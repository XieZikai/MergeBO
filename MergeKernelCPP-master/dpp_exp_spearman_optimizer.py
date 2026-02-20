from bayes_opt.target_space import _hashable

import warnings
from itertools import product, permutations
from tqdm import tqdm

import random

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Hyperparameter, _check_length_scale, Kernel

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from cpp_mallows_optimizer import CPPBayesianOptimization

from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.util import UtilityFunction
from bayes_opt.util import acq_max
from bayes_opt.target_space import TargetSpace
from bayes_opt.event import Events
from bayes_opt.constraint import ConstraintModel


# Probably don't need it?
class SpearmanKernel(Kernel):
    def is_stationary(self):
        return True

    def diag(self, X):
        return np.ones(X.shape[0])

    def __init__(self, spearman_length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.spearman_length_scale = spearman_length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return False

    @property
    def hyperparameter_spearman_length_scale(self):
        return Hyperparameter(
            "spearman_length_scale",
            "numeric",
            self.length_scale_bounds,
        )


class SpearmanPermutationBayesianOptimization(BayesianOptimization):
    def __init__(self, f, n_length, random_state=None, verbose=2,
                 bounds_transformer=None):
        self.n_length = n_length
        pbounds = {}
        for i in range(n_length):
            pbounds['x' + str(i)] = (0, n_length - 1)

        super().__init__(f=f, pbounds=pbounds, random_state=random_state, verbose=verbose,
                         bounds_transformer=bounds_transformer)

        def constraint_fun(**kwargs):
            return np.sum(list(kwargs.values()))

        constraint = ConstraintModel(
            fun=constraint_fun,
            lb=0,
            ub=n_length * (n_length-1) / 2
        )

        self._space = TargetSpace(
            f,
            pbounds,
            constraint=constraint,
            random_state=random_state
        )

        self.is_constrained = True

        self._gp = GaussianProcessRegressor(
            kernel=RBF(),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

    def suggest(self, utility_function):

        if len(self._space) == 0:
            return self._space.array_to_params(np.random.permutation(self.n_length))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
        self.constraint._model[0]._check_n_features(self._space.params, reset=True)

        suggestion = acq_max(ac=utility_function.utility,
                             gp=self._gp,
                             constraint=self.constraint,
                             y_max=self._space.target.max(),
                             bounds=self._space.bounds,
                             random_state=self._random_state)

        suggestion_permutation = np.argsort(suggestion)
        suggestion_permutation = np.array(suggestion_permutation, dtype=float)

        # add a strategy to avoid duplicate

        if suggestion_permutation in self._space:
            new_permutation = suggestion_permutation.copy()
            while new_permutation in self._space:
                new_permutation = suggestion_permutation.copy()
                a, b = random.sample(range(self.n_length), 2)
                new_permutation[a], new_permutation[b] = new_permutation[b], new_permutation[a]

            suggestion_permutation = new_permutation

        return self._space.array_to_params(suggestion_permutation)

    def register(self, params, target, constraint_value=None):
        """Expect observation with known target"""
        if constraint_value is None:
            if isinstance(params, dict):
                constraint_value = self.constraint.eval(params)

            elif isinstance(params, np.ndarray):
                if len(params.shape) == 1:
                    constraint_value = np.sum(params)
                else:
                    constraint_value = np.sum(params, axis=1)

        self._space.register(params, target, constraint_value)
        self.dispatch(Events.OPTIMIZATION_STEP)


if __name__ == "__main__":
    def toy_function(x):
        return x[1] / (x[4] + 1) + x[2] * x[3] + np.log(x[5]+1) * np.sqrt(x[0])

    def constraint_value(x):
        return np.sum(x)

    bo = SpearmanPermutationBayesianOptimization(
        f=None,
        n_length=6,
    )

    util = UtilityFunction(kind='ucb')

    for _ in range(5):
        random_point = np.random.permutation(6)
        random_point = np.array(random_point, dtype=float)
        print(random_point)
        result = toy_function(random_point)
        print('result: ', result)
        # bo.register(random_point, result, constraint_value(random_point))
        bo.register(random_point, result)
    print('-----')
    for i in range(50):
        suggestion = bo.suggest(util)
        print(suggestion)
        l = np.array(list(suggestion.values()))
        result = toy_function(l)

        print('result: ', result)
        bo.register(l, result)

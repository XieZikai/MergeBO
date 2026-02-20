from bayes_opt.target_space import _hashable

import warnings
from itertools import product, permutations
from tqdm import tqdm

from dpp_mallows_gpr import DPPMallowsKernel, DPPGPR
from cpp_mallows_optimizer import CPPBayesianOptimization

import numpy as np
from scipy.optimize import minimize
from utils import (featurize_mallows, reverse_featurize_mallows, split_value_permutation,
                   concat_value_permutation, generate_guess_with_constraint)


class DPPBayesianOptimization(CPPBayesianOptimization):
    def __init__(self, f, n_items, permutation_length=None, pbounds=(0, 1), random_state=None, verbose=2,
                 bounds_transformer=None, kernel=None, random_embedding=None):
        """
        :param f:
        :param n_items: length of knapsack items
        :param n: length of permutation vector, or 'cities'
        :param pbounds:
        :param random_state:
        :param verbose:
        :param bounds_transformer:
        :param kernel:
        """
        bounds = {}
        for i in range(n_items):
            bounds['n{}'.format(i)] = pbounds
        super().__init__(f=f, pbounds=bounds, random_state=random_state, verbose=verbose,
                         bounds_transformer=bounds_transformer, random_embedding=random_embedding,
                         permutation_length=permutation_length)
        random_embedding = n_items if random_embedding is None else random_embedding
        if kernel is None:
            kernel = DPPMallowsKernel(n=random_embedding)
        self._gp = DPPGPR(
            n_items=random_embedding,
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state
        )
        self.n = n_items

    def _prime_queue(self, init_points):
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        choice_vectors = []
        permutation_vectors = []
        for _ in range(init_points):
            choice_vector = np.random.choice([0, 1], size=self.n)
            while choice_vector in choice_vectors:
                choice_vector = np.random.choice([0, 1], size=self.n)
            permutation_vector = np.random.permutation(self.permutation_length)
            while permutation_vector in permutation_vectors:
                permutation_vector = np.random.permutation(self.permutation_length)

            choice_vectors.append(choice_vector)
            permutation_vectors.append(permutation_vector)

        for i in range(init_points):
            self._queue.add(choice_vectors[i])
            self._permutation_queue.add(permutation_vectors[i])

    def suggest(self, utility_function, constraints=None):
        if len(self._space) == 0:
            return self._space.array_to_params(np.random.choice([0, 1], size=self.n)), \
                np.random.permutation(self.permutation_length)

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding = self.value_embed(self._space.params)
            self._gp.fit((embedding, self.permutation_embed(np.array(self._permutation_space))), self._space.target)

        utility_function.set_random_embedding_matrix(self.random_embedding_matrix)
        while 1:

            suggestion = self.acq_max_continuous(
                ac=utility_function.utility,
                gp=self._gp,
                y_max=self._space.target.max(),
                bounds=self.bounds,
                random_state=self._random_state,
                constraints=constraints,
            )

            x_value, x_permutation = suggestion
            x_permutation = self.permutation_restore(x_permutation)
            x_value = self._space.array_to_params(x_value)

            for i, key in enumerate(self._space._cache.keys()):
                if _hashable(self._space.params_to_array(x_value)) == key and list(x_permutation) == list(
                        self._permutation_space[i]):
                    continue

            return x_value, x_permutation

    def acq_max_continuous(self, ac, gp, y_max, bounds, random_state, n_warmup=1000, n_iter=25, constraints=None):
        x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(n_warmup, bounds.shape[0]))
        x_permutations = np.array([np.random.permutation(np.arange(self.permutation_length)) for _ in range(n_warmup)])
        x_permutations = self.permutation_embed(x_permutations)
        x_tries = concat_value_permutation(x_tries, x_permutations)

        ys = ac(x_tries, gp=gp, y_max=y_max)
        x_max_value, x_max_permutation = split_value_permutation(x_tries[ys.argmax()], len(self.bounds))
        max_acq = ys.max()

        x_permutations = np.array(
            [self.permutation_embed(np.random.permutation(np.arange(self.permutation_length))) for _ in range(n_iter)])

        if constraints is None:
            x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                           size=(n_iter, bounds.shape[0]))
        else:
            x_seeds = []
            for _ in range(n_iter):
                x_seeds.append(generate_guess_with_constraint(constraints, bounds))
            x_seeds = np.array(x_seeds)

        # Explore the parameter space more throughly
        embed_permutation_bound = self.random_embedding_matrix_permutation @ np.array(
            [[-1, 1] for _ in range(self.random_embedding_matrix_permutation.shape[-1])])
        embed_permutation_bound = np.column_stack((np.minimum(embed_permutation_bound[:, 0], embed_permutation_bound[:, 1]),
                                       np.maximum(embed_permutation_bound[:, 0], embed_permutation_bound[:, 1])))
        combined_bound = np.concatenate([bounds, embed_permutation_bound], axis=0)

        for x_try, x_permute in tqdm(zip(x_seeds, x_permutations), desc='Generating probe point'):
            x_combine = concat_value_permutation(x_try, x_permute)
            if constraints is not None:
                res = minimize(
                    lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                    x_combine,  # x_try.reshape(1, -1),
                    bounds=combined_bound,
                    method="trust-constr",
                    constraints=constraints)

            else:
                res = minimize(
                    lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                    x_combine.reshape(1, -1),
                    bounds=combined_bound,
                    method="L-BFGS-B")
            # See if success
            if not res.success:
                continue

            # See if success
            if not res.success:
                continue

            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun >= max_acq:
                x_max = res.x
                max_acq = -res.fun
                # Clip output to make sure it lies within the bounds. Due to floating
                # point technicalities this is not always the case.
                x_max = np.clip(x_max, combined_bound[:, 0], combined_bound[:, 1])
                x_max = split_value_permutation(x_max, len(self.bounds))
                x_max_value = np.round(x_max[0])
                x_max_permutation = x_max[1]

        return x_max_value, x_max_permutation


def dpp_constraint(item_weight, total_capacity):
    return lambda continuous_plan: total_capacity - np.dot(np.round(continuous_plan), item_weight)


def dpp_acq_max(ac, gp, y_max, p_length, random_numbers=100000):
    """Random optimization"""
    choices = [0, 1]
    # wrong
    choice_vectors = list(product(choices, repeat=gp.n_items))
    permutation_vectors = list(permutations(range(p_length)))
    total_choices = min(len(choice_vectors), len(permutation_vectors), random_numbers)

    chosen_vectors_index = np.random.choice(range(len(choice_vectors)), size=total_choices, replace=False)
    path_vectors_index = np.random.choice(range(len(permutation_vectors)), size=total_choices, replace=False)
    ys = []

    for i in range(total_choices):
        chosen_vector = np.array(choice_vectors[chosen_vectors_index[i]])
        permutation_vector = np.array(permutation_vectors[path_vectors_index[i]])

        ys.append(ac(chosen_vector.reshape(1, -1), permutation=permutation_vector.reshape(1, -1), gp=gp, y_max=y_max)[0])

    choice_max = choice_vectors[chosen_vectors_index[np.argmax(ys)]]
    path_max = permutation_vectors[path_vectors_index[np.argmax(ys)]]

    return choice_max, path_max
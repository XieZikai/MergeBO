import warnings
from tqdm import tqdm
import numpy as np

from abstract_gpr import AbstractGPR

from bayes_opt.bayesian_optimization import BayesianOptimization, Queue, TargetSpace
from bayes_opt.event import Events
from bayes_opt.logger import ScreenLogger
from bayes_opt.target_space import _hashable
from bayes_opt.util import Colours

from scipy.optimize import minimize
from scipy.stats import norm

from utils import (featurize_mallows, reverse_featurize_mallows, split_value_permutation,
                   concat_value_permutation, generate_guess_with_constraint)


def _get_default_logger(verbose):
    return CPPScreenLogger(verbose=verbose)


class CPPScreenLogger(ScreenLogger):
    def __init__(self, verbose=2):
        super().__init__(verbose=verbose)

    def _header(self, instance):
        cells = []
        cells.append(self._format_key("iter"))
        cells.append(self._format_key("target"))
        for key in instance.space.keys:
            cells.append(self._format_key(key))
        for i in range(len(instance.space.keys)):
            cells.append(self._format_key("order_" + str(i)))

        line = "| " + " | ".join(cells) + " |"
        self._header_length = len(line)
        return line + "\n" + ("-" * self._header_length)

    def update(self, event, instance):
        if event == Events.OPTIMIZATION_START:
            line = self._header(instance) + "\n"
        elif event == Events.OPTIMIZATION_STEP:
            is_new_max = self._is_new_max(instance)
            if self._verbose == 1 and not is_new_max:
                line = ""
            else:
                colour = Colours.purple if is_new_max else Colours.black
                line = self._step(instance, colour=colour) + "\n"
        elif event == Events.OPTIMIZATION_END:
            line = "=" * self._header_length + "\n"

        if self._verbose:
            print(line, end="")
        self._update_tracker(event, instance)

    def _step(self, instance, colour=Colours.black):
        res = instance.res[-1]
        cells = []

        cells.append(self._format_number(self._iterations + 1))
        cells.append(self._format_number(res["target"]))

        for key in instance.space.keys:
            cells.append(self._format_number(res["params"][key]))
        for i in instance.permutation_space[-1]:
            cells.append(self._format_number(int(i)))

        return "| " + " | ".join(map(colour, cells)) + " |"


class CPPTargetSpace(TargetSpace):
    def register(self, params, target):
        """Check uniqueness during the loop, no need to check here since the permutation could be different."""
        x = self._as_array(params)

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])


class AbstractBayesianOptimization(BayesianOptimization):
    def __init__(self, f, n_items, kernel, permutation_length=None, pbounds=(0, 1), random_state=None, verbose=2,
                 bounds_transformer=None, random_embedding=None):
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
        self.name = kernel.__class__.__name__
        bounds = {}
        for i in range(n_items):
            bounds['n{}'.format(i)] = pbounds
        super().__init__(f=f, pbounds=bounds, random_state=random_state, verbose=verbose,
                         bounds_transformer=bounds_transformer)

        self.bounds = bounds
        random_embedding = n_items if random_embedding is None else random_embedding

        self.permutation_length = permutation_length if permutation_length is not None else len(self.bounds)

        if random_embedding is None:
            self.random_embedding_matrix = np.eye(len(self.bounds))
            self.random_embedding_matrix_permutation = np.eye(int(self.permutation_length * (self.permutation_length - 1) / 2))
        else:
            self.random_embedding_matrix = np.random.randn(random_embedding, len(self.bounds))  # random embedding
            self.random_embedding_matrix_permutation = (
                np.random.randn(random_embedding, int(self.permutation_length * (self.permutation_length - 1) / 2)))

        self._permutation_space = []
        self._permutation_queue = Queue()

        self._space = CPPTargetSpace(f, self.bounds, random_state=None)

        self._gp = AbstractGPR(
            n_items=random_embedding,
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state
        )
        self.n = n_items
        self.bounds = np.array(
            [item[1] for item in sorted(self.bounds.items(), key=lambda x: x[0])],
            dtype=float)

    def permutation_embed(self, x_permutation: np.ndarray):
        """
        Featurize the permutation value, then turn the real permutation value into the embedding(compressed) space.
        """
        if len(x_permutation.shape) == 1:
            x_permutation = x_permutation.reshape(1, -1)
            if x_permutation.shape[-1] == self.permutation_length:
                x_permutation = featurize_mallows(x_permutation)
            return (x_permutation @ self.random_embedding_matrix_permutation.T)[0]
        else:
            if x_permutation.shape[-1] == self.permutation_length:
                x_permutation = featurize_mallows(x_permutation)
            return x_permutation @ self.random_embedding_matrix_permutation.T

    def permutation_restore(self, x_permutation: np.ndarray):
        """
        Undo the embedding as well as featurization.
        """
        if len(x_permutation.shape) == 1:
            x_permutation = x_permutation.reshape(1, -1)
            x_permutation = (x_permutation @ np.linalg.pinv(self.random_embedding_matrix_permutation.T))[0]
        else:
            x_permutation = x_permutation @ np.linalg.pinv(self.random_embedding_matrix_permutation.T)
        x_permutation = reverse_featurize_mallows(x_permutation)
        return x_permutation

    def value_embed(self, x_value: np.ndarray):
        """
        Turn the real value into the embedding(compressed) space.
        """
        if len(x_value.shape) == 1:
            x_value = x_value.reshape(1, -1)
            return (x_value @ self.random_embedding_matrix.T)[0]
        else:
            return x_value @ self.random_embedding_matrix.T

    def value_restore(self, x_value: np.ndarray):
        """Turn the embedding(compressed) value into the real space. """
        if len(x_value.shape) == 1:
            x_value = x_value.reshape(1, -1)
            x_value = x_value @ np.linalg.pinv(self.random_embedding_matrix.T)
            return x_value[0]
        else:
            return x_value @ np.linalg.pinv(self.random_embedding_matrix.T)

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    @property
    def permutation_space(self):
        return self._permutation_space

    def _prime_queue(self, init_points):
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        permutation_vectors = []
        for _ in range(init_points):
            permutation_vector = np.random.permutation(len(self._space.bounds))
            while permutation_vector in permutation_vectors:
                permutation_vector = np.random.permutation(len(self._space.bounds))
            permutation_vectors.append(permutation_vector)

        for i in range(init_points):
            self._queue.add(self._space.random_sample())
            self._permutation_queue.add(permutation_vectors[i])

    def register(self, params, permutation, target):
        self._permutation_space.append(permutation)
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, permute, lazy=True):
        if lazy:
            self._queue.add(params)
            self._permutation_queue.add(permute)
        else:
            if isinstance(params, dict):
                params = np.array(list(params.values()))
            target = self._space.target_func(([params], [permute]), )[0]
            self._space.register(params, target)
            self._permutation_space.append(permute)
            self.dispatch(Events.OPTIMIZATION_STEP)

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

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 debug=False,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
                x_permute = next(self._permutation_queue)
            except StopIteration:
                util.update_params()
                x_probe, x_permute = self.suggest(util)
                iteration += 1

                if debug and hasattr(self._gp, 'kernel_'):
                    x_probe_embed = self.params_embed(x_probe)
                    x_space_embed = self.params_embed(self._space.params)
                    K = self._gp.kernel_.get_both_kernel_results((x_probe_embed, x_permute), (
                        x_space_embed, np.array(self.permutation_space)))
                    print('Kernels: ')
                    print(K)
                    x = np.array(list(x_probe.values()))
                    x = x.reshape(1, -1)
                    x_permute_ = x_permute.reshape(1, -1)
                    x = (self.params_embed(x), x_permute_)
                    mean, conv = self._gp.predict(x, return_cov=True)
                    print('Mean: ', mean)
                    print('Conv: ', conv)

            self.probe(x_probe, x_permute, lazy=False)

            result = self._space.probe(x_probe)
            self.result_dataframe.append(result)

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def acq_max(self, ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10, constraints=None):
        x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(n_warmup, bounds.shape[0]))
        x_permutations = np.array([np.random.permutation(np.arange(self.permutation_length)) for _ in range(n_warmup)])
        x_permutations = self.permutation_embed(x_permutations)
        x_tries = concat_value_permutation(x_tries, x_permutations)

        ys = ac(x_tries, gp=gp, y_max=y_max)
        x_max = x_tries[ys.argmax()]
        x_max_permutation = x_permutations[ys.argmax()]
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
        embed_permutation_bound = self.random_embedding_matrix_permutation @ np.array([[-1, 1] for _ in range(self.random_embedding_matrix_permutation.shape[-1])])
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

            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun >= max_acq:
                x_max = res.x
                max_acq = -res.fun

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        x_max = np.clip(x_max, combined_bound[:, 0], combined_bound[:, 1])
        x_max = split_value_permutation(x_max, len(self.bounds))
        return x_max[0], x_max[1]

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


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, value_length, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0):

        self.value_length = value_length

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay
        self._random_embedding_matrix = None

        self.xi = xi

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def set_random_embedding_matrix(self, matrix):
        self._random_embedding_matrix = matrix

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        x_value, x_permutation = split_value_permutation(x, self.value_length)

        if self._random_embedding_matrix is None:
            pass
        else:
            if len(x_value.shape) == 1:
                x_value = x_value.reshape(1, -1) @ self._random_embedding_matrix.T
                x_value = x_value[0]
            else:
                x_value = x_value @ self._random_embedding_matrix.T

        if self.kind == 'ucb':
            return self._ucb(x_value, x_permutation, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x_value, x_permutation, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x_value, x_permutation, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, permutation, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict_with_permute(x, permutation, return_std=True)
        if mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        return mean + kappa * std

    @staticmethod
    def _ei(x, permutation, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict_with_permute(x, permutation, return_std=True)
        if mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, permutation, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict_with_permute(x, permutation, return_std=True)
        if mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        z = (mean - y_max - xi) / std
        return norm.cdf(z)

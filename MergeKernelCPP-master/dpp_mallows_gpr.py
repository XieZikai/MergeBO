import numpy as np
from operator import itemgetter
import warnings
import copy

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Hyperparameter, _check_length_scale, Kernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import clone
from sklearn.utils import check_random_state
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import check_array
from sklearn.preprocessing._data import _handle_zeros_in_scale

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import pdist, cdist, squareform
import scipy.optimize
import scipy
from cpp_mallows_gpr import MallowsKernel

GPR_CHOLESKY_LOWER = True


class DiscreteDiffusionKernel(Kernel):
    def __init__(self, n: int,  # vector length
                 length_scale=1.0,
                 length_scale_bounds=(1e-5, 1e5)):
        self.n = n
        self.length_scale_bounds = length_scale_bounds
        if n == 1:
            self.length_scale = length_scale
        else:
            if isinstance(length_scale, float):
                self.length_scale = np.array([length_scale for _ in range(n)])
            else:
                self.length_scale = length_scale

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  self.n)
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        assert eval_gradient is False, "Discrete kernel cannot calculate gradient!"
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            Y = X.copy()
        expanded_X = np.expand_dims(X, axis=1)
        expanded_Y = np.expand_dims(Y, axis=0)
        K_base = np.apply_along_axis(lambda x: (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x)), axis=0,
                                     arr=length_scale)
        K = np.prod(K_base ** (np.absolute(expanded_X - expanded_Y)), axis=-1)
        return K

    def is_stationary(self):
        return False

    def diag(self, X):
        return np.ones(X[0].shape[0])


class DPPMallowsKernel(DiscreteDiffusionKernel, MallowsKernel):
    def __init__(self, n: int,  # vector length for the discrete diffusion kernel
                 length_scale=1.0,
                 length_scale_bounds=(1e-5, 1e5),
                 mallows_length_scale=1.0,
                 **kwargs):
        DiscreteDiffusionKernel.__init__(self, n, length_scale, length_scale_bounds)
        MallowsKernel.__init__(self, mallows_length_scale=mallows_length_scale, length_scale_bounds=length_scale_bounds)
        self.mallows_length_scale = mallows_length_scale

    def __call__(self, X_combine, Y_combine=None, eval_gradient=False):
        assert eval_gradient is False, "Discrete kernel cannot calculate gradient!"
        X, X_permutation = X_combine
        # 没有featurize了 - 传入参数X_permutation就是featurized过后的

        if Y_combine is None:
            Y = X.copy()
            Y_permutation = X_permutation.copy()
        else:
            Y, Y_permutation = Y_combine
        K_mallows = MallowsKernel.__call__(self, X_permutation, Y_permutation)
        K_ddk = DiscreteDiffusionKernel.__call__(self, X, Y)
        return K_mallows * K_ddk

    def get_both_kernel_results(self, X_permute, Y_permute):
        X, X_permutation = X_permute
        if isinstance(X, dict):
            X = np.array([list(X.values())])
            X_permutation = np.array([X_permutation])
        X_permutation = self.featurize(X_permutation)
        X = np.atleast_2d(X)

        Y, Y_permutation = Y_permute
        # if isinstance(Y, dict):
        #     Y = np.array([list(Y.keys())])
        #     Y_permutation = np.array([Y_permutation])
        Y_permutation = self.featurize(Y_permutation)
        # Y = np.atleast_2d(Y)

        length_scale = _check_length_scale(X, self.length_scale)
        mallows_length_scale = _check_length_scale(X_permutation, self.mallows_length_scale)

        dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
        dists_permute = cdist(X_permutation * mallows_length_scale, Y_permutation * mallows_length_scale,
                              metric="sqeuclidean")

        K_rbf = np.exp(-0.5 * dists)
        K_mallows = np.exp(-dists_permute)
        K = K_rbf * K_mallows
        return K, K_rbf, K_mallows


class DiscreteDiffusionGPR(GaussianProcessRegressor):
    def __init__(
            self,
            n,
            kernel=None,
            *,
            alpha=1e-10,
            optimizer="nelder-mead",
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            random_state=None,
    ):
        super().__init__(kernel=kernel,
                         alpha=alpha,
                         optimizer=optimizer,
                         n_restarts_optimizer=n_restarts_optimizer,
                         normalize_y=normalize_y,
                         copy_X_train=copy_X_train,
                         random_state=random_state)
        self.n = n
        # 这个也许要删掉
        if not hasattr(self, '_K_inv'):
            self._K_inv = None

    def fit(self, X, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if self.kernel is None:  # Use an RBF kernel as default
            # self.kernel_ = AlteredConstantKernel(1.0, constant_value_bounds="fixed") * TTPMallowsKernel(
            #     n=self.n, length_scale_bounds="fixed"
            # )
            self.kernel_ = DiscreteDiffusionKernel(n=self.n, length_scale=1.0,
                                            length_scale_bounds=(1e-5, 1e5))
        else:
            self.kernel_ = clone(self.kernel)
        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=ensure_2d,
            dtype=dtype,
        )

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({self.alpha.shape[0]} != {y.shape[0]})"
                )

        self.X_train_ = copy.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            def obj_func(theta):
                return -self.log_marginal_likelihood(theta, clone_kernel=False)

            optima = [
                (
                    self._constrained_optimization(
                        obj_func, self.kernel_.theta, self.kernel_.bounds
                    )
                )
            ]
            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            # self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                           f"The kernel, {self.kernel_}, is not returning a positive "
                           "definite matrix. Try gradually increasing the 'alpha' "
                           "parameter of your GaussianProcessRegressor estimator.",
                       ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        return self

    def predict(self, X, return_std=False, return_cov=False):
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")
        if self.kernel is None or self.kernel.requires_vector_input:
            X = check_array(X, ensure_2d=True, dtype="numeric")
        else:
            X = check_array(X, ensure_2d=False, dtype=None)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                # kernel = (AlteredConstantKernel(1.0, constant_value_bounds="fixed") *
                #           TTPMallowsKernel(self.n, length_scale_bounds="fixed"))
                kernel = DPPMallowsKernel(self.n, length_scale=1.0,
                                          length_scale_bounds=(1e-5, 1e5),
                                          mallows_length_scale=1.0)
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)

            # undo normalisation
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6

                # undo normalisation
                y_cov = y_cov * self._y_train_std ** 2

                return y_mean, y_cov
            elif return_std:
                # cache result of K_inv computation
                self._K_inv = None
                if self._K_inv is None:
                    # compute inverse K_inv of K based on its Cholesky
                    # decomposition L and its inverse L_inv
                    L_inv = solve_triangular(self.L_.T,
                                             np.eye(self.L_.shape[0]))
                    self._K_inv = L_inv.dot(L_inv.T)

                # Compute variance of predictive distribution
                y_var = self.kernel_.diag(X)
                y_var -= np.einsum("ij,ij->i",
                                   np.dot(K_trans, self._K_inv), K_trans)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0

                # undo normalisation
                y_var = y_var * self._y_train_std ** 2

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "nelder-mead":
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="Nelder-Mead",
                jac=False,
                bounds=bounds,
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min


class DPPGPR(GaussianProcessRegressor):
    def __init__(
            self,
            n_items,
            kernel=None,
            *,
            alpha=1e-10,
            optimizer="nelder-mead",
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            random_state=None,
    ):
        super().__init__(kernel=kernel,
                         alpha=alpha,
                         optimizer=optimizer,
                         n_restarts_optimizer=n_restarts_optimizer,
                         normalize_y=normalize_y,
                         copy_X_train=copy_X_train,
                         random_state=random_state)
        self.n_items = n_items
        # 这个也许要删掉
        if not hasattr(self, '_K_inv'):
            self._K_inv = None

    def fit(self, X_combine, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        X, X_permutation = X_combine
        if self.kernel is None:  # Use an RBF kernel as default
            # self.kernel_ = AlteredConstantKernel(1.0, constant_value_bounds="fixed") * TTPMallowsKernel(
            #     n=self.n, length_scale_bounds="fixed"
            # )
            self.kernel_ = DPPMallowsKernel(n=self.n_items, length_scale=1.0,
                                            length_scale_bounds=(1e-5, 1e5),
                                            mallows_length_scale=1.0)
        else:
            self.kernel_ = clone(self.kernel)
        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=ensure_2d,
            dtype=dtype,
        )
        X = (X, X_permutation)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({self.alpha.shape[0]} != {y.shape[0]})"
                )

        self.X_train_ = copy.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            def obj_func(theta):
                return -self.log_marginal_likelihood(theta, clone_kernel=False)

            optima = [
                (
                    self._constrained_optimization(
                        obj_func, self.kernel_.theta, self.kernel_.bounds
                    )
                )
            ]
            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            # self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                           f"The kernel, {self.kernel_}, is not returning a positive "
                           "definite matrix. Try gradually increasing the 'alpha' "
                           "parameter of your GaussianProcessRegressor estimator.",
                       ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        return self

    def predict(self, X_combine, return_std=False, return_cov=False):
        X, X_permutation = X_combine
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")
        if self.kernel is None or self.kernel.requires_vector_input:
            X = check_array(X, ensure_2d=True, dtype="numeric")
        else:
            X = check_array(X, ensure_2d=False, dtype=None)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                # kernel = (AlteredConstantKernel(1.0, constant_value_bounds="fixed") *
                #           TTPMallowsKernel(self.n, length_scale_bounds="fixed"))
                kernel = DPPMallowsKernel(self.n_items, length_scale=1.0,
                                          length_scale_bounds=(1e-5, 1e5),
                                          mallows_length_scale=1.0)
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            X = (X, X_permutation)
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            X = (X, X_permutation)
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)

            # undo normalisation
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6

                # undo normalisation
                y_cov = y_cov * self._y_train_std ** 2

                return y_mean, y_cov
            elif return_std:
                # cache result of K_inv computation
                self._K_inv = None
                if self._K_inv is None:
                    # compute inverse K_inv of K based on its Cholesky
                    # decomposition L and its inverse L_inv
                    L_inv = solve_triangular(self.L_.T,
                                             np.eye(self.L_.shape[0]))
                    self._K_inv = L_inv.dot(L_inv.T)

                # Compute variance of predictive distribution
                y_var = self.kernel_.diag(X)
                y_var -= np.einsum("ij,ij->i",
                                   np.dot(K_trans, self._K_inv), K_trans)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0

                # undo normalisation
                y_var = y_var * self._y_train_std ** 2

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def predict_with_permute(self, X, X_permute, return_std=False, return_cov=False):
        return self.predict((X, X_permute), return_std=return_std, return_cov=return_cov)

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "nelder-mead":
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="Nelder-Mead",
                jac=False,
                bounds=bounds,
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min


if __name__ == '__main__':
    # Generating training data
    N = 4  # Total item num
    M = 3  # Total city num
    TRAIN_NUM = 100
    train_picking = np.random.choice([0, 1], size=(TRAIN_NUM, N))
    train_path = np.array([np.random.permutation(M) for _ in range(TRAIN_NUM)])
    train_y = np.random.rand(TRAIN_NUM) * 10
    train = (train_picking, train_path)

    TEST_NUM = 20
    test_picking = np.random.choice([0, 1], size=(TEST_NUM, N))
    test_path = np.array([np.random.permutation(M) for _ in range(TEST_NUM)])
    test_y = np.random.rand(TEST_NUM) * 10
    test = (test_picking, test_path)

    gpr = DPPGPR(n_items=N)
    gpr.fit(train, train_y)

    print(gpr.predict(test, return_cov=True))

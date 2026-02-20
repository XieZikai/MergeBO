import numpy as np
from scipy.optimize import minimize

from operator import itemgetter
import warnings
import copy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import clone
from sklearn.utils import check_random_state
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import check_array
from sklearn.preprocessing._data import _handle_zeros_in_scale

import scipy
from scipy.linalg import cholesky, cho_solve, solve_triangular

GPR_CHOLESKY_LOWER = True


class AbstractGPR(GaussianProcessRegressor):

    def __init__(
            self,
            n_items,
            kernel,
            *,
            alpha=1e-10,
            optimizer="nelder-mead",
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            random_state=None,
    ):
        self.kernel_name = kernel.__class__.__name__
        super().__init__(kernel=kernel,
                         alpha=alpha,
                         optimizer=optimizer,
                         n_restarts_optimizer=n_restarts_optimizer,
                         normalize_y=normalize_y,
                         copy_X_train=copy_X_train,
                         random_state=random_state)
        self.n_items = n_items

        if not hasattr(self, '_K_inv'):
            self._K_inv = None

    def fit(self, X_combine, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        X, X_permutation = X_combine
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
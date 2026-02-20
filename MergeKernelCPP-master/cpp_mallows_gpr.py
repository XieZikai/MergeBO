import numpy as np
from operator import itemgetter
import warnings
import copy
from utils import _num_samples

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
GPR_CHOLESKY_LOWER = True


class MallowsKernel(Kernel):
    def is_stationary(self):
        return False

    def diag(self, X):
        return np.ones(X.shape[0])

    def __init__(self, mallows_length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.mallows_length_scale = mallows_length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return False

    @property
    def hyperparameter_mallows_length_scale(self):
        return Hyperparameter(
            "mallows_length_scale",
            "numeric",
            self.length_scale_bounds,
        )

    def __call__(self, x, y=None, eval_gradient=False):

        mallows_length_scale = _check_length_scale(x, self.mallows_length_scale)

        if y is None:
            dists_permute = pdist(x * mallows_length_scale, metric="sqeuclidean")
            K_mallows = np.exp(-dists_permute)
            K_mallows = squareform(K_mallows)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists_permute = cdist(x * mallows_length_scale, y * mallows_length_scale, metric="sqeuclidean")
            K_mallows = np.exp(-dists_permute)

        if eval_gradient:
            if self.hyperparameter_mallows_length_scale.fixed:
                return K_mallows, np.empty((x.shape[0], x.shape[0], 0))
            else:
                K_gradient_mallows = -(K_mallows * squareform(dists_permute))[:, :, np.newaxis]
                return K_mallows, K_gradient_mallows

        # NO NEED
        if y is None:
            y = x.copy()
        if len(x.shape) > 2:
            kernel_mat = np.sum((x - y) ** 2, axis=-1)
        else:
            kernel_mat = np.sum((x[:, None, :] - y) ** 2, axis=-1)
        return np.exp(- kernel_mat * self.mallows_length_scale ** 2)

    def mallows_kernel(self, x1, x2):
        """
        Calculate the mallows kernel result.
        """
        if len(x1.shape) > 2:
            kernel_mat = np.sum((x1 - x2) ** 2, axis=-1)
        else:
            kernel_mat = np.sum((x1[:, None, :] - x2) ** 2, axis=-1)
        return np.exp(- kernel_mat * self.mallows_length_scale ** 2)


class CPPMallowsKernel(RBF, MallowsKernel):

    def __init__(self, mallows_length_scale=1.0, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), random_embedding=None):
        RBF.__init__(self, length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        MallowsKernel.__init__(self, mallows_length_scale=mallows_length_scale, length_scale_bounds=length_scale_bounds)
        self.random_embedding = random_embedding
        self.random_embedding_matrix = None

    def random_embedding_matrix_init(self, p_length):
        if self.random_embedding is None:
            self.random_embedding_matrix = np.eye(p_length)
        elif self.random_embedding_matrix is None:
            self.random_embedding_matrix = np.random.rand(self.random_embedding, p_length)  # random embedding

    def __call__(self, X_combine, Y_combine=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient of the RBF part.
        Gradient only counts value array part.

        Parameters
        ----------
        X_combine : Two ndarray: value array and permutation array
            Value array: ndarray of shape (n_samples_X, n_features)
            Permutation array:
            Left argument of the returned kernel k(X, Y)

        Y_combine : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X, X_permutation = X_combine
        # 没有featurize了 - 传入参数X_permutation就是featurized过后的
        # self.random_embedding_matrix_init(X_permutation.shape[-1])

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        mallows_length_scale = _check_length_scale(X_permutation, self.mallows_length_scale)

        if Y_combine is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            dists_permute = pdist(X_permutation * mallows_length_scale, metric="sqeuclidean")
            K_rbf = np.exp(-0.5 * dists)
            K_mallows = np.exp(-dists_permute)
            # convert from upper-triangular matrix to square matrix
            K_rbf = squareform(K_rbf)
            K_mallows = squareform(K_mallows)
            K = K_rbf * K_mallows
            np.fill_diagonal(K, 1)
        else:
            Y, Y_permutation = Y_combine
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            dists_permute = cdist(X_permutation * mallows_length_scale, Y_permutation * mallows_length_scale, metric="sqeuclidean")

            K_rbf = np.exp(-0.5 * dists)
            K_mallows = np.exp(-dists_permute)
            K = K_rbf * K_mallows

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed or self.hyperparameter_mallows_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                # K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                # return K, K_gradient

                K_gradient_rbf = (K_rbf * squareform(dists))[:, :, np.newaxis] * K_mallows[:, :, np.newaxis]
                K_gradient_mallows = -(K_mallows * squareform(dists_permute))[:, :, np.newaxis] * K_rbf[:, :, np.newaxis]
                return K, np.concatenate([K_gradient_rbf, K_gradient_mallows], axis=-1)

            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                # K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                #         length_scale ** 2
                # )
                # K_gradient *= K[..., np.newaxis]
                # return K, K_gradient
                K_gradient_rbf = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                         length_scale ** 2
                )
                K_gradient_rbf *= K_rbf[..., np.newaxis]
                K_gradient_rbf *= K_mallows[..., np.newaxis]

                K_gradient_mallows = -(K_mallows * squareform(dists_permute))[:, :, np.newaxis] * K_rbf[:, :,
                                                                                                  np.newaxis]

                '''
                K_gradient_mallows = (X_permutation[:, np.newaxis, :] - X_permutation[np.newaxis, :, :]) ** 2 * (
                         -2 * length_scale ** 2
                )
                K_gradient_mallows *= K_mallows[..., np.newaxis]
                K_gradient_mallows *= K_rbf[..., np.newaxis]
                '''

                return K, np.concatenate([K_gradient_rbf, K_gradient_mallows], axis=-1)

        else:
            return K

    def diag(self, X):
        return np.ones(X[0].shape[0])

    def get_both_kernel_results(self, X_permute, Y_permute):
        X, X_permutation = X_permute
        if isinstance(X, dict):
            X = np.array([list(X.values())])
            X_permutation = np.array([X_permutation])
        X = np.atleast_2d(X)

        Y, Y_permutation = Y_permute

        length_scale = _check_length_scale(X, self.length_scale)
        mallows_length_scale = _check_length_scale(X_permutation, self.mallows_length_scale)

        dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
        dists_permute = cdist(X_permutation * mallows_length_scale, Y_permutation * mallows_length_scale,
                              metric="sqeuclidean")

        K_rbf = np.exp(-0.5 * dists)
        K_mallows = np.exp(-dists_permute)
        K = K_rbf * K_mallows
        return K, K_rbf, K_mallows


class CPPGPR(GaussianProcessRegressor):
    def __init__(
        self,
        kernel=None,
        *,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
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
        # 这个也许要删掉
        if not hasattr(self, '_K_inv'):
            self._K_inv = None

    def fit(self, X_combine, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        X, X_permutation = X_combine
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = AlteredConstantKernel(1.0, constant_value_bounds="fixed") * CPPMallowsKernel(
                1.0, length_scale_bounds="fixed"
            )
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
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False
                    )
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)

            # First optimize starting from theta specified in kernel
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
                kernel = (AlteredConstantKernel(1.0, constant_value_bounds="fixed") *
                          RBF(1.0, length_scale_bounds="fixed"))
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
                y_cov = y_cov * self._y_train_std**2

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
                y_var = y_var * self._y_train_std**2

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min

    def log_marginal_likelihood(
            self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

        # Alg 2.1, page 19, line 7
        # -0.5 . y^T . alpha - sum(log(diag(L))) - n_samples / 2 log(2*pi)
        # y is originally thought to be a (1, n_samples) row vector. However,
        # in multioutputs, y is of shape (n_samples, 2) and we need to compute
        # y^T . alpha for each output, independently using einsum. Thus, it
        # is equivalent to:
        # for output_idx in range(n_outputs):
        #     log_likelihood_dims[output_idx] = (
        #         y_train[:, [output_idx]] @ alpha[:, [output_idx]]
        #     )
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        # the log likehood is sum-up across the outputs
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        if eval_gradient:
            # Eq. 5.9, p. 114, and footnote 5 in p. 114
            # 0.5 * trace((alpha . alpha^T - K^-1) . K_gradient)
            # alpha is supposed to be a vector of (n_samples,) elements. With
            # multioutputs, alpha is a matrix of size (n_samples, n_outputs).
            # Therefore, we want to construct a matrix of
            # (n_samples, n_samples, n_outputs) equivalent to
            # for output_idx in range(n_outputs):
            #     output_alpha = alpha[:, [output_idx]]
            #     inner_term[..., output_idx] = output_alpha @ output_alpha.T
            inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
            # compute K^-1 of shape (n_samples, n_samples)
            K_inv = cho_solve(
                (L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False
            )
            # create a new axis to use broadcasting between inner_term and
            # K_inv
            inner_term -= K_inv[..., np.newaxis]
            # Since we are interested about the trace of
            # inner_term @ K_gradient, we don't explicitly compute the
            # matrix-by-matrix operation and instead use an einsum. Therefore
            # it is equivalent to:
            # for param_idx in range(n_kernel_params):
            #     for output_idx in range(n_output):
            #         log_likehood_gradient_dims[param_idx, output_idx] = (
            #             inner_term[..., output_idx] @
            #             K_gradient[..., param_idx]
            #         )
            log_likelihood_gradient_dims = 0.5 * np.einsum(
                "ijl,jik->kl", inner_term, K_gradient
            )
            # the log likehood gradient is the sum-up across the outputs
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood


class AlteredConstantKernel(C):
    def __call__(self, X_permute, Y=None, eval_gradient=False):
        X_value, X = X_permute
        if Y is None:
            Y = X_permute
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        Y_value, Y_permutation = Y
        K = np.full((_num_samples(X), _num_samples(Y_permutation)), self.constant_value,
                    dtype=np.array(self.constant_value).dtype)
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (K, np.full((_num_samples(X), _num_samples(X), 1),
                                   self.constant_value,
                                   dtype=np.array(self.constant_value).dtype))
            else:
                return K, np.empty((_num_samples(X), _num_samples(X), 0))
        else:
            return K

    def diag(self, X):
        return np.full(_num_samples(X), self.constant_value,
                       dtype=np.array(self.constant_value).dtype)


if __name__ == '__main__':

    def featurize_np(x):
        featurized_x = []
        for nums in range(x.shape[0]):
            vec = []
            for i in range(x.shape[1]):
                for j in range(i + 1, x.shape[1]):
                    vec.append(1 if x[nums][i] > x[nums][j] else -1)
            featurized_x.append(vec)
        normalizer = np.sqrt(x.shape[1] * (x.shape[1] - 1) / 2)
        return featurized_x / normalizer

    k = CPPMallowsKernel()
    gpr = CPPGPR()

    x_train_value = np.random.rand(10, 5) * 100
    x_train_order = np.array([[3, 1, 2, 4, 5], [1, 2, 3, 4, 5]]).repeat(5, axis=0)
    x_train_order = featurize_np(x_train_order)
    a = (x_train_value, x_train_order)

    y = np.random.rand(10, 1) * 10
    gpr.fit(a, y)
    print(gpr.predict(a, return_cov=True))

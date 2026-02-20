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
from scipy.special import comb
import scipy.optimize
import scipy
from cpp_mallows_gpr import MallowsKernel
from dpp_mallows_gpr import DiscreteDiffusionKernel

GPR_CHOLESKY_LOWER = True


class KendallKernel(Kernel):
    def is_stationary(self):
        return False

    def diag(self, X):
        return np.ones(X.shape[0])

    def __init__(self):
        pass

    @property
    def anisotropic(self):
        return False

    def __call__(self, x, y=None, eval_gradient=False):
        d = int(np.sqrt(x.shape[-1] * 2))
        if eval_gradient:
            raise ValueError("Discrete kernel cannot calculate gradient")

        if y is None:
            dists_permute = pdist(x, metric="sqeuclidean")
            n_d = squareform(dists_permute)
            k_kendall = 1 - 2 * n_d / comb(d, 2)
        else:
            n_d = cdist(x, y, metric="sqeuclidean")
            k_kendall = 1 - 2 * n_d / comb(d, 2)

        return k_kendall


class CPPKendallKernel(RBF, KendallKernel):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), random_embedding=None):
        RBF.__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.random_embedding = random_embedding
        self.random_embedding_matrix = None

    def random_embedding_matrix_init(self, p_length):
        if self.random_embedding is None:
            self.random_embedding_matrix = np.eye(p_length)
        elif self.random_embedding_matrix is None:
            self.random_embedding_matrix = np.random.rand(self.random_embedding, p_length)  # random embedding

    def __call__(self, X_combine, Y_combine=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError("Discrete kernel cannot calculate gradient")
        X, X_permutation = X_combine
        d = int(np.sqrt(X_permutation.shape[-1] * 2))

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)

        if Y_combine is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            dists_permute = pdist(X_permutation, metric="sqeuclidean")
            K_rbf = np.exp(-0.5 * dists)
            K_kendall = 1 - 2 * dists_permute / comb(d, 2)
            K_rbf = squareform(K_rbf)
            K_kendall = squareform(K_kendall)
            K = K_rbf * K_kendall
            np.fill_diagonal(K, 1)

        else:
            Y, Y_permutation = Y_combine
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            dists_permute = cdist(X_permutation, Y_permutation, metric="sqeuclidean")

            K_rbf = np.exp(-0.5 * dists)
            K_kendall = 1 - 2 * dists_permute / comb(d, 2)
            K = K_rbf * K_kendall

        return K

    def diag(self, X):
        return np.ones(X[0].shape[0])


class DPPKendallKernel(DiscreteDiffusionKernel, KendallKernel):
    def __init__(self, n: int,  # vector length for the discrete diffusion kernel
                 length_scale=1.0,
                 length_scale_bounds=(1e-5, 1e5)):
        DiscreteDiffusionKernel.__init__(self, n, length_scale, length_scale_bounds)

    def __call__(self, X_combine, Y_combine=None, eval_gradient=False):
        assert eval_gradient is False, "Discrete kernel cannot calculate gradient!"
        X, X_permutation = X_combine
        
        if Y_combine is None:
            Y = X.copy()
            Y_permutation = X_permutation.copy()
        else:
            Y, Y_permutation = Y_combine

        K_kendall = KendallKernel.__call__(self, X_permutation, Y_permutation)
        K_ddk = DiscreteDiffusionKernel.__call__(self, X, Y)
        return K_kendall * K_ddk
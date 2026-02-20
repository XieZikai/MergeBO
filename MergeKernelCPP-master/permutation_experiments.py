import torch
import gpytorch
import numpy as np

import scipy.io

from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import module_to_array
from botorch.acquisition import ExpectedImprovement
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel, RBFKernel
import matlab.engine

from dpp_exp_spearman_optimizer import SpearmanPermutationBayesianOptimization
from bayes_opt.util import UtilityFunction

class MallowsKernel(Kernel):
    has_lengthscale = True
    def forward(self, X, X2, **params):
        if len(X.shape) > 2:
            kernel_mat = torch.sum((X - X2)**2, axis=-1)
        else:
            kernel_mat = torch.sum((X[:, None, :] - X2)**2, axis=-1)
        return torch.exp(-self.lengthscale * kernel_mat)


def featurize(x):
    featurized_x = []
    for nums in range(x.size(0)):
        vec = []
        for i in range(x.size(1)):
            for j in range(i+1, x.size(1)):
                vec.append(1 if x[nums][i] > x[nums][j] else -1)
        featurized_x.append(vec)
    normalizer = np.sqrt(x.size(1)*(x.size(1) - 1)/2)
    return torch.tensor(featurized_x/normalizer).double()


def evaluate_qap(x, benchmark_index, dim):
    if x.dim() == 2:
        x = x.squeeze(0)
    x = x.numpy()
    A = np.asarray(scipy.io.loadmat('./permutation_experiments/qap_synthetic/QAP_LIB_A'+str(benchmark_index+1)+'.mat')['A'])
    B = np.asarray(scipy.io.loadmat('./permutation_experiments/qap_synthetic/QAP_LIB_'+str(benchmark_index+1)+'.mat')['B'])
    E = np.eye(dim)
    permutation = np.array([np.arange(dim), x])

    P = np.zeros([dim, dim]) #initialize the permutation matrix

    for i in range(dim):
        P[:, i] = E[:, permutation[1][i]]
    result = (np.trace(P.dot(B).dot(P.T).dot(A.T)))
    print(f'QAP objective value: {result}')
    return result


def initialize_model(train_x, train_obj, covar_module=None, state_dict=None):
    # define models for objective and constraint
    if covar_module is not None:
        model = SingleTaskGP(train_x, train_obj, covar_module=covar_module)
    else:
        model = SingleTaskGP(train_x, train_obj)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()#noise_constraint=gpytorch.constraints.Positive())
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def EI_local_search(AF, x):
    best_val = AF(featurize(x.unsqueeze(0)).unsqueeze(1).detach()).detach().numpy()
    best_point = x.numpy()
    for num_steps in range(100):
        # print(f"best AF value : {best_val} at best_point = {best_point}")
        all_vals = []
        all_points = []
        for i in range(len(best_point)):
            for j in range(i+1, len(best_point)):
                x_new = best_point.copy()
                x_new[i], x_new[j] = x_new[j], x_new[i]
                all_vals.append(AF(featurize(torch.from_numpy(x_new).unsqueeze(0)).unsqueeze(1)).detach().numpy())
                all_points.append(x_new)
        idx = np.argmax(all_vals)
        if all_vals[idx] > best_val:
            best_point = all_points[idx]
            best_val = all_vals[idx]
        else:
              break
    print(f"best AF value : {best_val.item()} at best_point = {best_point}")
    return torch.from_numpy(best_point), best_val


def mallows_bo_loop_qap(benchmark_index=3, kernel_type='mallows'):
    n_init = 20
    n_evals = 200
    for nruns in range(20):
        torch.manual_seed(nruns)
        np.random.seed(nruns)
        dim = np.asarray(scipy.io.loadmat('./permutation_experiments/qap_synthetic/QAP_LIB_A'+str(benchmark_index+1)+'.mat')['A']).shape[0]
        print(f'Input dimension {dim}')
        train_x = torch.from_numpy(np.array([np.random.permutation(np.arange(dim)) for _ in range(n_init)]))
        outputs = []
        for i in range(n_init):
            outputs.append(evaluate_qap(train_x[i], benchmark_index, dim))
        train_y = -1*torch.tensor(outputs)
        for num_iters in range(n_init, n_evals):
            inputs = featurize(train_x)
            if kernel_type == 'mallows':
                covar_module = MallowsKernel()
            elif kernel_type == 'rbf':
                covar_module = RBFKernel()
            train_y = (train_y - torch.mean(train_y))/(torch.std(train_y))
            inputs = inputs.double()
            train_y = train_y.double()
            mll_bt, model_bt = initialize_model(inputs, train_y.unsqueeze(1), covar_module)
            model_bt.likelihood.noise_covar.noise = torch.tensor(0.0001).double()
            mll_bt.model.likelihood.noise_covar.raw_noise.requires_grad = False
            fit_gpytorch_model(mll_bt)
            # print(train_y.dtype)
            print(f'\n -- NLL: {mll_bt(model_bt(inputs), train_y.double())}')
            EI = ExpectedImprovement(model_bt, best_f = train_y.max().item())
            # Multiple random restarts
            best_point, ls_val = EI_local_search(EI, torch.from_numpy(np.random.permutation(np.arange(dim))))
            for _ in range(10):
                new_point, new_val = EI_local_search(EI, torch.from_numpy(np.random.permutation(np.arange(dim))))
                if new_val > ls_val:
                    best_point = new_point
                    ls_val = new_val
            print(f"Best Local search value: {ls_val}")
            if not torch.all(best_point.unsqueeze(0) == train_x, axis=1).any():
                best_next_input = best_point.unsqueeze(0)
            else:
                print(f"Generating randomly !!!!!!!!!!!")
                best_next_input = torch.from_numpy(np.random.permutation(np.arange(dim))).unsqueeze(0)
            # print(best_next_input)
            next_val = evaluate_qap(best_next_input, benchmark_index, dim)
            train_x = torch.cat([train_x, best_next_input])
            outputs.append(next_val)
            train_y = -1*torch.tensor(outputs)
            # train_y = torch.cat([train_y, torch.tensor([next_val])])
            print(f"\n\n Iteration {num_iters} with value: {outputs[-1]}")
            print(f"Best value found till now: {np.min(outputs)}")
            torch.save({'inputs_selected':train_x, 'outputs':outputs, 'train_y':train_y}, './results_collection/permutation/qap_botorch_'+kernel_type+'_EI_benchmark_index_'+str(benchmark_index)+'_nrun_'+str(nruns)+'.pkl')


def fastmvg(Phi, alpha, D):
    # fastmvg sampler (code from BOCS) https://github.com/baptistar/BOCS
    # Fast sampler for multivariate Gaussian distributions (large p, p > n) of
    #  the form N(mu, S), where
    #       mu = S Phi' y
    #       S  = inv(Phi'Phi + inv(D))
    # Reference:
    #   Fast sampling with Gaussian scale-mixture priors in high-dimensional
    #   regression, A. Bhattacharya, A. Chakraborty and B. K. Mallick
    #   arXiv:1506.04778
    n, p = Phi.shape

    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi,u) + delta
    mult_vector = np.vectorize(np.multiply)
    Dpt = mult_vector(Phi.T, d[:,np.newaxis])
    w = np.linalg.solve(np.matmul(Phi,Dpt) + np.eye(n), alpha - v)
    x = u + np.dot(Dpt,w)

    return x


def kendall_bo_loop_qap():
    n_init = 20
    n_evals = 200
    benchmark_index = 3 # int(sys.argv[1]) # set to 3
    for nruns in range(20):
        torch.manual_seed(nruns)
        np.random.seed(nruns)
        dim = np.asarray(scipy.io.loadmat('./permutation_experiments/qap_synthetic/QAP_LIB_A'+str(benchmark_index+1)+'.mat')['A']).shape[0]
        train_x = torch.from_numpy(np.array([np.random.permutation(np.arange(dim)) for _ in range(n_init)]))
        outputs = []
        for i in range(n_init):
            outputs.append(evaluate_qap(train_x[i], benchmark_index, dim))
        train_y = torch.tensor(outputs)
        for num_iters in range(n_init, n_evals):
            X = featurize(train_x).numpy()
            for i in range(X.shape[1]):
               X[:, i] = (X[:, i] - np.mean(X[:, i]))/(np.std(X[:, i]))
            y = train_y.numpy()
            y = (y-np.mean(y))/np.std(y)
            theta = fastmvg(X, y, np.eye(X.shape[1]))
            theta_ls_matrix = np.zeros((dim, dim))
            theta_ls_matrix[np.triu_indices(dim, k=1)] = theta

            # perm = np.array(range(1, dim + 1))
            # onevec = np.ones(dim)
            # PO = np.outer(perm, onevec)
            # tls_matrix = np.power(PO - PO.T, 2)

            tls_matrix = np.zeros((dim, dim))
            tls_matrix[np.triu_indices(dim, k=1)] = 1/np.sqrt(dim*(dim - 1)/2)
            tls_matrix.T[np.triu_indices(dim, k=1)] = -1/np.sqrt(dim*(dim - 1)/2)

            scipy.io.savemat('bo_qap.mat', {'theta_ls_matrix':tls_matrix, 'tls_matrix':theta_ls_matrix, 'theta':theta, 'dim':dim})
            eng = matlab.engine.start_matlab()
            eng.run_sdp_qap(nargout=0)
            best_point = torch.from_numpy(np.argwhere(np.asarray(eng.workspace['results']['P_stochastic']) == 1)[:, 1])
            if not torch.all(best_point.unsqueeze(0) == train_x, axis=1).any():
                best_next_input = best_point.unsqueeze(0)
            else:
                print(f"Point already existing. Generating randomly!")
                best_next_input = torch.from_numpy(np.random.permutation(train_x[0].numpy())).unsqueeze(0)
            next_val = evaluate_qap(best_next_input, benchmark_index, dim)
            train_x = torch.cat([train_x, best_next_input])
            train_y = torch.cat([train_y, torch.tensor([next_val])])
            torch.save({'inputs_selected':train_x, 'outputs':train_y}, './results_collection/permutation/qap_kendall_ts_sdp_benchmark_index_'+str(benchmark_index)+'_'+str(nruns)+'.pkl')


def evaluate_tsp(x, benchmark_index, dim):
    if x.dim() == 2:
        x = x.squeeze(0)
    x = x.numpy()
    A = np.asarray(scipy.io.loadmat('./permutation_experiments/tsp_synthetic/pcb_dim_'+str(dim) + '_'+str(benchmark_index+1)+'.mat')['A'])
    B = np.asarray(scipy.io.loadmat('./permutation_experiments/tsp_synthetic/pcb_dim_'+str(dim) + '_'+str(benchmark_index+1)+'.mat')['B'])
    E = np.eye(dim)

    permutation = np.array([np.arange(dim), x])

    P = np.zeros([dim, dim]) #initialize the permutation matrix

    for i in range(0,dim):
        P[:, i] = E[:, permutation[1][i]]

    result = (np.trace(P.dot(B).dot(P.T).dot(A.T)))
    print(f"Objective value: {result/10000}")
    return result/10000


def mallows_bo_loop_tsp(dim=10, benchmark_index=0, kernel_type='mallows'):
    n_init = 20
    n_evals = 200
    for nruns in range(20):
        torch.manual_seed(nruns)
        np.random.seed(nruns)
        print(f'Input dimension {dim}')
        train_x = torch.from_numpy(np.array([np.random.permutation(np.arange(dim)) for _ in range(n_init)]))
        outputs = []
        for i in range(n_init):
            outputs.append(evaluate_tsp(train_x[i], benchmark_index, dim))
        outputs = np.array(outputs)
        train_y = -1*torch.Tensor(outputs)
        for num_iters in range(n_init, n_evals):
            inputs = featurize(train_x)
            if kernel_type == 'mallows':
                covar_module = MallowsKernel()
            elif kernel_type == 'rbf':
                covar_module = RBFKernel()
            train_y = (train_y - torch.mean(train_y))/(torch.std(train_y))
            inputs = inputs.double()
            train_y = train_y.double()
            mll_bt, model_bt = initialize_model(inputs, train_y.unsqueeze(1), covar_module)
            model_bt.likelihood.noise_covar.noise = torch.tensor(0.0001).double()
            mll_bt.model.likelihood.noise_covar.raw_noise.requires_grad = False
            fit_gpytorch_model(mll_bt)
            # print(train_y.dtype)
            print(f'\n -- NLL: {mll_bt(model_bt(inputs), train_y.double())}')
            EI = ExpectedImprovement(model_bt, best_f = train_y.max().item())
            # Multiple random restarts
            best_point, ls_val = EI_local_search(EI, torch.from_numpy(np.random.permutation(np.arange(dim))))
            for _ in range(10):
                new_point, new_val = EI_local_search(EI, torch.from_numpy(np.random.permutation(np.arange(dim))))
                if new_val > ls_val:
                    best_point = new_point
                    ls_val = new_val
            print(f"Best Local search value: {ls_val}")
            if not torch.all(best_point.unsqueeze(0) == train_x, axis=1).any():
                best_next_input = best_point.unsqueeze(0)
            else:
                print(f"Generating randomly !!!!!!!!!!!")
                best_next_input = torch.from_numpy(np.random.permutation(np.arange(dim))).unsqueeze(0)
            # print(best_next_input)
            next_val = evaluate_tsp(best_next_input, benchmark_index, dim)
            train_x = torch.cat([train_x, best_next_input])
            outputs = np.append(outputs, next_val)
            train_y = -1*torch.tensor(outputs)
            # train_y = torch.cat([train_y, torch.tensor([next_val])])
            print(f"\n\n Iteration {num_iters} with value: {outputs[-1]}")
            print(f"Best value found till now: {np.min(outputs)}")
            torch.save({'inputs_selected':train_x, 'outputs':outputs, 'train_y':train_y}, './results_collection/permutation/tsp_botorch_'+kernel_type+'_EI_dim_'+str(dim)+'benchmark_index_'+str(benchmark_index)+'_nrun_'+str(nruns)+'.pkl')

def spearman_bo_loop_tsp():
    n_init = 20
    n_evals = 200
    dim = 10  # int(sys.argv[1])
    benchmark_index = 0
    for nruns in range(20):
        np.random.seed(nruns)
        bo = SpearmanPermutationBayesianOptimization(
            f=None,
            n_length=dim,
        )
        train_x = torch.from_numpy(np.array([np.random.permutation(np.arange(dim)) for _ in range(n_init)]))
        outputs = []
        for i in range(n_init):
            outputs.append(evaluate_tsp(train_x[i], benchmark_index, dim))
        train_x_np = train_x.numpy()
        train_y = -torch.tensor(outputs)
        outputs = np.array(outputs)
        for x, y in zip(train_x_np, outputs):
            bo.register(x, -y)
        util = UtilityFunction(kind='ucb')
        for num_iters in range(n_init, n_evals):
            suggestion = bo.suggest(util)
            next_input = torch.from_numpy(np.array(list(suggestion.values()), dtype=int)).unsqueeze(0)
            next_val = -evaluate_tsp(next_input, benchmark_index, dim)
            train_x = torch.cat([train_x, next_input])
            train_y = torch.cat([train_y, torch.tensor([next_val])])
            x = next_input.numpy()[0]
            bo.register(x, next_val)
            print(f"Best value found till now: {-train_y.max().item()}")
            torch.save({'inputs_selected': train_x, 'outputs': -train_y},
                       './results_collection/permutation/tsp_spearman_ts_sdp_benchmark_index_' + str(
                           benchmark_index) + '_' + str(dim) + '_' + str(nruns) + '.pkl')


def spearman_bo_loop_qap():
    n_init = 20
    n_evals = 200
    benchmark_index = 3  # int(sys.argv[1])
    dim = np.asarray(
        scipy.io.loadmat('./permutation_experiments/qap_synthetic/QAP_LIB_A' + str(benchmark_index + 1) + '.mat')[
            'A']).shape[0]
    for nruns in range(20):
        np.random.seed(nruns)
        bo = SpearmanPermutationBayesianOptimization(
            f=None,
            n_length=dim,
        )
        train_x = torch.from_numpy(np.array([np.random.permutation(np.arange(dim)) for _ in range(n_init)]))
        outputs = []
        for i in range(n_init):
            outputs.append(evaluate_qap(train_x[i], benchmark_index, dim))
        train_x_np = train_x.numpy()
        train_y = -torch.tensor(outputs)
        outputs = np.array(outputs)
        for x, y in zip(train_x_np, outputs):
            bo.register(x, -y)
        util = UtilityFunction(kind='ucb')
        for num_iters in range(n_init, n_evals):
            suggestion = bo.suggest(util)
            next_input = torch.from_numpy(np.array(list(suggestion.values()), dtype=int)).unsqueeze(0)
            next_val = -evaluate_qap(next_input, benchmark_index, dim)
            train_x = torch.cat([train_x, next_input])
            train_y = torch.cat([train_y, torch.tensor([next_val])])
            x = next_input.numpy()[0]
            bo.register(x, next_val)
            print(f"Best value found till now: {-train_y.max().item()}")
            torch.save({'inputs_selected': train_x, 'outputs': -train_y},
                       './results_collection/permutation/qap_spearman_ts_sdp_benchmark_index_' + str(
                           benchmark_index) + '_' + str(dim) + '_' + str(nruns) + '.pkl')


if __name__ == '__main__':
    # spearman_bo_loop_qap()
    # spearman_bo_loop_tsp()
    # kendall_bo_loop_qap()
    # kendall_bo_loop_tsp()
    # mallows_bo_loop_qap()
    # mallows_bo_loop_tsp()
    # mallows_bo_loop_tsp(kernel_type='rbf')
    mallows_bo_loop_qap(kernel_type='rbf')
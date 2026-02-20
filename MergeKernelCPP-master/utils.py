from typing import Optional, List, Tuple
import torch
from torch import Tensor
import numpy as np
import numbers

from botorch import settings
from botorch.models.utils import check_no_nans, check_min_max_scaling, check_standardization
from botorch.exceptions import InputDataError, InputDataWarning
from scipy.optimize import fsolve


def merge_sort(arr):
    # print(f'Handling array: {arr}')
    if len(arr) == 1:
        return []
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        left_feature = merge_sort(left_half)
        right_feature = merge_sort(right_half)

        # print(f'L & R: {left_feature}, {right_feature}')
        feature = [] + left_feature + right_feature

        i = j = k = 0

        # Merge the two halves into the original list
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
                feature.append(0)
            else:
                arr[k] = right_half[j]
                j += 1
                feature.append(1)
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
            feature.append(1)

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
            feature.append(0)

        # print('Return: ', feature)
        return feature[:-1]


def featurize_merge(x):
    """
    Featurize the permutation vector into continuous space using merge kernel. Only available to permutation vector.
    """
    assert len(x.shape) == 2, "Only featurize 2 dimension permutation vector"
    feature = []
    for arr in x:
        feature.append(merge_sort(arr))
    return feature


def restore_merge_sort(arr, permutation):
    # print(f'Handling array: {arr} with permutation {permutation}')
    if len(permutation) == 2:
        if arr[0] == 0:
            return permutation
        else:
            return [permutation[1], permutation[0]]
    if len(permutation) == 3:
        if arr[1] + arr[2] == 0:
            right_permutation = [permutation[1], permutation[2]]
            left_permutation = [permutation[0]]
        elif arr[1] + arr[2] == 1:
            right_permutation = [permutation[0], permutation[2]]
            left_permutation = [permutation[1]]
        else:
            right_permutation = [permutation[0], permutation[1]]
            left_permutation = [permutation[2]]

        right_permutation = [right_permutation[0], right_permutation[1]] if arr[0] == 0 else [right_permutation[1],
                                                                                              right_permutation[0]]
        return left_permutation + right_permutation

    permutation_length = len(permutation)
    order = arr[-permutation_length + 1:]
    arr = arr[:-permutation_length + 1]

    left_permutation = []
    right_permutation = []

    # print('order: ', order)
    for index, i in enumerate(order):
        if i == 0:
            left_permutation.append(permutation[index])
        else:
            right_permutation.append(permutation[index])

        if len(left_permutation) == len(permutation) // 2:
            for j in range(index + 1, len(permutation)):
                right_permutation.append(permutation[j])
            break
        elif len(right_permutation) == int(np.ceil(len(permutation) / 2)):
            for j in range(index + 1, len(permutation)):
                left_permutation.append(permutation[j])
            break

    # print('left & right: ', left_permutation, right_permutation)
    if len(left_permutation) == len(right_permutation):
        mid = len(arr) // 2
        left_arr = arr[:mid]
        right_arr = arr[mid:]
    else:
        difference = int(np.floor(np.log2(len(left_permutation))) + 1)
        # print(difference)
        mid = (len(arr) - difference) // 2
        left_arr = arr[:mid]
        right_arr = arr[mid:]

    left = restore_merge_sort(left_arr, left_permutation)
    right = restore_merge_sort(right_arr, right_permutation)
    return left + right


def restore_featurize_merge(x, permutation):
    feature = []
    for arr in x:
        arr_norm = []
        for i in arr:
            if i > 0:
                arr_norm.append(1)
            else:
                arr_norm.append(0)
        feature.append(restore_merge_sort(arr_norm, permutation))
    return feature


def merge_sort_longer(arr):
    # print(f'Handling array: {arr}')
    if len(arr) == 1:
        return []
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        # print(len(left_half), len(right_half))

        postfix = []
        if len(left_half) > 1 or len(right_half) > 1:
            if left_half[-1] < right_half[-1]:
                postfix = [0]
            else:
                postfix = [1]

        left_feature = merge_sort_longer(left_half)
        right_feature = merge_sort_longer(right_half)

        # print(f'L & R: {left_feature}, {right_feature}')
        feature = [] + left_feature + right_feature

        i = j = k = 0

        # Merge the two halves into the original list
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
                feature.append(0)
            else:
                arr[k] = right_half[j]
                j += 1
                feature.append(1)
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
            feature.append(1)

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
            feature.append(0)

        # print('Return: ', feature)
        return feature[:-1] + postfix


def restore_merge_sort_longer(arr, permutation):
    # print(f'Handling array: {arr} with permutation {permutation}')
    if len(permutation) == 2:
        if arr[0] == 0:
            return permutation
        else:
            return [permutation[1], permutation[0]]
    if len(permutation) == 3:
        if arr[1] + arr[2] == 0:
            right_permutation = [permutation[1], permutation[2]]
            left_permutation = [permutation[0]]
        elif arr[1] + arr[2] == 1:
            right_permutation = [permutation[0], permutation[2]]
            left_permutation = [permutation[1]]
        else:
            right_permutation = [permutation[0], permutation[1]]
            left_permutation = [permutation[2]]

        right_permutation = [right_permutation[0], right_permutation[1]] if arr[0] == 0 else [right_permutation[1],
                                                                                              right_permutation[0]]
        return left_permutation + right_permutation

    arr = arr[:-1]
    permutation_length = len(permutation)
    order = arr[-permutation_length + 1:]
    arr = arr[:-permutation_length + 1]

    left_permutation = []
    right_permutation = []

    # print('order: ', order)
    for index, i in enumerate(order):
        if i == 0:
            left_permutation.append(permutation[index])
        else:
            right_permutation.append(permutation[index])

        if len(left_permutation) == len(permutation) // 2:
            for j in range(index + 1, len(permutation)):
                right_permutation.append(permutation[j])
            break
        elif len(right_permutation) == int(np.ceil(len(permutation) / 2)):
            for j in range(index + 1, len(permutation)):
                left_permutation.append(permutation[j])
            break

    # print('left & right: ', left_permutation, right_permutation)
    if len(left_permutation) == len(right_permutation):
        mid = len(arr) // 2
        left_arr = arr[:mid]
        right_arr = arr[mid:]
    else:
        dummy_permutation = list(range(len(left_permutation)))
        dummy_right = list(range(len(left_permutation) + 1))
        difference = len(merge_sort_longer(dummy_right)) - len(merge_sort_longer(dummy_permutation))

        # print(difference)
        mid = (len(arr) - difference) // 2
        left_arr = arr[:mid]
        right_arr = arr[mid:]

    left = restore_merge_sort_longer(left_arr, left_permutation)
    right = restore_merge_sort_longer(right_arr, right_permutation)
    return left + right


def featurize_merge_longer(x):
    """
    Featurize the permutation vector into continuous space using merge kernel. Only available to permutation vector.
    """
    assert len(x.shape) == 2, "Only featurize 2 dimension permutation vector"
    feature = []
    for arr in x:
        feature.append(merge_sort_longer(arr))
    return feature


def restore_featurize_merge_longer(x, permutation):
    feature = []
    for arr in x:
        arr_norm = []
        for i in arr:
            if i > 0:
                arr_norm.append(1)
            else:
                arr_norm.append(0)
        feature.append(restore_merge_sort_longer(arr_norm, permutation))
    return feature


def featurize_mallows(x):
    """
    Featurize the permutation vector into continuous space. Only available to permutation vector.
    """
    assert len(x.shape) == 2, "Only featurize 2 dimension permutation vector"
    x_repeat = np.repeat(x, x.shape[-1], axis=1).reshape(x.shape[0], x.shape[1], -1)
    x_feature = np.transpose(x_repeat, (0, 2, 1)) - x_repeat
    x_feature = np.sign([i[np.triu_indices(x.shape[-1], k=1)] for i in x_feature])
    normalizer = np.sqrt(x.shape[1] * (x.shape[1] - 1) / 2)
    return x_feature / normalizer


def reverse_featurize_mallows(x_feature):
    assert len(x_feature.shape) == 2 or len(x_feature.shape) == 1, "Only reverse featurize 1 or 2 dimension permutation vector"
    expand = False
    if len(x_feature.shape) == 1:
        x_feature = x_feature.reshape(1, -1)
        expand = True
    # x_feature = np.sign(x_feature)
    permutation_length = int(np.sqrt(x_feature.shape[-1] * 2)) + 1
    x = []
    for i in range(x_feature.shape[0]):
        x_i = np.zeros((permutation_length, permutation_length))
        row_idx = 0
        col_idx = row_idx + 1
        for num in x_feature[i]:
            x_i[row_idx, col_idx] = num
            if col_idx == permutation_length - 1:
                row_idx += 1
                col_idx = row_idx + 1
                continue
            col_idx += 1
        x_permutation_i = []
        for j in range(permutation_length):
            permutation_i = np.sum(x_i[j]) - np.sum(x_i[:, j])
            x_permutation_i.append(permutation_i)
        x_permutation_i = np.argsort(x_permutation_i)
        temp = [0 for i in range(permutation_length)]
        for j in range(permutation_length):
            temp[x_permutation_i[j]] = j
        x.append(temp)
    x = np.array(x)
    if expand:
        x = x[0]
    return x


def split_value_permutation(x_combine, value_length):
    return np.split(x_combine, indices_or_sections=[value_length], axis=-1)


def concat_value_permutation(x_value, x_permutation):
    return np.concatenate([x_value, x_permutation], axis=-1)


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


def validate_input_double_scaling(
        train_X: Tuple[Tensor, Tensor],
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        raise_on_fail: bool = False,
        ignore_X_dims: Optional[List[int]] = None,
) -> None:
    r"""Helper function to validate input data to models.

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of
            training features.
        train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
            training observations.
        train_Yvar: A `batch_shape x n x m` or `batch_shape x n x m` (batch mode)
            tensor of observed measurement noise.
        raise_on_fail: If True, raise an error instead of emitting a warning
            (only for normalization/standardization checks, an error is always
            raised if NaN values are present).
        ignore_X_dims: For this subset of dimensions from `{1, ..., d}`, ignore the
            min-max scaling check.

    This function is typically called inside the constructor of standard BoTorch
    models. It validates the following:
    (i) none of the inputs contain NaN values
    (ii) the training data (`train_X`) is normalized to the unit cube for all
    dimensions except those in `ignore_X_dims`.
    (iii) the training targets (`train_Y`) are standardized (zero mean, unit var)
    No checks (other than the NaN check) are performed for observed variances
    (`train_Yvar`) at this point.
    """
    if settings.validate_input_scaling.off():
        return
    check_no_nans(train_X[0])
    check_no_nans(train_X[1])
    check_no_nans(train_Y)
    if train_Yvar is not None:
        check_no_nans(train_Yvar)
        if torch.any(train_Yvar < 0):
            raise InputDataError("Input data contains negative variances.")
    check_min_max_scaling(
        X=train_X[0], raise_on_fail=raise_on_fail, ignore_dims=ignore_X_dims
    )
    check_min_max_scaling(
        X=train_X[1], raise_on_fail=raise_on_fail, ignore_dims=ignore_X_dims
    )
    check_standardization(Y=train_Y, raise_on_fail=raise_on_fail)


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    if isinstance(x, tuple):
        return x[0].shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def generate_guess_with_constraint(constraint, bounds):
    def bound_check(result):
        for i in range(len(result)):
            if result[i] > bounds[i][1] or result[i] < bounds[i][0]:
                return False
        return True

    if constraint['type'] == 'eq':
        while True:
            initial_guess = np.array([np.random.rand() for _ in range(len(bounds))])
            result = fsolve(constraint['fun'], initial_guess)
            if abs(constraint['fun'](result).sum() < 1e-6) and bound_check(result):
                return result
    elif constraint['type'] == 'ineq':
        while True:
            initial_guess = []
            for i in range(len(bounds)):
                initial_guess.append(np.random.uniform(bounds[i][0], bounds[i][1]))
            initial_guess = np.array(initial_guess)
            if constraint['fun'](initial_guess) > 0:
                return initial_guess


def item_dict_to_values(items):
    result = []
    for i in range(len(items)):
        if items['n' + str(i)] == 1:
            result.append(i + 1)
    return np.array(result)


def get_problem_name(problem_args):
    return '_'.join(problem_args)

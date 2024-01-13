import random
import copy
import numpy as np


# ======================= Functions to simplify the algorithm ==================
def DictMin(Dict):
    min_key = min(Dict, key=Dict.get)
    return min_key


def XInt(x):
    # The output is a list of touples.
    # The first element in the touple shows row index and the second shows the column index of non-integer.
    non_integer = []
    for key in x.keys():
        if x[key] != int(x[key]):
            non_integer.append(key)
    return non_integer


def YInt(y):
    non_integer = []
    y_diff = {key: min(y[key]-0, 1-y[key]) for key in y.keys()}
    min_diff = 1
    selected_key = None
    for key in y_diff.keys():
        if y_diff[key] != int(y_diff[key]):
            non_integer.append(key)
            if y_diff[key] <= min_diff:
                selected_key = key
                min_diff = y_diff[key]
    return non_integer, selected_key


def IndexUp(dict):
    new_dict = {}
    for key in dict.keys():
        new_dict[(key[0] + 1, key[1] + 1)] = copy.copy(dict[key])
    return new_dict


def SparseRowCount(Dict):
    return np.max([i[0] for i in Dict.keys()])


if __name__ == '__main__':
    # YInt
    e1 = {1: 0.2, 2: 0.6, 3: 0}
    keys, min = YInt(e1)
    if (keys, min) == ([1, 2], 1):
        print('YInt works properly.')
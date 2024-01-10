import random
import copy
import numpy as np


# ======================= Functions to simplify the algorithm ==================
def DictMin(Dict):
    minim = min(Dict.values())
    arg = 0
    for key, value in zip(Dict.keys(), Dict.values()):
        if value == minim:
            arg = key
            break
    return arg, minim


def XInt(x):
    # The output is a list of touples.
    # The first element in the touple shows row index and the second shows the column index of non-integer.
    non_integer = []
    for l in x.keys():
        for ii_d in x[l].keys():
            if x[l][ii_d] != int(x[l][ii_d]):
                non_integer.append([l, ii_d])
    return non_integer


def YInt(y):
    non_integer = []
    for element in y.keys():
        if y[element] != int(y[element]):
            non_integer.append(element)
    return non_integer


def IndexUp(dict):
    new_dict = {}
    for key in dict.keys():
        new_dict[(key[0] + 1, key[1] + 1)] = copy.copy(dict[key])
    return new_dict


def SparseRowCount(Dict):
    return np.max([i[0] for i in Dict.keys()])


if __name__ == '__main__':
    # DictMin
    d = {1: 45, 2: 76, 3: -24}
    key, value = DictMin(d)
    if (key, value) == (3, -24.):
        print('DictMin works properly.')

    # XInt
    e1 = {1: {(1, 1): 2.2, (1, 2): 5, (1, 3): 8},
          2: {(1, 1): 4, (1, 2): 6.4, (1, 3): 9}}
    output1 = XInt(e1)
    if output1 == [[1, (1, 1)], [2, (1, 2)]]:
        print('XInt works properly.')

    # IndexUp
    d = {(0, 0): 3, (0, 1): 8, (1, 2): 9}
    if IndexUp(d) == {(1, 1): 3, (1, 2): 8, (2, 3): 9}:
        print('IndexUp works properly')

    # SparseRowCount
    dd = {(0, 0): 3, (0, 1): 8, (1, 2): 9, (2, 0): 9}
    if SparseRowCount(d) == 1:
        print('SparseRowCount works properly')

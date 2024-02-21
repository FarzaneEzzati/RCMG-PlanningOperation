import random
import copy
import numpy as np


# ======================= Functions to simplify the algorithm ==================
def DictMin(Dict):
    min_key = min(Dict, key=Dict.get)
    return min_key





def YInt(y):
    non_integer = []
    for key in y.keys():
        if y[key] != int(y[key]):
            non_integer.append(key)
    return non_integer

def SelectSplitKey(y, non_int):
    y_diff = {key: min(y[key] - 0, 1 - y[key]) for key in y.keys()}
    min_diff = 1
    selected_key = None
    for key in non_int:
        if y[key] < min_diff:
            selected_key = key
            min_diff = y[key]
    return selected_key

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
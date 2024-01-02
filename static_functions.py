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
    for element in x.keys():
        if x[element] != int(x[element]):
            non_integer.append(element)
    return non_integer


def IndexUp(dict):
    new_dict = {}
    for key in dict.keys():
        new_dict[(key[0]+1, key[1]+1)] = copy.copy(dict[key])
    return new_dict


if __name__ == '__main__':
    # DictMin
    d = {1: 45, 2: 76, 3: -24}
    key, value = DictMin(d)
    if (key, value) == (3, -24.):
        print('DictMin works properly.')

    # XInt
    e1 = {1: 1, 2: 2.2, 3: 5, 4: 2.1, 5: 5}
    output1 = XInt(e1)
    e2 = {1: 1, 2: 2.0, 3: 5, 4: 2.0, 5: 5}
    output2 = XInt(e2)
    if (output1 == [2, 4], output2 == []) == (True, True):
        print('XInt works properly.')

    # IndexUp
    d = {(0, 0): 3, (0, 1): 8, (1, 2): 9}
    if IndexUp(d) == {(1, 1): 3, (1, 2): 8, (2, 3): 9}:
        print('IndexUp works properly')
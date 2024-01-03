import pickle
import random
import copy
import numpy as np
from static_functions import DictMin, IndexUp, XInt
import gurobipy
from gurobipy import quicksum, GRB
env = gurobipy.Env()
env.setParam('OutputFlag', 0)

M = 100
master = gurobipy.Model('MaterToy', env=env)
indices = [1, 2]
X = master.addVars(indices, lb=0, ub=1, name='X')
eta = master.addVar(lb=-M, name='eta')
master.setObjective(- 1.5 * X[1] - 4 * X[2] + eta, sense=GRB.MINIMIZE)
master.update()
master.write('Models/MasterToy.mps')
with open(f'Models/MasterToy-indices.pkl', 'wb') as handle:
    pickle.dump(indices, handle)
handle.close()

def SubProb(rDict, scen):
    sub = gurobipy.Model('Sub', env=env)
    X_indices = [1, 2]
    X = sub.addVars(X_indices, lb=0, ub=1, name='X')
    XX = range(1, 3)

    Y_indices = [1, 2, 3, 4]
    Y = sub.addVars(Y_indices, name='Y')
    R = sub.addVar(name='R')
    YY = range(3, 8)
    sub.update()
    sub.addConstr(2*Y[1] - 3*Y[2] + 4*Y[3] + 5*Y[4] - R + X[1] <= rDict[1], name='C1')
    sub.addConstr(6*Y[1] + Y[2] + 3*Y[3] + 2*Y[4] - R + X[2] <= rDict[2], name='C2')
    for i in Y_indices:
        sub.addConstr(Y[i] <= 5, name='PrimalUpper')
        sub.addConstr(-Y[i] <= 0, name='PrimalLower')
    sub.setObjective(-16 * Y[1] - 19 * Y[2] - 23 * Y[3] - 28 * Y[4] + 100 * R, GRB.MINIMIZE)
    sub.update()
    sub.write(f'Models/SubToy{scen}.mps')
    with open(f'Models/SubToy{scen}-indices.pkl', 'wb') as f:
        pickle.dump([X_indices, Y_indices, XX, YY], f)
    f.close()


if __name__ == '__main__':
    SubProb({1: 10, 2: 4}, 1)
    SubProb({1: 6, 2: 2}, 2)

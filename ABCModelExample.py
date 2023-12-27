import pickle
import time
import random
import numpy as np
import pandas as pd
import gurobipy
from gurobipy import quicksum, GRB
env = gurobipy.Env()
env.setParam('OutputFlag', 1)

'''M = float('inf')
master = gurobipy.Model('MaterToy')
X = master.addVars([1, 2], name='X')
E = master.addVar(name='eta')
master.addConstr(E >= -M, name='eta_lb')
master.setObjective(-1.5 * X[1] - 4 * X[2] + E)
master.write('Models/MasterToy.mps')'''

sub = gurobipy.Model('SubToy1')

Y_indices = [1, 2, 3, 4]
Y = sub.addVars(Y_indices, ub=5, lb=0, name='Y')
R = sub.addVar(name='R')
YY = list(range(5))

X_indices = [1, 2]
Xfix = sub.addVars(X_indices, name='X')
XX = list(range(5, 7))

sub.addConstr(2*Y[1] - 3*Y[2] + 4*Y[3] + 5*Y[4] - R + Xfix[1] <= 10)
sub.addConstr(6*Y[1] + Y[2] + 3*Y[3] + 2*Y[4] - R + Xfix[2] <= 3)
sub.addConstr(Xfix[1] == 0)
sub.addConstr(Xfix[2] == 0)
sub.setObjective(-16 * Y[1] - 19 * Y[2] - 23 * Y[3] - 28 * Y[4] + 100 * R, GRB.MINIMIZE)
sub.update()
Coefs = sub.getA().todok()  # dictionary
shape = Coefs.shape
# 4-2 is the count of equality constraints that we do not need
m = shape[0]-2
A = {}
b = {}
T = {}
W = {}
r = {}
for row in range(m):
    for yindex in YY:
        if (row, yindex) in Coefs:
            W[(row+1, yindex+1)] = Coefs[(row, yindex)]
    for xindex in XX:
        if (row, xindex) in Coefs:
            T[(row+1, xindex+1)] = Coefs[(row, xindex)]

print(Coefs.todok())
print(W)
print(T)


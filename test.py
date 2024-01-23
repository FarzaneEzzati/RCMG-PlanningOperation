# Dec 14, 2023. Repository was created to keep track of codings for DOE project.

import copy
import pickle
import random
import math
import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import GRB

master = gp.Model()
X = master.addVars((1, 2), ub=1, name='X')
eta = master.addVar(lb=-100000, name='eta')
A = {}
b = {}
master.setObjective(-1.5 * X[1] - 4 * X[2] + eta, GRB.MINIMIZE)
master.write('Models/Master.mps')
with open(f'Models/Master.pkl', 'wb') as handle:
    pickle.dump([A, b, {1: 1, 2: 1}, {1: 0, 2: 0}], handle)
handle.close()
X_keys = [1, 2]
Y_keys = [3, 4, 5, 6, 7]
Y_int_keys = [3, 4, 5, 6]

sub1 = gp.Model()
X = sub1.addVars((1, 2), ub=1, name='X')
Y = sub1.addVars((1, 2, 3, 4), ub=1, name='Y')
R = sub1.addVar(lb=0, name='R')
sub1.addConstr(-(2 * Y[1] + 3 * Y[2] + 4 * Y[3] + 5 * Y[4]) + R >= -10 + X[1], name='C1')
sub1.addConstr(-(6 * Y[1] + 1 * Y[2] + 3 * Y[3] + 2 * Y[4]) + R >= -3 + X[2], name='C2')
sub1.setObjective(-16 * Y[1] - 19 * Y[2] - 23 * Y[3] - 28 * Y[4] + 100 * R, sense=GRB.MINIMIZE)
sub1.update()
W = {(1, 3): -2, (1, 4): -3, (1, 5): -4, (1, 6): -5, (1, 7): 1,
     (2, 3): -6, (2, 4): -1, (2, 5): -3, (2, 6): -2, (2, 7): 1}
T = {(1, 1): -1, (2, 2): -1}
r = {1: -10, 2: -3}
ub = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1000}
lb = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
sub1.write('Models/Sub1.mps')
with open(f'Models/Sub1.pkl', 'wb') as f:
    pickle.dump([T, W, r, ub, lb], f)
f.close()

sub2 = gp.Model()
X = sub2.addVars((1, 2), ub=1, name='X')
Y = sub2.addVars((1, 2, 3, 4), ub=1, name='Y')
R = sub2.addVar(lb=0, name='R')
sub2.addConstr(-(2 * Y[1] + 3 * Y[2] + 4 * Y[3] + 5 * Y[4]) + R >= -5 + X[1], name='C1')
sub2.addConstr(-(6 * Y[1] + 1 * Y[2] + 3 * Y[3] + 2 * Y[4]) + R >= -2 + X[2], name='C2')
sub2.setObjective(-16 * Y[1] - 19 * Y[2] - 23 * Y[3] - 28 * Y[4] + 100 * R, sense=GRB.MINIMIZE)
sub2.update()
W = {(1, 3): -2, (1, 4): -3, (1, 5): -4, (1, 6): -5, (1, 7): 1,
     (2, 3): -6, (2, 4): -1, (2, 5): -3, (2, 6): -2, (2, 7): 1}
T = {(1, 1): -1, (2, 2): -1}
r = {1: -5, 2: -2}
ub = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1000}
lb = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
sub2.write('Models/Sub2.mps')
with open(f'Models/Sub2.pkl', 'wb') as f:
    pickle.dump([T, W, r, ub, lb], f)
f.close()

with open(f'Models/Indices.pkl', 'wb') as f:
    pickle.dump([X_keys, Y_keys, Y_int_keys], f)
f.close()


model = gp.Model()
X = model.addVars((1, 2), vtype=GRB.BINARY, name='X')
Y1 = model.addVars((1, 2, 3, 4), vtype=GRB.BINARY, name='Y')
R1 = model.addVar(name='R')
Y2 = model.addVars((1, 2, 3, 4), vtype=GRB.BINARY, name='Y')
R2 = model.addVar(name='R')
model.addConstr(-(2 * Y1[1] + 3 * Y1[2] + 4 * Y1[3] + 5 * Y1[4]) + R1 >= -10 + X[1], name='C1')
model.addConstr(-(6 * Y1[1] + 1 * Y1[2] + 3 * Y1[3] + 2 * Y1[4]) + R1 >= -3 + X[2], name='C2')
model.addConstr(-(2 * Y2[1] + 3 * Y2[2] + 4 * Y2[3] + 5 * Y2[4]) + R2 >= -5 + X[1], name='C1')
model.addConstr(-(6 * Y2[1] + 1 * Y2[2] + 3 * Y2[3] + 2 * Y2[4]) + R2 >= -2 + X[2], name='C2')
model.setObjective(-1.5 * X[1] - 4 * X[2] +
                   0.5 * (-16 * Y1[1] - 19 * Y1[2] - 23 * Y1[3] - 28 * Y1[4] + 100 * R1) +
                   0.5 * (-16 * Y2[1] - 19 * Y2[2] - 23 * Y2[3] - 28 * Y2[4] + 100 * R2), sense=GRB.MINIMIZE)
model.optimize()
print(X)
print(model.ObjVal)
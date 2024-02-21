import pickle
import random

from static_functions import *
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import quicksum, GRB
from static_functions import IndexUp
from MasterSub import SubProb, MasterProb
import time
env = gp.Env()
env.setParam('OutputFlag', 0)

class Model:
    def __init__(self, m, X_keys):
        self.m = m
        Xs = self.m.getVars()
        self.X = {key: Xs[key - 1] for key in X_keys}


Xs = [1, 2, 3, 4]
m = gp.Model('KP', env=env)
X = m.addVars(Xs, lb=0, ub=1, name='X')
m.addConstr(4*X[1] + 5*X[2] + 3*X[3] + 6*X[4] <= 8, name='A')
m.addConstrs((X[i] <= 1 for i in (1, 2, 3, 4)), name='Upper')
m.addConstrs((-X[i] <= 0 for i in (1, 2, 3, 4)), name='Lower')
m.setObjective(12*X[1] + 14*X[2] + 7*X[3] + 12*X[4], sense=GRB.MAXIMIZE)
m.update()
m.optimize()

v, V = float('inf'), -float('inf')
counter_t = 0
while True:
    print(f'V:{V},    v:{v}')
    m.optimize()
    X_star = {x: X[x].x for x in Xs}
    print(X_star)
    non_integers = [key for key in X_star if X_star[key] != int(X_star[key])]
    if not non_integers:
        print(f'Optimal solution found and is: {X_star}')
        break
    else:
        if m.ObjVal < v:
            v = m.ObjVal
        for_branch = random.choice(non_integers)
        lb1 = {}
        lb2 = {}
        ub1 = {}
        ub2 = {}
        for var in X:
            if var == for_branch:
                lb1[var] = 1
                ub1[var] = 1
                lb2[var] = 0
                ub2[var] = 0
            else:
                lb1[var] = m.getConstrByName(f'Lower[{var}]').RHS
                ub1[var] = m.getConstrByName(f'Upper[{var}]').RHS
                lb2[var] = m.getConstrByName(f'Lower[{var}]').RHS
                ub2[var] = m.getConstrByName(f'Upper[{var}]').RHS

        # Build CGLP
        constrs = m.getConstrs()
        A_constrs = {constrs.index(c) + 1: c for c in constrs if 'A' in c.ConstrName}
        A_matrix = IndexUp(m.getA().todok())
        A_coefficients = {key: A_matrix[key] for key in A_matrix if key[0] in A_constrs}
        A_constrs_count = len(A_constrs)
        cglp = gp.Model('CGLP', env=env)
        L = cglp.addVars(A_constrs, name='L')
        l1_lb = cglp.addVars(Xs, name='l1_lb')
        l1_ub = cglp.addVars(Xs, name='l1_ub')
        l2_lb = cglp.addVars(Xs, name='l2_lb')
        l2_ub = cglp.addVars(Xs, name='l2_ub')
        pi0 = cglp.addVar(lb=-1, ub=1, name='pi0')
        pi1 = cglp.addVars(Xs, lb=-1, ub=1, name='pi1')

        cglp.addConstr(pi0 >= sum(L[i] * A_constrs[i].RHS for i in A_constrs) +
                       sum(l1_lb[x] * lb1[x] + l1_ub[x] * ub1[x] for x in Xs))
        cglp.addConstr(pi0 >= sum(L[i] * A_constrs[i].RHS for i in A_constrs) +
                       sum(l2_lb[x] * lb2[x] + l2_ub[x] * ub2[x] for x in Xs))
        cglp.addConstrs(pi1[x] <= sum(L[i[0]] * A_coefficients[i] for i in A_coefficients if i[1] == x) +
                       l1_lb[x] - l1_ub[x] for x in Xs)
        cglp.addConstrs(pi1[x] <= sum(L[i[0]] * A_coefficients[i] for i in A_coefficients if i[1] == x) +
                       l2_lb[x] - l2_ub[x] for x in Xs)
        cglp.addConstr(sum(L[i] for i in L) +
                       sum(l1_lb[i] for i in l1_lb) + sum(l1_ub[i] for i in l1_ub) +
                       sum(l2_lb[i] for i in l2_lb) + sum(l2_ub[i] for i in l2_ub) == 1)

        cglp.setObjective(sum(X_star[x] * pi1[x] for x in Xs) - pi0, sense=GRB.MAXIMIZE)
        cglp.optimize()

        PI0 = pi0.x
        PI1 = {x: pi1[x].x for x in Xs}

        m.addConstr(sum(X[x]*PI1[x] for x in Xs) <= PI0, name='A')
        m.update()
        cglp.dispose()

    counter_t += 1


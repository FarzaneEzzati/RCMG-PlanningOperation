import pickle
from static_functions import *
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import quicksum, GRB

from MasterSub import SubProb, MasterProb
import time
env = gp.Env()
env.setParam('OutputFlag', 0)


with open(f'Models/Indices.pkl', 'rb') as f:
    X_keys, Y_keys, Y_int_keys = pickle.load(f)
f.close()
D = 2
delta = 0.0005


def ConvCGSEP(master, sub, x_k, Gz, gz):
    y_k, v_k = sub.FixXSolve(x_k)
    m1 = len(master.master.getConstrs())
    m2 = len(sub.sub.getConstrs())
    A, b = master.A, master.b
    T, W, r = sub.ReturnT_W_r()

    cgsep = gp.Model(env=env)
    u1 = cgsep.addVars(range(1, m1 + 1), ub=1-delta, name='u1')
    u2 = cgsep.addVars(range(1, m2 + 1), ub=1-delta, name='u2')
    ax = cgsep.addVars(sub.X.keys(), vtype=GRB.INTEGER, name='ax')
    ay = cgsep.addVars(sub.Y.keys(), vtype=GRB.INTEGER, name='ay')
    ax0 = cgsep.addVar(vtype=GRB.INTEGER, name='ax0')
    ay0 = cgsep.addVar(vtype=GRB.INTEGER, name='ay0')

    fx = cgsep.addVars(sub.X.keys(), ub=1-delta, name='fx')
    fy = cgsep.addVars(sub.Y.keys(), ub=1-delta, name='fy')
    f0 = cgsep.addVar(ub=1-delta, name='fx0')

    C = sub.sub.getConstrs()
    for c in C:
        if c.Sense == '=':
            u2[C.index(c) + 1].LB = -GRB.INFINITY

    for y_key in sub.Y.keys():
        if y_key not in sub.YIK:
            ay[y_key].LB = 0
            ay[y_key].UB = 0

    uA = {x_key: 0 for x_key in sub.X.keys()}
    for key in A.keys():
        uA[key[1]] += u1[key[0]] * A[key]
    uT = {x_key: 0 for x_key in sub.X.keys()}
    for key in T.keys():
        uT[key[1]] += u2[key[0]] * T[key]
    cgsep.addConstrs((fx[x_key] == uA[x_key] + uT[x_key] - ax[x_key] for x_key in sub.X.keys()), name='fx')

    uW = {y_key: 0 for y_key in sub.Y.keys()}
    for key in W.keys():
        uW[key[1]] += u2[key[0]] * W[key]
    cgsep.addConstrs((fy[y_key] == uW[y_key] - ay[y_key] for y_key in sub.Y.keys()), name='fy')

    cgsep.addConstr(f0 == sum(u1[i1] * b[i1] for i1 in b.keys()) +
                    sum(u2[i2] * r[i2] for i2 in r.keys()) -
                    ax0 - ay0, name='f0')
    cgsep.setObjective(sum(ax[x_key] * x_k[x_key] for x_key in sub.X.keys()) +
                    sum(ay[y_key] * y_k[y_key] for y_key in sub.Y.keys()) - ax0 - ay0, sense=GRB.MAXIMIZE)
    cgsep.update()
    cgsep.optimize()

    AX = quicksum(ax[x_key].x * sub.X[x_key] for x_key in sub.X.keys())
    AY = quicksum(ay[y_key].x * sub.Y[y_key] for y_key in sub.Y.keys())
    sub.sub.addConstr(AX + AY >= ax0.x + ay0.x, name=f'CGLP{gz}')
    print(AX + AY >= ax0.x + ay0.x)
    sub.sub.update()
    nr = len(sub.sub.getConstrs())  # nr: new row
    for x_key in ax.keys():
        sub.T[(nr, x_key)] = ax[x_key].x
    for y_key in ay.keys():
        sub.W[(nr, y_key)] = ay[y_key].x
    sub.r[nr] = ax0.x + ay0.x





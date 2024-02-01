import pickle
from static_functions import *
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import quicksum, GRB
from SubBB import SubBB
from MasterSub import SubProb, MasterProb
import time
env = gp.Env()
env.setParam('OutputFlag', 0)


class CGLP:
    def __init__(self, master, sub, T2, x_k, y_d):
        self.sub, self.T2 = sub, T2
        self.T, self.W, self.r = sub.ReturnT_W_r()
        ub_2 = {t2: T2[t2].ub for t2 in self.T2.keys()}
        lb_2 = {t2: T2[t2].lb for t2 in self.T2.keys()}
        A_1, b_1, ub_1, lb_1 = master.ReturnA_b_ub_lb()
        A_cons = master.master.getConstrs()
        sub_cons = sub.sub.getConstrs()
        self.cglp = gp.Model('CGLP', env=env)
        L1 = self.cglp.addVars(range(1, len(A_cons) + 1), lb=0, name='L1')
        self.Mu1 = self.cglp.addVars(master.X.keys(), lb=0, name='Mu1')
        self.Nu1 = self.cglp.addVars(master.X.keys(), lb=0, name='Nu1')
        self.Mu2 = {t2: self.cglp.addVars(sub.Y.keys(), lb=0, name='Mu2') for t2 in T2.keys()}
        self.Nu2 = {t2: self.cglp.addVars(sub.Y.keys(), lb=0, name='Nu2') for t2 in T2.keys()}
        self.PI0 = self.cglp.addVar(lb=-1, ub=1, name='P0')
        self.PI1 = self.cglp.addVars(master.X.keys(), lb=-1, ub=1, name='P1')
        self.PI2 = self.cglp.addVars(sub.Y.keys(), lb=-1, ub=1, name='P2')
        self.L2 = {t2: self.cglp.addVars(range(1, len(sub_cons) + 1), name='L2') for t2 in T2.keys()}

        for c in A_cons:
            if c.Sense == '=':
                L1[A_cons.index(c) + 1].LB = -GRB.INFINITY
        for c in sub_cons:
            if c.Sense == '=':
                for t2 in self.T2.keys():
                    self.L2[t2][sub_cons.index(c) + 1].LB = -GRB.INFINITY

        self.rL2 = {t2: quicksum(self.r[key] * self.L2[t2][key] for key in self.r.keys()) for t2 in T2.keys()}
        self.MuLo = {t2: quicksum(lb_2[t2][key] * self.Mu2[t2][key] for key in sub.Y.keys()) for t2 in T2.keys()}
        self.NuUp = {t2: quicksum(ub_2[t2][key] * self.Nu2[t2][key] for key in sub.Y.keys()) for t2 in T2.keys()}

        self.bL1 = quicksum(b_1[key] * L1[key] for key in b_1.keys())
        self.MuLo1 = quicksum(lb_1[key] * self.Mu1[key] for key in sub.X.keys())
        self.NuUp1 = quicksum(ub_1[key] * self.Nu1[key] for key in sub.X.keys())

        self.TL2 = {t2: {x_key: 0 for x_key in sub.X.keys()} for t2 in self.T2.keys()}
        for t2 in T2.keys():
            for key in self.T.keys():
                self.TL2[t2][key[1]] += self.T[key] * self.L2[t2][key[0]]

        self.AL1 = {x_key: 0 for x_key in sub.X.keys()}
        for key in A_1.keys():
            self.AL1[key[1]] += A_1[key] * L1[key[0]]

        self.WL2 = {t2: {y_key: 0 for y_key in sub.Y.keys()} for t2 in self.T2.keys()}
        for t2 in self.T2.keys():
            for key in self.W.keys():
                self.WL2[t2][key[1]] += self.W[key] * self.L2[t2][key[0]]
        self.SetConstrs()
        self.cglp.setObjective(self.PI0 -
                          quicksum(self.PI1[x_key] * x_k[x_key] for x_key in sub.X.keys()) -
                          quicksum(self.PI2[y_key] * y_d[y_key] for y_key in sub.Y.keys()), sense=GRB.MAXIMIZE)
        self.cglp.update()

    def Update(self, pi0, pi1, pi2):
        new_key = len(self.L2[1].keys()) + 1
        for t2 in self.T2.keys():
            self.L2[t2][new_key] = self.cglp.addVar(name='L2')
            self.rL2[t2] += pi0 * self.L2[t2][new_key]
            for pi1_key in pi1.keys():
                self.TL2[t2][pi1_key] += pi1[pi1_key] * self.L2[t2][new_key]
            for pi2_key in pi2.keys():
                self.WL2[t2][pi2_key] += pi2[pi2_key] * self.L2[t2][new_key]
        self.cglp.remove(self.cglp.getConstrs())
        self.SetConstrs()
        self.cglp.update()

    def SetConstrs(self):
        self.cglp.addConstrs((self.PI0 <= self.bL1 + self.rL2[t2] + self.MuLo[t2] - self.NuUp[t2] + self.MuLo1 - self.NuUp1
                        for t2 in self.T2.keys()), name='pi0')
        self.cglp.addConstrs((self.PI1[x_key] == self.AL1[x_key] + self.TL2[t2][x_key] + self.Mu1[x_key] - self.Nu1[x_key]
                        for t2 in self.T2.keys() for x_key in self.sub.X.keys()), name='pi1')
        self.cglp.addConstrs((self.PI2[y_key] == self.WL2[t2][y_key] + self.Mu2[t2][y_key] - self.Nu2[t2][y_key]
                        for t2 in self.T2.keys() for y_key in self.sub.Y.keys()), name='pi2')
        self.cglp.update()

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
        self.x_k, self.y_d = x_k, y_d


        # Get A and b from master
        constrs_coefs_1 = IndexUp(master.master.getA().todok())
        constr_1 = master.master.getConstrs()

        benders_cuts_cuont = sum([1 for c in constr_1 if 'Benders' in c.ConstrName])

        if benders_cuts_cuont == len(constr_1):
            A_1 = None
            b_1 = None
        else:
            A_length = len(constr_1) - benders_cuts_cuont
            A_1, b_1 = {}, {}
            for key in constrs_coefs_1:
                if key[0] <= A_length:
                    A_1[key] = constrs_coefs_1[key]
                    b_1[key[0]] = constr_1[key[0] - 1].RHS


        b_1 = {constr_1.index(c) + 1: c.RHS for c in constr_1}
        ub_1 = {x: master.X[x].UB for x in master.X}
        lb_1 = {x: master.X[x].LB for x in master.X}

        # Get A and b for sub
        A_2 = IndexUp(sub.sub.getA().todok())
        constr_2 = sub.sub.getConstrs()
        b_2 = {constr_2.index(c) + 1: c.RHS for c in constr_2}
        ub_2 = {t2: {y: self.T2[t2].Y[y].UB for y in self.T2[t2].Y} for t2 in self.T2}
        lb_2 = {t2: {y: self.T2[t2].Y[y].LB for y in self.T2[t2].Y} for t2 in self.T2}
        self.L1_length = len(constr_1)
        self.L2_length = len(constr_2)

        pi_limit = 1

        self.cglp = gp.Model('CGLP', env=env)
        self.L1 = self.cglp.addVars(range(1, self.L1_length + 1), name='L1')
        self.L2 = {t2: self.cglp.addVars(range(1, self.L2_length + 1), name='L2') for t2 in T2}
        self.Mu1 = self.cglp.addVars(master.X, lb=0, name='Mu1')
        self.Nu1 = self.cglp.addVars(master.X, lb=0, name='Nu1')
        self.Mu2 = {t2: self.cglp.addVars(sub.Y, lb=0, name='Mu2') for t2 in self.T2}
        self.Nu2 = {t2: self.cglp.addVars(sub.Y, lb=0, name='Nu2') for t2 in self.T2}
        self.PI0 = self.cglp.addVar(lb=-pi_limit, ub=pi_limit, name='P0')
        self.PI1 = self.cglp.addVars(sub.X, lb=-pi_limit, ub=pi_limit, name='P1')
        self.PI2 = self.cglp.addVars(sub.Y, lb=-pi_limit, ub=pi_limit, name='P2')

        for c in constr_1:
            if c.Sense == '=':
                self.L1[constr_1.index(c) + 1].LB = -100000
                # self.L1[constr_1.index(c) + 1].UB = 0
        for c in constr_2:
            if c.Sense == '=':
                for t2 in self.T2:
                    self.L2[t2][constr_2.index(c) + 1].LB = -100000
                    # self.L2[t2][constr_2.index(c) + 1].UB = 0

        self.bL2 = {t2: quicksum(b_2[key] * self.L2[t2][key] for key in b_2) for t2 in self.T2}
        self.MuLo2 = {t2: quicksum(lb_2[t2][key] * self.Mu2[t2][key] for key in sub.Y) for t2 in self.T2}
        self.NuUp2 = {t2: quicksum(ub_2[t2][key] * self.Nu2[t2][key] for key in sub.Y) for t2 in self.T2}


        self.MuLo1 = quicksum(lb_1[key] * self.Mu1[key] for key in sub.X)
        self.NuUp1 = quicksum(ub_1[key] * self.Nu1[key] for key in sub.X)

        self.TL2 = {t2: {x: 0 for x in sub.X} for t2 in self.T2}
        self.WL2 = {t2: {y: 0 for y in sub.Y} for t2 in self.T2}
        for t2 in T2:
            for key in A_2:
                if key[1] in sub.X:
                    self.TL2[t2][key[1]] += A_2[key] * self.L2[t2][key[0]]
                else:
                    self.WL2[t2][key[1]] += A_2[key] * self.L2[t2][key[0]]

        self.AL1 = {x: 0 for x in sub.X}
        self.bL1 = 0
        if A_1 is not None:
            for key in A_1:
                self.AL1[key[1]] += A_1[key] * self.L1[key[0]]
            self.bL1 = quicksum(b_1[key] * self.L1[key] for key in b_1)

        self.SetConstrs()
        self.cglp.setObjective(self.PI0 -
                              quicksum(self.PI1[x] * self.x_k[x] for x in sub.X) -
                              quicksum(self.PI2[y] * y_d[y] for y in sub.Y),
                              sense=GRB.MAXIMIZE)
        self.cglp.update()


    def UpdateModel(self, pi0, pi1, pi2, y_d):
        self.L2_length += 1
        for t2 in self.T2:
            self.L2[t2][self.L2_length] = self.cglp.addVar(name='L2')
            self.bL2[t2] += pi0 * self.L2[t2][self.L2_length]
            for pi1_key in pi1:
                self.TL2[t2][pi1_key] += pi1[pi1_key] * self.L2[t2][self.L2_length]
            for pi2_key in pi2:
                self.WL2[t2][pi2_key] += pi2[pi2_key] * self.L2[t2][self.L2_length]
        self.cglp.remove(self.cglp.getConstrs())
        self.SetConstrs()
        self.cglp.setObjective(self.PI0 -
                          quicksum(self.PI1[x] * self.x_k[x] for x in self.sub.X) -
                          quicksum(self.PI2[y] * y_d[y] for y in self.sub.Y), sense=GRB.MAXIMIZE)
        self.cglp.update()

    def SetConstrs(self):
        self.cglp.addConstrs((self.PI0 >= self.bL1 + self.bL2[t2] + self.MuLo2[t2] + self.NuUp2[t2] + self.MuLo1 + self.NuUp1
                        for t2 in self.T2), name='pi0')
        self.cglp.addConstrs((self.PI1[x] <= self.AL1[x] + self.TL2[t2][x] + self.Mu1[x] - self.Nu1[x]
                        for t2 in self.T2 for x in self.sub.X), name='pi1')
        self.cglp.addConstrs((self.PI2[y] <= self.WL2[t2][y] + self.Mu2[t2][y] - self.Nu2[t2][y]
                        for t2 in self.T2 for y in self.sub.Y), name='pi2')
        self.cglp.update()

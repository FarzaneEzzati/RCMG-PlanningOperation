# Dec 14, 2023. Repository was created to keep track of codings for DOE project.

import copy
import pickle
import random
import math
import numpy as np
import pandas as pd
import gurobipy
from gurobipy import quicksum, GRB
import time
env = gurobipy.Env()
env.setParam('OutputFlag', 0)


def geneCases():
    # Each element represents the scenarios for a month (1, ..., 12)
    with open('Data/OutageScenarios.pkl', 'rb') as handle:
        scens, probs = pickle.load(handle)
    handle.close()
    # Load profile of households
    lp1 = pd.read_csv('Data/Load_profile_1.csv')
    lp2 = pd.read_csv('Data/Load_profile_2.csv')
    # PV output for one unit (4kW)
    pv_profile = pd.read_csv('Data/PV_profiles.csv')
    return scens, probs, [lp1, lp2], pv_profile


# Global parameters
Budget = 100000
Years = 20
Interest_Rate = 0.08
Operational_Rate = 0.01
PA_Factor = ((1 + Interest_Rate) ** Years - 1) / (Interest_Rate * (1 + Interest_Rate) ** Years)
C = {1: 600, 2: 2780 / 4, 3: 150}
CO = {i: C[i] * PA_Factor + C[i] * Operational_Rate for i in (1, 2, 3)}
UB = [166, 80, 40]
LB = [20, 10, 2]
FuelPrice = 3.7
alpha, beta = 0.5, 0.2
GridPlus = 0.1497
GridMinus = alpha * GridPlus
LoadPrice = GridPlus
GenerPrice = beta * GridPlus
VoLL = np.array([2.1, 1.8, 1.4]) * GridPlus
PVSellPrice = (alpha + beta) * GridPlus
DGSellPrice = PVSellPrice
PVCurPrice = (alpha + beta) * GridPlus
DGCurPrice = (alpha + beta) * GridPlus
SOC_UB, SOC_LB = 0.9, 0.1
ES_gamma = 0.85
DG_gamma = 0.4
Eta_c = 0.8
Eta_i = 0.9

# Import data
Scens, Prob, Load, PV_Unit = geneCases()
Scens = Scens[0]

# Ranges need to be used
T = 168
SCount = len(Scens)
DVCCount = 3
MCount = 1
HCount = 2
OutageStart = 3 * 24 + 15
RNGDvc = range(1, DVCCount + 1)
RNGTime = range(1, T + 1)
RNGTimeMinus = range(1, T)
RNGMonth = range(1, MCount + 1)
RNGScen = range(1, SCount + 1)
RNGScenMinus = range(1, SCount)
RNGHouse = range(1, HCount + 1)


class SingleScenario:
    def __init__(self):
        # Define the load profiles and PV profiles
        L = {(h, t, g): Load[h - 1][f'Month {g}'].iloc[t - 1] for h in RNGHouse for t in RNGTime for g in RNGMonth}
        PV = {(t, g): PV_Unit[f'Month {g}'].iloc[t - 1] for t in RNGTime for g in RNGMonth}
        Out_Time = {(g, s): 0 for s in RNGScen for g in RNGMonth}
        for s in RNGScen:
            if Scens[s - 1] != 0:
                for g in RNGMonth:
                    Out_Time[(g, s)] = [OutageStart + i for i in range(int(Scens[s - 1]))]
        Pr = {s: Prob[s - 1] for s in RNGScen}

        # Generate the Model (with Lagrangian Dual)
        model = gurobipy.Model('LD-MIP', env=env)
        X_indices = [(s, d) for s in RNGScen for d in RNGDvc]  # 1: ES, 2: PV, 3: DG
        X = model.addVars(X_indices, vtype=GRB.INTEGER, name='X')

        # Bounds on X decisions
        model.addConstrs(X[(s, d)] <= UB[d - 1] for s in RNGScen for d in RNGDvc)
        model.addConstrs(X[(s, d)] >= LB[d - 1] for s in RNGScen for d in RNGDvc)

        # First stage constraint
        for s in RNGScen:
            model.addConstr(quicksum([X[(s, j)] * C[j] for j in RNGDvc]) <= Budget, name='Budget')

        # Second Stage Variables
        Y_tgs = [(t, g, s) for t in RNGTime for g in RNGMonth for s in RNGScen]
        Y_htgs = [(h, t, g, s) for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen]
        Y_tg = [(t, g) for t in RNGTime for g in RNGMonth]

        self.Y_tgs = Y_tgs
        self.Y_htgs = Y_htgs
        self.Y_tg = Y_tg

        Y_PVES = model.addVars(Y_tgs, name='Y_PVES')
        Y_DGES = model.addVars(Y_tgs, name='Y_DGES')
        Y_GridES = model.addVars(Y_tgs, name='Y_GridES')

        Y_PVL = model.addVars(Y_tgs, name='Y_PVL')
        Y_DGL = model.addVars(Y_tgs, name='Y_DGL')
        Y_ESL = model.addVars(Y_tgs, name='Y_ESL')
        Y_GridL = model.addVars(Y_tgs, name='Y_GridL')

        Y_LH = model.addVars(Y_htgs, name='Y_LH')  # Load served
        Y_LL = model.addVars(Y_htgs, name='Y_LL')  # Load lost
        Y_LT = model.addVars(Y_htgs, name='Y_LT')  # Load transferred

        Y_PVCur = model.addVars(Y_tgs, name='Y_PVCur')
        Y_DGCur = model.addVars(Y_tgs, name='Y_DGCur')

        Y_PVGrid = model.addVars(Y_tgs, name='Y_DGGrid')
        Y_DGGrid = model.addVars(Y_tgs, name='Y_DGGrid')
        Y_ESGrid = model.addVars(Y_tgs, name='Y_ESGrid')

        E = model.addVars(Y_tgs, name='E')

        u = model.addVars(Y_tgs, vtype=GRB.BINARY, name='u')

        # Lambda definition
        #lmda = [[0, 0, 0] for s in RNGScenMinus]

        # Second stage constraints
        # Energy storage level
        # model.addConstrs(E[(1, g, s)] == E[(T, g, s)] for s in RNGScen for g in RNGMonth)  REMOVE FOR NOW
        model.addConstrs(E[(1, g, s)] == SOC_UB * X[(s, 1)] for s in RNGScen for g in RNGMonth)
        model.addConstrs(SOC_LB * X[(s, 1)] <= E[(t, g, s)] for t in RNGTime for s in RNGScen for g in RNGMonth)
        model.addConstrs(E[(t, g, s)] <= SOC_UB * X[(s, 1)] for t in RNGTime for s in RNGScen for g in RNGMonth)

        # Balance of power flow
        model.addConstrs(E[(t + 1, g, s)] == E[(t, g, s)] +
                         ES_gamma * (Y_PVES[(t, g, s)] + Y_DGES[(t, g, s)] + Eta_c * Y_GridES[(t, g, s)]) -
                         Eta_i * (Y_ESL[(t, g, s)] + Y_ESGrid[(t, g, s)]) / ES_gamma
                         for t in RNGTimeMinus for s in RNGScen for g in RNGMonth)

        # Assigned load decomposition
        model.addConstrs(quicksum(L[(h, t, g)] for h in RNGHouse) >=
                         Eta_i * (Y_ESL[(t, g, s)] + Y_DGL[(t, g, s)] + Y_PVL[(t, g, s)]) +
                         Y_GridL[(t, g, s)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        # Load decomposition
        model.addConstrs(Y_LH[(h, t, g, s)] + Y_LL[(h, t, g, s)] + Y_LT[(h, t, g, s)] == L[(h, t, g)]
                         for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen)

        # PV power decomposition
        model.addConstrs(Y_PVL[(t, g, s)] + Y_PVES[(t, g, s)] + Y_PVCur[(t, g, s)] + Y_PVGrid[(t, g, s)] ==
                         PV[(t, g)] * X[(s, 2)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        # DG power decomposition
        model.addConstrs(Y_DGL[(t, g, s)] + Y_DGES[(t, g, s)] + Y_DGGrid[(t, g, s)] + Y_DGCur[(t, g, s)] == X[(s, 3)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        # ES charging/discharging constraints
        model.addConstrs(Y_ESL[(t, g, s)] + Y_ESGrid[(t, g, s)] <= UB[0] * u[(t, g, s)]
                         for t in RNGTime for s in RNGScen for g in RNGMonth)
        model.addConstrs(Y_PVES[(t, g, s)] + Y_GridES[(t, g, s)] + Y_DGES[(t, g, s)] <= UB[0] * (1 - u[(t, g, s)])
                         for t in RNGTime for s in RNGScen for g in RNGMonth)

        # Prohibited transaction with the grid during outage
        GridImport = quicksum(Y_GridL[(t, g, s)] + Y_GridES[(t, g, s)]
                            for t in RNGTime for s in RNGScen for g in RNGMonth)
        GridExport = quicksum(Y_PVGrid[(t, g, s)] + Y_ESGrid[(t, g, s)] + Y_DGGrid[(t, g, s)]
                             for t in RNGTime for s in RNGScen for g in RNGMonth)
        for s in RNGScen:
            for g in RNGMonth:
                if Out_Time[(g, s)] != 0:
                    model.addConstrs(GridImport == 0 for t in Out_Time[(g, s)])
                    model.addConstrs(GridExport == 0 for t in Out_Time[(g, s)])

        # Investment cost
        Cost1 = quicksum(Pr[s] * quicksum(X[(s, d)] * (CO[d])
                                            for d in RNGDvc)
                         for s in RNGScen)
        # Curtailment cost
        Cost2 = quicksum(Pr[s] * quicksum(PVCurPrice * (Y_PVCur[(t, g, s)] + Y_DGCur[(t, g, s)])
                                            for t in RNGTime for g in RNGMonth)
                         for s in RNGScen)
        # Losing load cost
        Cost3 = quicksum(Pr[s] * quicksum(VoLL[h - 1] * Y_LL[(h, t, g, s)]
                                          for h in RNGHouse for t in RNGTime for g in RNGMonth)
                         for s in RNGScen)
        # DG cost
        Cost4 = FuelPrice * DG_gamma * quicksum([Pr[s] * quicksum(Y_DGL[(t, g, s)] + Y_DGGrid[(t, g, s)] +
                                                                  Y_DGCur[(t, g, s)] + Y_DGES[(t, g, s)]
                                                                  for t in RNGTime for g in RNGMonth)
                                                 for s in RNGScen])
        # Import/Export cost
        Cost5 = quicksum(Pr[s] * quicksum(GridPlus * GridImport[(t, g, s)] -
                                            GridMinus * GridExport[(t, g, s)] -
                                            GenerPrice * X[(s, 2)] * PV[(t, g)] -
                                            quicksum(LoadPrice * Y_LH[(h, t, g, s)]
                                                      for h in RNGHouse)
                                            for t in RNGTime for g in RNGMonth)
                          for s in RNGScen)

        # Cost6 = quicksum(l[s - 1][d - 1] * (X[(s, d)] - X[(s + 1, d)]) for s in RNGScenMinus for d in RNGDvc) REMOVE FOR NOW
        model.setObjective(Cost1 + (365 / 7) * (Cost2 + Cost3 + Cost4 + Cost5), sense=GRB.MINIMIZE)
        model.update()
        self.model = model

    def Solve(self):
        self.model.optimize()
        return self.model.Status()



if __name__ == '__main__':
    m = SingleScenario()
    print(m.Solve())



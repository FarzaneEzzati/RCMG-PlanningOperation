import pickle
import time
import random
import numpy as np
import pandas as pd
import gurobipy
from gurobipy import quicksum, GRB
env = gurobipy.Env()
env.setParam('OutputFlag', 1)


def geneCases(consumer):
    # Each element represents the scenarios for a month (1, ..., 12)
    with open('../Data/OutageScenarios.pkl', 'rb') as handle:
        scens, probs = pickle.load(handle)
    handle.close()
    # Load profile of households
    lp1 = pd.read_csv('../Data/Load_profile_1.csv')
    lp2 = pd.read_csv('../Data/Load_profile_2.csv')
    # PV output for one unit (4kW)
    pv_profile = pd.read_csv('../Data/PV_profiles.csv')
    return scens, probs, [random.choice([lp1, lp2]) for _ in range(consumer)], pv_profile


# Counts
T = 90
DVCCount = 3
MCount = 12
HCount = 10
OutageStart = 16

# Import data
Scens, Probs, Load, PV_Unit = geneCases(consumer=HCount)

# Ranges
RNGDvc = range(1, DVCCount + 1)
RNGTime = range(1, T + 1)
RNGTimeMinus = range(1, T)
RNGMonth = range(1, MCount + 1)
RNGHouse = range(1, HCount + 1)
RNGScen = range(1, len(Scens) + 1)
RNGSta = (0, 1)  # 0: year before reinvestment, 1: year after reinvestment
DontTran = [24 * i + 16 + j for i in range(5) for j in range(5)]  # Each week from hour 16 to 20 transfer is prohibited.

# Global parameters
Budget1 = 250000
Budget2 = Budget1 / 2
Years = 20
Interest_Rate = 0.08
Operational_Rate = 0.01
PA_Factor = (Interest_Rate * (1 + Interest_Rate) ** Years) / ((1 + Interest_Rate) ** Years - 1)
C = {1: 600, 2: 2780, 3: 500}  # order is: [ES, PV, DG]
CO = {i: C[i] * (PA_Factor + Operational_Rate) for i in (1, 2, 3)}


UB0 = {1: 100, 2: 100, 3: 60}
LB0 = {1: 20, 2: 20, 3: 10}
UB1 = {1: 50, 2: 40, 3: 30}
LB1 = {1: 0, 2: 0, 3: 0}

alpha, beta, zeta = 0.5, 0.2, 1.95
# alpha: selling to grid coefficient, beta: renewable energy generation, zeta: selling to households coefficient
GridPlus = 0.1497  # $/kWh
GridMinus = alpha * GridPlus
GenerPrice = beta * GridPlus
LoadPrice = zeta * GridPlus

VoLL = {h: 1.5 * GridPlus for h in RNGHouse}
TransMax = 0.3  # %
TransPrice = TransMax * np.mean([VoLL[key] for key in VoLL.keys()])

ES_gamma = 0.85
DG_gamma = 0.4  # litr/kW
FuelPrice = 1.05  # $/litr
DGEffic = DG_gamma * FuelPrice  # Fuel cost of DG: $/kWh

GridSellPrice = alpha * GridPlus
PVSellPrice = GridSellPrice
DGSellPrice = GridSellPrice
PVCurPrice = GridSellPrice
DGCurPrice = GridSellPrice + DGEffic

SOC_UB, SOC_LB = 0.9, 0.1
Eta_c = 0.8
Eta_i = 0.9
GenPar = (365 / 7) / MCount

X_d = [(ii, d) for ii in RNGSta for d in RNGDvc]
Y_tg = [(t, g)
        for t in RNGTime for g in RNGMonth]
Y_htg = [(h, t, g)
         for h in RNGHouse for t in RNGTime for g in RNGMonth]
Y_ttg = [(t, to, g)
         for t in RNGTime for to in RNGTime for g in RNGMonth]


class TS_SPModel:

    def __init__(self):
        pass


    @staticmethod
    def BuildMasterProb():
        model = gurobipy.Model('MasterProb', env=env)
        tt = time.time()
        '''Investment & Reinvestment variables'''
        X = model.addVars(X_d, name='X')  # if 0, Integer defined, if 1, continuous defined

        for d in RNGDvc:
            # Bounds on X decisions
            model.addConstr(X[(0, d)] <= UB0[d], name=f'UpB0{d}')
            model.addConstr(X[(0, d)] >= LB0[d], name=f'LoB0{d}')
            model.addConstr(X[(1, d)] <= UB1[d], name=f'UpB1{d}')
            model.addConstr(X[(1, d)] >= LB1[d], name=f'LoB1{d}')

        # Investment constraint
        model.addConstr(quicksum(X[(0, d)] * C[d] for d in RNGDvc) <= Budget1, name='IB')
        # ReInvestment constraint
        model.addConstr(quicksum(X[(1, d)] * C[d] for d in RNGDvc) <= Budget2, name='RIB')

        # Formulate capital cost
        Capital = quicksum((X[(0, d)] + X[(1, d)]) * CO[d] for d in RNGDvc)


        ''' 
        we need to save the master problem and start creating subproblems separately.
        Note that Y variables should not be defined for the master problem because the benders' cut only have x vars in.
        '''
        eta = model.addVar(lb=-float('inf'), name='eta')
        model.setObjective(Capital + eta, sense=GRB.MINIMIZE)
        model.update()
        model.write('Models/MasterProb.mps')
        print(model)

    @staticmethod
    def BuildSubProb(X, s):
        # X must be a dictionary, not gurobi variable. X is 2by2.
        # X[0][j] refers to capacity device j before reinvestment.
        # X[1][j] refers to capacity device j added after reinvestment.
        name = f'sub-problem-{s}'
        model = gurobipy.Model(name)

        '''Scheduling variables'''
        Y_PVES = [model.addVars(Y_tg, name=f'Y_PVES{i}') for i in RNGSta]  # PV to ES
        Y_DGES = [model.addVars(Y_tg, name=f'Y_DGES{i}') for i in RNGSta]  # DE to ES
        Y_GridES = [model.addVars(Y_tg, name=f'Y_GridES{i}') for i in RNGSta]  # Grid to ES
        Y_PVL = [model.addVars(Y_tg, name=f'Y_PVL{i}') for i in RNGSta]  # Pv to L
        Y_DGL = [model.addVars(Y_tg, name=f'Y_DGL{i}') for i in RNGSta]  # Dg to L
        Y_ESL = [model.addVars(Y_tg, name=f'Y_ESL{i}') for i in RNGSta]  # ES to L
        Y_GridL = [model.addVars(Y_tg, name=f'Y_GridL{i}') for i in RNGSta]  # Grid to L
        Y_LH = [model.addVars(Y_htg, name=f'Y_LH{i}') for i in RNGSta]  # Load served
        Y_LL = [model.addVars(Y_htg, name=f'Y_LL{i}') for i in RNGSta]  # Load lost
        Y_LT = [model.addVars(Y_ttg, name=f'Y_LT{i}') for i in RNGSta]  # Load transferred
        Y_PVCur = [model.addVars(Y_tg, name=f'Y_PVCur{i}') for i in RNGSta]  # PV Curtailed
        Y_DGCur = [model.addVars(Y_tg, name=f'Y_DGCur{i}') for i in RNGSta]  # DG curtailed
        Y_PVGrid = [model.addVars(Y_tg, name=f'Y_DGGrid{i}') for i in RNGSta]  # PV to Grid
        Y_DGGrid = [model.addVars(Y_tg, name=f'Y_PVGrid{i}') for i in RNGSta]  # Dg to Grid
        Y_ESGrid = [model.addVars(Y_tg, name=f'Y_ESGrid{i}') for i in RNGSta]  # ES to Grid
        E = [model.addVars(Y_tg, name=f'E{i}') for i in RNGSta]  # ES level of energy
        U_E = [model.addVars(Y_tg, name=f'U_ES{i}') for i in RNGSta]  # Charge/discharge binary
        U_G = [model.addVars(Y_tg, name=f'U_G{i}') for i in RNGSta]  # Import/export binary

        '''Specify Load Demand, PV, Outage Duration for the scenario s'''
        # Define the load profiles and PV profiles
        L1 = {(h, t, g): Load[h - 1][f'Month {g}'].iloc[t - 1]
              for h in RNGHouse for t in RNGTime for g in RNGMonth}
        L2 = {(h, t, g): 1.3 * Load[h - 1][f'Month {g}'].iloc[t - 1]
              for h in RNGHouse for t in RNGTime for g in RNGMonth}
        L = [L1, L2]

        PV = {(t, g): PV_Unit[f'Month {g}'].iloc[t - 1]
              for t in RNGTime for g in RNGMonth}

        Out_Time = {g: 0 for g in RNGMonth}
        if Scens[s] != 0:
            for g in RNGMonth:
                Out_Time[g] = [OutageStart + j for j in range(int(Scens[s]))]

        '''Constraints'''
        for ii in RNGSta:  # RNGSta = (0, 1)
            for g in RNGMonth:
                # ES levels
                model.addConstr(E[ii][(1, g)] == SOC_UB * (X[(0, 1)] + ii * X[(1, 1)]))

                for t in RNGTime:
                    # Limits on energy level in ES
                    model.addConstr(SOC_LB * (X[(0, 1)] + ii * X[(1, 1)]) <= E[ii][(t, g)], name='')
                    model.addConstr(E[ii][(t, g)] <= SOC_UB * (X[(0, 1)] + ii * X[(1, 1)]), name='')

                    # Assigned load decomposition
                    model.addConstr(quicksum(Y_LH[ii][(h, t, g)] for h in RNGHouse) ==
                                    Eta_i * (Y_ESL[ii][(t, g)] + Y_DGL[ii][(t, g)] + Y_PVL[ii][(t, g)]) +
                                    Y_GridL[ii][(t, g)], name='')

                    for h in RNGHouse:
                        # Load decomposition
                        model.addConstr(Y_LH[ii][(h, t, g)] +
                                        Y_LL[ii][(h, t, g)] +
                                        quicksum(Y_LT[ii][(t, to, g)] for to in range(t, T + 1)) ==
                                        L[ii][(h, t, g)], name='')

                    if t in DontTran:
                        # Don't allow transfer
                        model.addConstrs(Y_LT[ii][(t, to, g)] == 0 for to in range(t, T + 1))
                    else:
                        # Max load transfer
                        model.addConstr(quicksum(Y_LT[ii][(t, to, g)] for to in range(t, T + 1)) <=
                                        TransMax * np.sum([L[ii][(h, t, g)] for h in RNGHouse]), name='')

                    # Load transfer and E level
                    model.addConstr(quicksum(Y_LT[ii][(to, t, g)] for to in range(1, t)) <= E[ii][(t, g)],  name='')

                    # Prohibited transfer to self
                    model.addConstr(Y_LT[ii][(t, t, g)] == 0, name='')

                    # PV power decomposition
                    model.addConstr(Y_PVL[ii][(t, g)] + Y_PVES[ii][(t, g)] +
                                    Y_PVCur[ii][(t, g)] + Y_PVGrid[ii][(t, g)] ==
                                    PV[(t, g)] * (X[(0, 2)] + ii * X[(1, 2)]), name='')

                    # DG power decomposition
                    model.addConstr(Y_DGL[ii][(t, g)] + Y_DGES[ii][(t, g)] +
                                    Y_DGGrid[ii][(t, g)] + Y_DGCur[ii][(t, g)] ==
                                    (X[(0, 3)] + ii * X[(1, 3)]), name='')

                    # ES charging/discharging constraints
                    model.addConstr(Y_ESL[ii][(t, g)] + Y_ESGrid[ii][(t, g)] <=
                                    (UB0[1] + ii * UB1[1]) * U_E[ii][(t, g)], name='')
                    model.addConstr(Y_PVES[ii][(t, g)] + Y_GridES[ii][(t, g)] + Y_DGES[ii][(t, g)] <=
                                    (UB0[1] + ii * UB1[1]) * (1 - U_E[ii][(t, g)]), name='')

                    model.addConstr(Y_ESGrid[ii][(t, g)] + Y_PVGrid[ii][(t, g)] + Y_DGGrid[ii][(t, g)] <=
                                    (UB0[1] + UB0[2] + UB0[3] + ii * (UB1[1] + UB1[2] + UB1[3])) * U_G[ii][
                                        (t, g)], name='')
                    model.addConstr(Y_GridES[ii][(t, g)] + Y_GridL[ii][(t, g)] <=
                                    (UB0[1] + UB0[2] + UB0[3] + ii * (UB1[1] + UB1[2] + UB1[3])) * (
                                                1 - U_G[ii][(t, g)]), name='')

                    # Prohibited transaction with the grid during outage
                    if Out_Time[g] != 0:
                        for ot in Out_Time[g]:
                            model.addConstr(Y_GridL[ii][(ot, g)] + Y_GridES[ii][(ot, g)] == 0, name='')
                            model.addConstr(Y_PVGrid[ii][(ot, g)] + Y_ESGrid[ii][(ot, g)] +
                                            Y_DGGrid[ii][(ot, g)] == 0, name='')

                for t in RNGTimeMinus:
                    # Balance of power flow
                    model.addConstr(E[ii][(t + 1, g)] ==
                                    E[ii][(t, g)] +
                                    ES_gamma * (Y_PVES[ii][(t, g)] + Y_DGES[ii][(t, g)] + Eta_c *
                                                Y_GridES[ii][(t, g)]) -
                                    Eta_i * (Y_ESL[ii][(t, g)] + Y_ESGrid[ii][(t, g)]) / ES_gamma, name='')

        '''Costs'''
        Costs = 0
        for ii in RNGSta:
            for g in RNGMonth:
                for t in RNGTime:
                    # Curtailment cost
                    Costs += PVCurPrice * (Y_PVCur[ii][(t, g)] + Y_DGCur[ii][(t, g)])
                    # Losing load cost
                    Costs += quicksum(VoLL[h] * Y_LL[ii][(h, t, g)] for h in RNGHouse)
                    # DG cost
                    Costs += DGEffic * (Y_DGL[ii][(t, g)] + Y_DGGrid[ii][(t, g)] +
                                       Y_DGCur[ii][(t, g)] + Y_DGES[ii][(t, g)])
                    # Import/Export cost

                    Costs += GridPlus * Y_GridES[ii][(t, g)] - \
                            GridMinus * (Y_PVGrid[ii][(t, g)] + Y_ESGrid[ii][(t, g)] + Y_DGGrid[ii][(t, g)]) - \
                            GenerPrice * X[(0, 2)] * PV[(t, g)] - \
                            LoadPrice * quicksum(Y_LH[ii][(h, t, g)] for h in RNGHouse)

                    # DRP cost
                    Costs += TransPrice * quicksum(Y_LT[ii][(to, t, g)] for to in RNGTime)

        total_cost = Probs[s] * GenPar * Costs
        model.setObjective(total_cost, sense=GRB.MINIMIZE)
        model.update()
        model.write(f'Models/{name}.mps')
        print(model)

    @staticmethod
    def Solve(self, model):
        print(model)
        model.optimize()


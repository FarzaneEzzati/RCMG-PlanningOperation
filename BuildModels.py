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
    with open('Data/OutageScenarios.pkl', 'rb') as handle:
        scens, probs = pickle.load(handle)
    handle.close()

    # Load profile of households
    lp1 = pd.read_csv('Data/Load_profile_1.csv')
    lp2 = pd.read_csv('Data/Load_profile_2.csv')

    # PV output for one unit (4kW)
    pv_profile = pd.read_csv('Data/PV_profiles.csv')
    return scens, probs, [random.choice([lp1, lp2]) for _ in range(consumer)], pv_profile


# Counts
T = 168
DVCCount = 3
MCount = 12
HCount = 30
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
RNGSta = (0, 1)

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
GridPlus = 0.1497 # $/kWh
GridMinus = alpha * GridPlus
GenerPrice = beta * GridPlus
LoadPrice = zeta * GridPlus

VoLL = np.multiply([1.3 for _ in RNGHouse], GridPlus)
TransMax = 0.3  # %
TransPrice = [v * TransMax for v in VoLL]

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

# Define the load profiles and PV profiles
L1 = {(h, t, g, s): Load[h - 1][f'Month {g}'].iloc[t - 1]
     for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen}

L2 = {(h, t, g, s): 1.3 * Load[h - 1][f'Month {g}'].iloc[t - 1]
for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen}

L = [L1, L2]
PV = {(t, g, s): PV_Unit[f'Month {g}'].iloc[t - 1]
      for t in RNGTime for g in RNGMonth for s in RNGScen}


class SingleScenario:
    def __init__(self, lmda, scen_indx, pr):
        Out_Time = {g: 0 for g in RNGMonth}
        if Scens[scen_indx] != 0:
            for g in RNGMonth:
                Out_Time[g] = [OutageStart + i for i in range(int(Scens[scen_indx]))]

        # Generate the Model (with Lagrangian Dual)
        model = gurobipy.Model('LD-MIP', env=env)
        X = model.addVars(RNGDvc, vtype=GRB.INTEGER, name='X')

        # Bounds on X decisions
        model.addConstrs(X[d] <= UB0[d] for d in RNGDvc)
        model.addConstrs(X[d] >= LB0[d] for d in RNGDvc)

        # First stage constraint
        model.addConstr(quicksum([X[d] * C[d] for d in RNGDvc]) <= Budget1, name='Budget')

        # Second Stage Variables
        Y_tgs = [(t, g) for t in RNGTime for g in RNGMonth]
        Y_htgs = [(h, t, g) for h in RNGHouse for t in RNGTime for g in RNGMonth]
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
        #Y_LT = model.addVars(Y_htgs, name='Y_LT')  # Load transferred

        Y_PVCur = model.addVars(Y_tgs, name='Y_PVCur')
        Y_DGCur = model.addVars(Y_tgs, name='Y_DGCur')

        Y_PVGrid = model.addVars(Y_tgs, name='Y_DGGrid')
        Y_DGGrid = model.addVars(Y_tgs, name='Y_DGGrid')
        Y_ESGrid = model.addVars(Y_tgs, name='Y_ESGrid')

        E = model.addVars(Y_tgs, name='E')

        u = model.addVars(Y_tgs, vtype=GRB.BINARY, name='u')

        # Second stage constraints
        # Energy storage level
        # model.addConstrs(E[(1, g)] == E[(T, g)]  for g in RNGMonth)  REMOVE FOR NOW
        model.addConstrs(E[(1, g)] == SOC_UB * X[1] for g in RNGMonth)
        model.addConstrs(SOC_LB * X[1] <= E[(t, g)] for t in RNGTime for g in RNGMonth)
        model.addConstrs(E[(t, g)] <= SOC_UB * X[1] for t in RNGTime for g in RNGMonth)

        # Balance of power flow
        model.addConstrs(E[(t + 1, g)] == E[(t, g)] +
                         ES_gamma * (Y_PVES[(t, g)] + Y_DGES[(t, g)] + Eta_c * Y_GridES[(t, g)]) -
                         Eta_i * (Y_ESL[(t, g)] + Y_ESGrid[(t, g)]) / ES_gamma
                         for t in RNGTimeMinus for g in RNGMonth)

        # Assigned load decomposition
        model.addConstrs(quicksum(L[(h, t, g)] for h in RNGHouse) >=
                         Eta_i * (Y_ESL[(t, g)] + Y_DGL[(t, g)] + Y_PVL[(t, g)]) +
                         Y_GridL[(t, g)]
                         for t in RNGTime for g in RNGMonth)

        # Load decomposition
        model.addConstrs(Y_LH[(h, t, g)] + Y_LL[(h, t, g)] == L[(h, t, g)]
                         for h in RNGHouse for t in RNGTime for g in RNGMonth)

        # PV power decomposition
        model.addConstrs(Y_PVL[(t, g)] + Y_PVES[(t, g)] + Y_PVCur[(t, g)] + Y_PVGrid[(t, g)] ==
                         PV[(t, g)] * X[2]
                         for t in RNGTime for g in RNGMonth)

        # DG power decomposition
        model.addConstrs(Y_DGL[(t, g)] + Y_DGES[(t, g)] + Y_DGGrid[(t, g)] + Y_DGCur[(t, g)] == X[3]
                         for t in RNGTime for g in RNGMonth)

        # ES charging/discharging constraints
        model.addConstrs(Y_ESL[(t, g)] + Y_ESGrid[(t, g)] <= UB0[1] * u[(t, g)]
                         for t in RNGTime for g in RNGMonth)
        model.addConstrs(Y_PVES[(t, g)] + Y_GridES[(t, g)] + Y_DGES[(t, g)] <= UB0[1] * (1 - u[(t, g)])
                         for t in RNGTime for g in RNGMonth)

        # Prohibited transaction with the grid during outage
        for g in RNGMonth:
            if Out_Time[g] != 0:
                model.addConstrs(Y_GridL[(t, g)] + Y_GridES[(t, g)] == 0 for t in Out_Time[g])
                model.addConstrs(Y_PVGrid[(t, g)] + Y_ESGrid[(t, g)] + Y_DGGrid[(t, g)] == 0 for t in Out_Time[g])

        # Defining grid import/export amount for cost evaluation
        GridImport = quicksum(Y_GridL[(t, g)] + Y_GridES[(t, g)]
                              for t in RNGTime for g in RNGMonth)
        GridExport = quicksum(Y_PVGrid[(t, g)] + Y_ESGrid[(t, g)] + Y_DGGrid[(t, g)]
                              for t in RNGTime for g in RNGMonth)

        # Investment cost
        Cost1 = pr * quicksum(X[d] * CO[d] for d in RNGDvc)
        # Curtailment cost
        Cost2 = pr * quicksum(PVCurPrice * (Y_PVCur[(t, g)] + Y_DGCur[(t, g)]) for t in RNGTime for g in RNGMonth)
        # Losing load cost
        Cost3 = pr * quicksum(VoLL[h - 1] * Y_LL[(h, t, g)] for h in RNGHouse for t in RNGTime for g in RNGMonth)
        # DG cost
        Cost4 = pr * FuelPrice * DG_gamma * quicksum(Y_DGL[(t, g)] + Y_DGGrid[(t, g)] + Y_DGCur[(t, g)] + Y_DGES[(t, g)]
                                                     for t in RNGTime for g in RNGMonth)
        # Import/Export cost
        Cost5 = pr * quicksum(GridPlus * GridImport -
                              GridMinus * GridExport -
                              GenerPrice * X[2] * PV[(t, g)] -
                              quicksum(LoadPrice * Y_LH[(h, t, g)]
                                       for h in RNGHouse)
                              for t in RNGTime for g in RNGMonth)
        Cost6 = quicksum(lmda[d - 1] * X[d] for d in RNGDvc)  # Note that lamda multiplier has dimension 1XDV, here 1X3

        self.primal_cost = Cost1 + (365 / 7) * (Cost2 + Cost3 + Cost4 + Cost5)
        self.dual_cost = Cost6
        model.setObjective(self.primal_cost + self.dual_cost, sense=GRB.MINIMIZE)
        model.update()
        self.model = model
        self.X = X

    def UpdateObj(self, l1):
        self.dual_cost = quicksum(l1[d - 1] * self.X[d] for d in RNGDvc)
        self.model.setObjective(self.primal_cost + self.dual_cost, sense=GRB.MINIMIZE)
        self.model.update()

    def Solve(self):
        self.model.optimize()
        return [self.X[1].x, self.X[2].x, self.X[3].x]


class RealScale:

    def __init__(self):
        Out_Time = {(g, s): 0 for s in RNGScen for g in RNGMonth}
        for s in RNGScen:
            if Scens[s] != 0:
                for g in RNGMonth:
                    Out_Time[(g, s)] = [OutageStart + j for j in range(int(Scens[s]))]

        model = gurobipy.Model('MIP', env=env)

        tt = time.time()
        '''Investment & Reinvestment variables'''
        X1 = model.addVars(RNGDvc, vtype=GRB.INTEGER, name='X1')
        X2 = model.addVars(RNGDvc, vtype=GRB.INTEGER, name='X2')

        for d in RNGDvc:
            # Bounds on X decisions
            model.addConstr(X1[d] <= UB0[d], name='UpB0')
            model.addConstr(X1[d] >= LB0[d], name='LoB0')
            model.addConstr(X2[d] <= UB1[d], name='UpB0')
            model.addConstr(X2[d] >= LB1[d], name='LoB0')
            # Investment constraint
        model.addConstr(quicksum(X1[d] * C[d] for d in RNGDvc) <= Budget1, name='IB')
        # ReInvestment constraint
        model.addConstr(quicksum(X2[d] * C[d] for d in RNGDvc) <= Budget2, name='RIB')

        '''Scheduling variables'''
        Y_tgs = [(t, g, s)
                 for t in RNGTime for g in RNGMonth for s in RNGScen]
        Y_htgs = [(h, t, g, s)
                  for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen]

        self.Y_tgs = Y_tgs
        self.Y_htgs = Y_htgs

        Y_PVES = [model.addVars(Y_tgs, name='Y_PVES') for _ in RNGSta]
        Y_DGES = [model.addVars(Y_tgs, name='Y_DGES') for _ in RNGSta]
        Y_GridES = [model.addVars(Y_tgs, name='Y_GridES') for _ in RNGSta]
        Y_PVL = [model.addVars(Y_tgs, name='Y_PVL') for _ in RNGSta]
        Y_DGL = [model.addVars(Y_tgs, name='Y_DGL') for _ in RNGSta]
        Y_ESL = [model.addVars(Y_tgs, name='Y_ESL') for _ in RNGSta]
        Y_GridL = [model.addVars(Y_tgs, name='Y_GridL') for _ in RNGSta]
        Y_LH = [model.addVars(Y_htgs, name='Y_LH') for _ in RNGSta]  # Load served
        Y_LL = [model.addVars(Y_htgs, name='Y_LL') for _ in RNGSta]  # Load lost
        #Y_LT = [model.addVars(Y_htgs, name='Y_LT') for _ in RNGSta]  # Load transferred
        Y_PVCur = [model.addVars(Y_tgs, name='Y_PVCur') for _ in RNGSta]
        Y_DGCur = [model.addVars(Y_tgs, name='Y_DGCur') for _ in RNGSta]
        Y_PVGrid = [model.addVars(Y_tgs, name='Y_DGGrid') for _ in RNGSta]
        Y_DGGrid = [model.addVars(Y_tgs, name='Y_DGGrid') for _ in RNGSta]
        Y_ESGrid = [model.addVars(Y_tgs, name='Y_ESGrid') for _ in RNGSta]
        E = [model.addVars(Y_tgs, name='E') for _ in RNGSta]
        U_E = [model.addVars(Y_tgs, vtype=GRB.BINARY, name='U_ES') for _ in RNGSta]
        U_G = [model.addVars(Y_tgs, vtype=GRB.BINARY, name='U_G') for _ in RNGSta]

        te = time.time() - tt
        tt = time.time()
        print(f'Build variables: {te}')

        '''Constraints'''
        for ii in RNGSta:  # RNGSta = (0, 1)
            for g in RNGMonth:
                for s in RNGScen:
                    # ES levels
                    model.addConstr(E[ii][(1, g, s)] == SOC_UB * (X1[1] + ii * X2[1]))


                    for t in RNGTime:
                        # Limits on energy level in ES
                        model.addConstr(SOC_LB * (X1[1] + ii * X2[1]) <= E[ii][(t, g, s)], name='')
                        model.addConstr(E[ii][(t, g, s)] <= SOC_UB * (X1[1] + ii * X2[1]), name='')

                        # Assigned load decomposition
                        model.addConstr(quicksum(Y_LH[ii][(h, t, g, s)] for h in RNGHouse) ==
                                         Eta_i * (Y_ESL[ii][(t, g, s)] + Y_DGL[ii][(t, g, s)] + Y_PVL[ii][(t, g, s)]) +
                                         Y_GridL[ii][(t, g, s)], name='')


                        for h in RNGHouse:
                            # Load decomposition
                            model.addConstr(Y_LH[ii][(h, t, g, s)] + Y_LL[ii][(h, t, g, s)]  ==
                                            L[ii][(h, t, g, s)], name='')

                            # Max load transfer
                            #model.addConstr(Y_LT[ii][(h, t, g, s)] <= TransMax * L[ii][(h, t, g, s)], name='')

                        # PV power decomposition
                        model.addConstr(Y_PVL[ii][(t, g, s)] + Y_PVES[ii][(t, g, s)] +
                                         Y_PVCur[ii][(t, g, s)] + Y_PVGrid[ii][(t, g, s)] ==
                                         PV[(t, g, s)] * (X1[2] + ii * X2[2]), name='')

                        # DG power decomposition
                        model.addConstr(Y_DGL[ii][(t, g, s)] + Y_DGES[ii][(t, g, s)] +
                                         Y_DGGrid[ii][(t, g, s)] + Y_DGCur[ii][(t, g, s)] ==
                                         (X1[3] + ii * X2[3]), name='')


                        # ES charging/discharging constraints
                        model.addConstr(Y_ESL[ii][(t, g, s)] + Y_ESGrid[ii][(t, g, s)] <=
                                            (UB0[1] + ii * UB1[1]) * U_E[ii][(t, g, s)], name='')
                        model.addConstr(Y_PVES[ii][(t, g, s)] + Y_GridES[ii][(t, g, s)] + Y_DGES[ii][(t, g, s)] <=
                                            (UB0[1] + ii * UB1[1]) * (1 - U_E[ii][(t, g, s)]), name='')

                        model.addConstr(Y_ESGrid[ii][(t, g, s)] + Y_PVGrid[ii][(t, g, s)] + Y_DGGrid[ii][(t, g, s)] <=
                        (UB0[1] + UB0[2] + UB0[3]+ ii * (UB1[1] + UB1[2] + UB1[3])) * U_G[ii][(t, g, s)], name='')
                        model.addConstr(Y_GridES[ii][(t, g, s)] + Y_GridL[ii][(t, g, s)] <=
                        (UB0[1] + UB0[2] + UB0[3]+ ii * (UB1[1] + UB1[2] + UB1[3])) * (1 - U_G[ii][(t, g, s)]), name='')


                        # Prohibited transaction with the grid during outage
                        if Out_Time[(g, s)] != 0:
                            for ot in Out_Time[(g, s)]:
                                model.addConstr(Y_GridL[ii][(ot, g, s)] + Y_GridES[ii][(ot, g, s)] == 0, name='')
                                model.addConstr(Y_PVGrid[ii][(ot, g, s)] + Y_ESGrid[ii][(ot, g, s)] +
                                             Y_DGGrid[ii][(ot, g, s)] == 0, name='')

                    for t in RNGTimeMinus:
                        # Balance of power flow
                        model.addConstr(E[ii][(t + 1, g, s)] ==
                                         E[ii][(t, g, s)] +
                                         ES_gamma * (Y_PVES[ii][(t, g, s)] + Y_DGES[ii][(t, g, s)] + Eta_c * Y_GridES[ii][(t, g, s)]) -
                                         Eta_i * (Y_ESL[ii][(t, g, s)] + Y_ESGrid[ii][(t, g, s)]) / ES_gamma, name='')

        te = time.time() - tt
        tt = time.time()
        print(f'Build constraints: {te}')

        '''Costs'''
        OprCosts = 0
        # Investment cost
        Capital = quicksum((X1[d] + X2[d]) * CO[d] for d in RNGDvc)
        for ii in RNGSta:
            for s in RNGScen:
                cost = 0
                for g in RNGMonth:
                    for t in RNGTime:
                        # Curtailment cost
                        cost += PVCurPrice * (Y_PVCur[ii][(t, g, s)] + Y_DGCur[ii][(t, g, s)])
                        # Losing load cost
                        cost += quicksum(VoLL[h - 1] * Y_LL[ii][(h, t, g, s)] for h in RNGHouse)
                        # DG cost
                        cost += DGEffic * (Y_DGL[ii][(t, g, s)] + Y_DGGrid[ii][(t, g, s)] +
                                                Y_DGCur[ii][(t, g, s)] + Y_DGES[ii][(t, g, s)])
                        # Import/Export cost

                        cost += GridPlus * Y_GridES[ii][(t, g, s)] - \
                                GridMinus * (Y_PVGrid[ii][(t, g, s)] + Y_ESGrid[ii][(t, g, s)] + Y_DGGrid[ii][(t, g, s)]) - \
                                GenerPrice * X1[2] * PV[(t, g, s)] - \
                                LoadPrice * quicksum(Y_LH[ii][(h, t, g, s)] for h in RNGHouse)

                        # DRP cost
                        #cost += quicksum(Y_LT[ii][(h, t, g, s)] * TransPrice[h - 1] for h in RNGHouse)
                OprCosts += Probs[s] * cost

        te = time.time() - tt
        tt = time.time()
        print(f'Build costs: {te}')

        total_cost = Capital + GenPar * OprCosts
        model.setObjective(total_cost, sense=GRB.MINIMIZE)
        model.update()
        self.model = model
        self.X1 = X1
        self.X2 = X2
        self.Y_LL1 = Y_LL[0]


    def Solve(self):
        print(self.model)
        print('Started Optimizing')
        self.model.optimize()

        lost1 =[self.Y_LL1[(h, t, g, s)].x for t in RNGTime for g in RNGMonth for s in RNGScen for h in RNGHouse]
        print(f'Total lost {np.sum(lost1)}')
        print(f'Load: {np.sum([L[0][(h, t, g, s)] for t in RNGTime for g in RNGMonth for s in RNGScen for h in RNGHouse])}')
        return [[self.X1[1].x, self.X1[2].x, self.X1[3].x], [self.X2[1].x, self.X2[2].x, self.X2[3].x]]

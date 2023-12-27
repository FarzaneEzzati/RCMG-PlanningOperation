import pickle
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
T = 168
DVCCount = 3
MCount = 12
HCount = 37
OutageStart = 16

# Import data
Scens, Probs, Load, PV_Unit = geneCases(consumer=HCount)

# Ranges
RNGDvc = range(1, DVCCount + 1)
RNGTime = range(1, T + 1)
RNGTimeMinus = range(1, T)
RNGMonth = range(1, MCount + 1)
RNGHouse = range(1, HCount + 1)
RNGScen = range(1, len(Scens)+1)
RNGSta = (1, 2)

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
VoLL = np.multiply([random.choice([1.8, 2.3, 1.2, 1.6, 2.5, 3, 2.2]) for _ in RNGHouse], GridPlus)
TransMax = 0.3
TransPrice = [v * TransMax for v in VoLL]
PVSellPrice = (alpha + beta) * GridPlus
DGSellPrice = PVSellPrice
PVCurPrice = (alpha + beta) * GridPlus
DGCurPrice = (alpha + beta) * GridPlus
SOC_UB, SOC_LB = 0.9, 0.1
ES_gamma = 0.85
DG_gamma = 0.4
Eta_c = 0.8
Eta_i = 0.9



# Define the load profiles and PV profiles
L = {(h, t, g, s, i): Load[h - 1][f'Month {g}'].iloc[t - 1]
     for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta}
PV = {(t, g, s, i): PV_Unit[f'Month {g}'].iloc[t - 1]
      for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta}


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
        model.addConstrs(X[d] <= UB[d - 1] for d in RNGDvc)
        model.addConstrs(X[d] >= LB[d - 1] for d in RNGDvc)

        # First stage constraint
        model.addConstr(quicksum([X[d] * C[d] for d in RNGDvc]) <= Budget, name='Budget')

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
        Y_LT = model.addVars(Y_htgs, name='Y_LT')  # Load transferred

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
        model.addConstrs(Y_LH[(h, t, g)] + Y_LL[(h, t, g)] + Y_LT[(h, t, g)] == L[(h, t, g)]
                         for h in RNGHouse for t in RNGTime for g in RNGMonth)

        # PV power decomposition
        model.addConstrs(Y_PVL[(t, g)] + Y_PVES[(t, g)] + Y_PVCur[(t, g)] + Y_PVGrid[(t, g)] ==
                         PV[(t, g)] * X[2]
                         for t in RNGTime for g in RNGMonth)

        # DG power decomposition
        model.addConstrs(Y_DGL[(t, g)] + Y_DGES[(t, g)] + Y_DGGrid[(t, g)] + Y_DGCur[(t, g)] == X[3]
                         for t in RNGTime for g in RNGMonth)

        # ES charging/discharging constraints
        model.addConstrs(Y_ESL[(t, g)] + Y_ESGrid[(t, g)] <= UB[0] * u[(t, g)]
                         for t in RNGTime for g in RNGMonth)
        model.addConstrs(Y_PVES[(t, g)] + Y_GridES[(t, g)] + Y_DGES[(t, g)] <= UB[0] * (1 - u[(t, g)])
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
        Cost3 = pr * quicksum(VoLL[h - 1] * Y_LL[(h, t, g)]for h in RNGHouse for t in RNGTime for g in RNGMonth)
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
        Cost6 = quicksum(lmda[d-1] * X[d] for d in RNGDvc)  # Note that lamda multiplier has dimension 1XDV, here 1X3

        self.primal_cost = Cost1 + (365 / 7) * (Cost2 + Cost3 + Cost4 + Cost5)
        self.dual_cost = Cost6
        model.setObjective(self.primal_cost + self.dual_cost, sense=GRB.MINIMIZE)
        model.update()
        self.model = model
        self.X = X

    def UpdateObj(self, l1):
        self.dual_cost = quicksum(l1[d-1] * self.X[d] for d in RNGDvc)
        self.model.setObjective(self.primal_cost + self.dual_cost, sense=GRB.MINIMIZE)
        self.model.update()

    def Solve(self):
        self.model.optimize()
        return [self.X[1].x, self.X[2].x, self.X[3].x]


class RealScale:

    def __init__(self):
        Out_Time = {(g, s, i): 0 for s in RNGScen for g in RNGMonth for i in RNGSta}
        for i in RNGSta:
            for s in RNGScen:
                if Scens[s] != 0:
                    for g in RNGMonth:
                        Out_Time[(g, s, i)] = [OutageStart + j for j in range(int(Scens[s]))]

        model = gurobipy.Model('MIP', env=env)

        '''Investment variables'''
        X1 = model.addVars(RNGDvc, vtype=GRB.INTEGER, name='X1')
        # Bounds on X decisions
        model.addConstrs(X1[d] <= UB[d - 1] for d in RNGDvc)
        model.addConstrs(X1[d] >= LB[d - 1] for d in RNGDvc)
        # First stage constraint
        model.addConstr(quicksum([X1[d] * C[d] for d in RNGDvc]) <= Budget, name='Budget')

        '''Reinvestment variables'''
        X2 = model.addVars(RNGDvc, vtype=GRB.INTEGER, name='X2')
        # Bounds on X decisions
        model.addConstrs(X2[d] <= UB[d - 1]/2 for d in RNGDvc)
        model.addConstrs(X2[d] >= LB[d - 1]/2 for d in RNGDvc)
        # First stage constraint
        model.addConstr(quicksum([X2[d] * C[d] for d in RNGDvc]) <= Budget/2, name='Budget')

        '''Scheduling variables'''
        Y_tgs = [(t, g, s, i)
                 for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta]
        Y_htgs = [(h, t, g, s, i)
                  for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta]

        self.Y_tgs = Y_tgs
        self.Y_htgs = Y_htgs

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

        U_ES = model.addVars(Y_tgs, vtype=GRB.BINARY, name='U_ES')

        '''Constraints'''
        # ES levels
        model.addConstrs(E[(1, g, s, 1)] == SOC_UB * X1[1] for g in RNGMonth for s in RNGScen)
        model.addConstrs(SOC_LB * X1[1] <= E[(t, g, s, 1)] for t in RNGTime for g in RNGMonth for s in RNGScen)
        model.addConstrs(E[(t, g, s, 1)] <= SOC_UB * X1[1] for t in RNGTime for g in RNGMonth for s in RNGScen)

        model.addConstrs(E[(1, g, s, 2)] == SOC_UB * (X1[1] + X2[1]) for g in RNGMonth for s in RNGScen)
        model.addConstrs(SOC_LB * (X1[1] + X2[1]) <= E[(t, g, s, 2)] for t in RNGTime for g in RNGMonth for s in RNGScen)
        model.addConstrs(E[(t, g, s, 2)] <= SOC_UB * (X1[1] + X2[1]) for t in RNGTime for g in RNGMonth for s in RNGScen)

        # Balance of power flow
        model.addConstrs(E[(t + 1, g, s, i)] == E[(t, g, s, i)] +
                         ES_gamma * (Y_PVES[(t, g, s, i)] + Y_DGES[(t, g, s, i)] + Eta_c * Y_GridES[(t, g, s, i)]) -
                         Eta_i * (Y_ESL[(t, g, s, i)] + Y_ESGrid[(t, g, s, i)]) / ES_gamma
                         for t in RNGTimeMinus for g in RNGMonth for s in RNGScen for i in RNGSta)

        # Assigned load decomposition
        model.addConstrs(quicksum(L[(h, t, g, s, i)] for h in RNGHouse) >=
                         Eta_i * (Y_ESL[(t, g, s, i)] + Y_DGL[(t, g, s, i)] + Y_PVL[(t, g, s, i)]) +
                         Y_GridL[(t, g, s, i)]
                         for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta)

        # Load decomposition
        model.addConstrs(Y_LH[(h, t, g, s, i)] + Y_LL[(h, t, g, s, i)] + Y_LT[(h, t, g, s, i)] == L[(h, t, g, s, i)]
                         for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta)

        # PV power decomposition
        model.addConstrs(Y_PVL[(t, g, s, i)] + Y_PVES[(t, g, s, i)] + Y_PVCur[(t, g, s, i)] + Y_PVGrid[(t, g, s, i)] ==
                         PV[(t, g, s, i)] * (X1[2] + (i-1)*X2[2])
                         for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta)

        # DG power decomposition
        model.addConstrs(Y_DGL[(t, g, s, i)] + Y_DGES[(t, g, s, i)] + Y_DGGrid[(t, g, s, i)] + Y_DGCur[(t, g, s, i)] ==
                         (X1[3] + (i-1)*X2[3])
                         for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta)

        # ES charging/discharging constraints
        model.addConstrs(Y_ESL[(t, g, s, i)] + Y_ESGrid[(t, g, s, i)] <= UB[0] * U_ES[(t, g, s, i)]
                         for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta)
        model.addConstrs(Y_PVES[(t, g, s, i)] + Y_GridES[(t, g, s, i)] + Y_DGES[(t, g, s, i)] <= UB[0] * (1 - U_ES[(t, g, s, i)])
                         for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta)


        # Prohibited transaction with the grid during outage
        for i in RNGSta:
            for s in RNGScen:
                for g in RNGMonth:
                    if Out_Time[(g, s, i)] != 0:
                        model.addConstrs(Y_GridL[(t, g, s, i)] + Y_GridES[(t, g, s, i)] == 0
                                         for t in Out_Time[(g, s, i)] for s in RNGScen)
                        model.addConstrs(Y_PVGrid[(t, g, s, i)] + Y_ESGrid[(t, g, s, i)] + Y_DGGrid[(t, g, s, i)] == 0
                                         for t in Out_Time[(g, s, i)] for s in RNGScen)

        # Defining grid import/export amount for cost evaluation
        GridImport = quicksum(Y_GridL[(t, g, s, i)] + Y_GridES[(t, g, s, i)]
                            for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta)
        GridExport = quicksum(Y_PVGrid[(t, g, s, i)] + Y_ESGrid[(t, g, s, i)] + Y_DGGrid[(t, g, s, i)]
                             for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta)

        '''Costs'''
        Costs = []
        # Investment cost
        Capital = quicksum((X1[d] + X2[d]) * CO[d] for d in RNGDvc)
        # Curtailment cost
        Costs.append(quicksum(Probs[s] * PVCurPrice * (Y_PVCur[(t, g, s, i)] + Y_DGCur[(t, g, s, i)])
                         for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta))
        # Losing load cost
        Costs.append(quicksum(Probs[s] * VoLL[h - 1] * Y_LL[(h, t, g, s, i)]
                         for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta))
        # DG cost
        Costs.append(FuelPrice * DG_gamma * quicksum(Probs[s] * (Y_DGL[(t, g, s, i)] + Y_DGGrid[(t, g, s, i)] +
                                                     Y_DGCur[(t, g, s, i)] + Y_DGES[(t, g, s, i)])
                                                     for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta))
        # Import/Export cost
        Costs.append(quicksum(Probs[s] * (GridPlus * GridImport -
                                     GridMinus * GridExport -
                                     GenerPrice * X1[2] * PV[(t, g, s, i)] -
                                     quicksum(LoadPrice * Y_LH[(h, t, g, s, i)]
                                              for h in RNGHouse))
                         for t in RNGTime for g in RNGMonth for s in RNGScen for i in RNGSta))

        # DRP cost
        Costs.append(quicksum(Probs[s] * quicksum(Y_LT[(h, t, g, s, i)] * TransPrice[h-1]
                                             for h in RNGHouse for t in RNGTime for g in RNGMonth for i in RNGSta)
                         for s in RNGScen))

        primal_cost = Capital + (365 / 7) * quicksum(Costs)
        model.setObjective(primal_cost, sense=GRB.MINIMIZE)
        model.update()
        self.model = model
        self.X1 = X1
        self.X2 = X2

    def Solve(self):
        print('Started Optimizing')
        self.model.optimize()
        return [[self.X1[1].x, self.X1[2].x, self.X1[3].x], [self.X2[1].x, self.X2[2].x, self.X2[3].x]]

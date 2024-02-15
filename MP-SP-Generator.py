import pickle
import random
import copy
import pandas as pd
import numpy as np
from static_functions import DictMin, IndexUp, XInt
import gurobipy as gp
from gurobipy import quicksum, GRB
env = gp.Env()
env.setParam('OutputFlag', 0)



def geneCases():
    # Each element represents the scenarios for a month (1, ..., 12)
    with open('Data/OutageScenarios.pkl', 'rb') as handle:
        scens, probs = pickle.load(handle)
    handle.close()
    # Load profile of households
    lp = pd.read_csv('Data/Load_profile_1.csv')
    # PV output for one unit (4kW)
    pv_profile = pd.read_csv('Data/PV_profiles.csv')
    return scens, probs, lp, pv_profile


# Counts
T = 4    # count of hours per week
LCount = 1    # count of locations
DVCCount = 3    # count of devices
MCount = 1   # count of months
HCount = 1    # count of households
OutageStart = 1  # hour that outage starts

# Import data
Scens, Probs, Load, PV_Unit = geneCases()

# Ranges
RNGLoc = range(1, LCount + 1)
RNGDvc = range(1, DVCCount + 1)
RNGTime = range(1, T + 1)
RNGTimeMinus = range(1, T)
RNGMonth = range(1, MCount + 1)
RNGHouse = range(1, HCount + 1)
RNGScen = range(1, len(Scens) + 1)
RNGSta = (1, 2)  # 0: year before reinvestment, 1: year after reinvestment
DontTran = [24 * i + 16 + j for i in range(5) for j in range(5)]  # Each week from hour 16 to 20 transfer is prohibited.


# Global parameters
Budget1 = 250000
Budget2 = Budget1 / 2
Years = 20
Interest_Rate = 0.08
Operational_Rate = 0.01
PA_Factor = (Interest_Rate * (1 + Interest_Rate) ** Years) / ((1 + Interest_Rate) ** Years - 1)
C = {1: 110, 2: 100, 3: 20}  # order is: [ES, PV, DG]
CO = {i: C[i] * (PA_Factor + Operational_Rate) for i in (1, 2, 3)}

UB1 = {1: 10000, 2: 10000.8, 3: 4000}
LB1 = {1: 10, 2: 10, 3: 2}
UB2 = {1: 50.1, 2: 40.1, 3: 30.1}
LB2 = {1: 0, 2: 0, 3: 0}

alpha, beta, zeta = 0.5, 0.2, 0.9
# alpha: selling to grid coefficient, beta: renewable energy generation, zeta: selling to households coefficient
GridPlus = 0.1497  # $/kWh
GridMinus = alpha * GridPlus
GenerPrice = beta * GridPlus
LoadPrice = zeta * GridPlus

VoLL = 2.5 * GridPlus
TransMax = 0.3  # %
TransPrice = TransMax * VoLL

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

X_ild = [(ii, l, d) for ii in RNGSta for l in RNGLoc for d in RNGDvc]
X_id = [(ii, d) for ii in RNGSta for d in RNGDvc]
Y_itg = [(ii, t, g)
        for ii in RNGSta for t in RNGTime for g in RNGMonth]
Y_ittg = [(ii, t, to, g)
         for ii in RNGSta for t in RNGTime for to in RNGTime for g in RNGMonth]

Y_itgs = [(ii, t, g, s)
        for ii in RNGSta for t in RNGTime for g in RNGMonth for s in RNGScen]
Y_ittgs = [(ii, t, to, g, s)
         for ii in RNGSta for t in RNGTime for to in RNGTime for g in RNGMonth for s in RNGScen]


eta_M = -10000000


def MasterProb():
        master = gp.Model('MasterProb', env=env)
        '''Investment & Reinvestment variables'''
        X = {}
        for l in RNGLoc:
            for d in RNGDvc:
                X[(1, l, d)] = master.addVar(lb=LB1[d], ub=UB1[d], name=f'X[1,{l},{d}]')
        for l in RNGLoc:
            for d in RNGDvc:
                X[(2, l, d)] = master.addVar(lb=LB2[d], ub=UB2[d], name=f'X[2,{l},{d}]')
        master.update()
        X_keys = range(1, len(X_ild)+1)

        # Investment constraint
        master.addConstrs(-quicksum(X[(1, l, d)] * C[d] for d in RNGDvc) >= -Budget1 for l in RNGLoc)
        # ReInvestment constraint
        master.addConstrs(-quicksum(X[(2, l, d)] * C[d] for d in RNGDvc) >= -Budget2 for l in RNGLoc)
        # Capacity limit
        master.addConstrs(-quicksum(X[(1, l, d)] for l in RNGLoc) >= -UB1[d] for d in RNGDvc)
        master.addConstrs(-quicksum(X[(2, l, d)] for l in RNGLoc) >= -UB2[d] for d in RNGDvc)
        master.addConstrs(quicksum(X[(1, l, d)] for l in RNGLoc) >= LB1[d] for d in RNGDvc)
        master.addConstrs(quicksum(X[(2, l, d)] for l in RNGLoc) >= LB2[d] for d in RNGDvc)

        # Assign Upper and Lower bounds to X
        for ild in X_ild:
            if ild[0] == 1:
                X[ild].UB = UB1[ild[2]]
                X[ild].LB = LB1[ild[2]]
            else:
                X[ild].UB = UB2[ild[2]]
                X[ild].LB = LB2[ild[2]]

        Capital = 0
        for l in RNGLoc:
            Capital += quicksum((X[(1, l, d)] + X[(2, l, d)]) * CO[d] for d in RNGDvc)
        eta = master.addVar(lb=eta_M, name='eta')
        master.setObjective(Capital + eta, sense=GRB.MINIMIZE)

        # Save master in mps + save data in pickle
        master.update()
        master.write('Models/Master.mps')
        return X_keys

def SubProb(scen):
        sub = gp.Model(env=env)
        '''X vars to keep the model accessible for TMatrix'''
        if True:
            X_fixed = sub.addVars(X_ild, name='X')
            X_keys = range(1, len(X_ild)+1)  # starts from 1
        '''Scheduling variables'''
        if True:
            U_E = sub.addVars(Y_itg, lb=0, ub=1, name=f'U_E')  # Charge/discharge binary
            U_G = sub.addVars(Y_itg, lb=0, ub=1, name=f'U_G')  # Import/export binary
            Y_PVES = sub.addVars(Y_itg, name=f'Y_PVES')  # PV to ES (1 before reinvestment, 2 after)
            Y_DGES = sub.addVars(Y_itg, name=f'Y_DGES')  # DE to ES
            Y_GridES = sub.addVars(Y_itg, name=f'Y_GridES')   # Grid to ES
            Y_PVL = sub.addVars(Y_itg, name=f'Y_PV')  # Pv to L
            Y_DGL = sub.addVars(Y_itg, name=f'Y_DGL')  # Dg to L
            Y_ESL = sub.addVars(Y_itg, name=f'Y_ESL')  # ES to L
            Y_GridL = sub.addVars(Y_itg, name=f'Y_GridL')  # Grid to L
            Y_PVCur = sub.addVars(Y_itg, name=f'Y_PVCur')  # PV Curtailed
            Y_DGCur = sub.addVars(Y_itg, name=f'Y_DGCur')  # DG curtailed
            Y_PVGrid = sub.addVars(Y_itg, name=f'Y_DGGrid')  # PV to Grid
            Y_DGGrid = sub.addVars(Y_itg, name=f'Y_PVGrid')  # Dg to Grid
            Y_ESGrid = sub.addVars(Y_itg, name=f'Y_ESGrid')  # ES to Grid
            Y_E = sub.addVars(Y_itg, name=f'Y_E')  # ES level of energy
            Y_LH = sub.addVars(Y_itg, name=f'Y_LH')  # Load served
            Y_LL = sub.addVars(Y_itg, name=f'Y_LL')  # Load lost
            Y_LT = sub.addVars(Y_ittg, name=f'Y_LT')  # Load transferred

            '''Saving all vars keys in W and T matrix'''
            sub.update()
            Y_keys = range(X_keys[-1] + 1, len(sub.getVars()) + 1)

            '''Obtain the index for integer variables in the second stage'''
            Y_intk = []
            for var in sub.getVars():
                if ('U_E' in var.VarName, 'U_G' in var.VarName) != (False, False):
                    Y_intk.append(var.index + 1)

        '''Specify Load Demand, PV, Outage Duration for the scenario s'''
        if True:
            # Define the load profiles and PV profiles
            L = {(ii, t, g):  (1 + 0.05 * scen) * 12 * 10 * Load[f'Month {g}'].iloc[t - 1]
                  for ii in RNGSta for t in RNGTime for g in RNGMonth}

            PV = {(t, g): (0.5 + 0.05 * scen) * PV_Unit[f'Month {g}'].iloc[t - 1] for t in RNGTime for g in RNGMonth}

            Out_Time = {g: 0 for g in RNGMonth}
            if Scens[scen] != 0:
                for g in RNGMonth:
                    Out_Time[g] = [OutageStart + j for j in range(int(Scens[scen]))]
        '''Scheduling constraints'''
        for ii in RNGSta:  # RNGSta = (1, 2)
            for g in RNGMonth:
                # ES levels
                sub.addConstr(Y_E[(ii, 1, g)] == SOC_LB * quicksum(X_fixed[(1, l, 1)] + (ii - 1) * X_fixed[(2, l, 1)] for l in RNGLoc), name='t1')

                for t in RNGTime:
                    # Limits on energy level in ES
                    sub.addConstr(Y_E[(ii, t, g)] >= SOC_LB * quicksum(X_fixed[(1, l, 1)] + (ii - 1) * X_fixed[(2, l, 1)] for l in RNGLoc), name='E_LB')


                    sub.addConstr(-Y_E[(ii, t, g)] >= -SOC_UB * quicksum(X_fixed[(1, l, 1)] + (ii - 1) * X_fixed[(2, l, 1)] for l in RNGLoc), name='E_UB')

                    # PV power decomposition
                    sub.addConstr((Y_PVL[(ii, t, g)] + Y_PVES[(ii, t, g)] +
                                        Y_PVCur[(ii, t, g)] + Y_PVGrid[(ii, t, g)]) ==
                                        PV[(t, g)] * quicksum(X_fixed[(1, l, 2)] + (ii - 1) * X_fixed[(2, l, 2)] for l in RNGLoc), name='PV')

                    # DG power decomposition
                    sub.addConstr(Y_DGL[(ii, t, g)] + Y_DGES[(ii, t, g)] +
                                    Y_DGGrid[(ii, t, g)] + Y_DGCur[(ii, t, g)] ==
                                    quicksum(X_fixed[(1, l, 3)] + (ii - 1) * X_fixed[(2, l, 3)] for l in RNGLoc), name='DG')


                    # Assigned load decomposition
                    sub.addConstr(Y_LH[(ii, t, g)] ==
                                    Eta_i * (Y_ESL[(ii, t, g)] + Y_DGL[(ii, t, g)] + Y_PVL[(ii, t, g)]) +
                                    Y_GridL[(ii, t, g)], name='LoadH')

                    for h in RNGHouse:
                        # Load decomposition
                        sub.addConstr(Y_LH[(ii, t, g)] + Y_LL[(ii, t, g)] +
                                        quicksum(Y_LT[(ii, t, to, g)] for to in range(t, T + 1)) ==
                                        L[(ii, t, g)], name='LoadD')

                    if t in DontTran:
                        # Don't allow transfer
                        sub.addConstrs((Y_LT[(ii, t, to, g)] == 0 for to in range(t, T + 1)), name='NoTrans')
                    else:
                        # Max load transfer
                        sub.addConstr(TransMax * L[(ii, t, g)] -
                                           quicksum(Y_LT[(ii, t, to, g)] for to in range(t, T + 1)) >= 0, name='MaxLoadTrans')

                    # Load transfer and E level
                    sub.addConstr(Y_E[(ii, t, g)] - quicksum(Y_LT[(ii, to, t, g)] for to in range(1, t)) >= 0,  name='ES Load limit')

                    # Prohibited transfer to self
                    sub.addConstr(Y_LT[(ii, t, t, g)] == 0, name='TransIt')

                    # ES charging/discharging constraints
                    sub.addConstr((UB1[1] + (ii - 1) * UB2[1]) * U_E[(ii, t, g)] -
                                       (Y_ESL[(ii, t, g)] + Y_ESGrid[(ii, t, g)]) >= 0, name='Discharge')
                    sub.addConstr((UB1[1] + (ii - 1) * UB2[1]) * (1 - U_E[(ii, t, g)]) -
                                       (Y_PVES[(ii, t, g)] + Y_GridES[(ii, t, g)] + Y_DGES[(ii, t, g)]) >= 0, name='Charge')

                    sub.addConstr(1000000 * U_G[(ii, t, g)] -
                                       (Y_ESGrid[(ii, t, g)] + Y_PVGrid[(ii, t, g)] + Y_DGGrid[(ii, t, g)]) >= 0,
                                       name='Grid+')
                    sub.addConstr(1000000 * (1 - U_G[(ii, t, g)]) -
                                       (Y_GridES[(ii, t, g)] + Y_GridL[(ii, t, g)]) >= 0, name='Grid-')

                    # Prohibited transaction with the grid during outage
                    if Out_Time[g] != 0:
                        for ot in Out_Time[g]:
                            sub.addConstr(Y_GridL[(ii, ot, g)] + Y_GridES[(ii, ot, g)] == 0, name='')
                            sub.addConstr(Y_PVGrid[(ii, ot, g)] + Y_ESGrid[(ii, ot, g)] +
                                            Y_DGGrid[(ii, ot, g)] == 0, name='Outage')
                            sub.addConstr(U_G[(ii, ot, g)] == 0)

                for t in range(1, T-1):
                    # Balance of power flow
                    sub.addConstr(Y_E[(ii, t + 1, g)] ==
                                       Y_E[(ii, t, g)] +
                                        ES_gamma * (Y_PVES[(ii, t, g)] + Y_DGES[(ii, t, g)] + Eta_c * Y_GridES[(ii, t, g)]) -
                                        (Eta_i / ES_gamma) * (Y_ESL[(ii, t, g)] + Y_ESGrid[(ii, t, g)]), name='Balance')
        '''Assigning Bounds'''
        if True:
            for itg in Y_itg:
                Y_ESL[itg].UB = SOC_UB*(UB1[1] + (itg[0] - 1) * UB2[1])
                Y_ESGrid[itg].UB = SOC_UB * (UB1[1] + (itg[0] - 1) * UB2[1])
                Y_PVL[itg].UB = UB1[2] + (itg[0] - 1) * UB2[2]
                Y_PVGrid[itg].UB = UB1[2] + (itg[0] - 1) * UB2[2]
                Y_PVES[itg].UB = UB1[2] + (itg[0] - 1) * UB2[2]
                Y_PVCur[itg].UB = UB1[2] + (itg[0] - 1) * UB2[2]
                Y_DGGrid[itg].UB = UB1[3] + (itg[0] - 1) * UB2[3]
                Y_DGL[itg].UB = UB1[3] + (itg[0] - 1) * UB2[3]
                Y_DGES[itg].UB = UB1[3] + (itg[0] - 1) * UB2[3]
                Y_DGCur[itg].UB = UB1[3] + (itg[0] - 1) * UB2[3]
                Y_GridES[itg].UB = SOC_UB*(UB1[1] + (itg[0] - 1) * UB2[1])
                Y_GridL[itg].UB = L[(itg[0], itg[1], itg[2])]
                Y_LL[itg].UB = L[(itg[0], itg[1], itg[2])]
            for ittg in Y_ittg:
                Y_LT[ittg].UB = L[(ittg[0], ittg[1], ittg[3])]

        '''Costs'''
        if True:
            Costs = 0
            for ii in RNGSta:
                for g in RNGMonth:
                    for t in RNGTime:
                        # Curtailment cost
                        Costs += PVCurPrice * (Y_PVCur[(ii, t, g)] + Y_DGCur[(ii, t, g)])
                        # Losing load cost
                        Costs += VoLL * Y_LL[(ii, t, g)]
                        # DG cost
                        Costs += DGEffic * (Y_DGL[(ii, t, g)] + Y_DGGrid[(ii, t, g)] +
                                           Y_DGCur[(ii, t, g)] + Y_DGES[(ii, t, g)])
                        # Import/Export cost

                        Costs += GridPlus * Y_GridES[(ii, t, g)] - \
                                GridMinus * (Y_PVGrid[(ii, t, g)] + Y_ESGrid[(ii, t, g)] + Y_DGGrid[(ii, t, g)]) - \
                                GenerPrice * quicksum(X_fixed[(ii, l, 2)] for l in RNGLoc) * PV[(t, g)] - \
                                LoadPrice * (Y_ESL[(ii, t, g)] + Y_DGL[(ii, t, g)] + Y_PVL[(ii, t, g)])

                        # DRP cost
                        Costs += TransPrice * quicksum(Y_LT[(ii, to, t, g)] for to in RNGTime)

        total_cost = GenPar * Costs
        sub.setObjective(total_cost, sense=GRB.MINIMIZE)
        sub.update()
        sub.write(f'Models/Sub{scen}.mps')
        return Y_keys, Y_intk

class DetModel:
    def __init__(self):
        real = gp.Model(env=env)
        X = {}
        for l in RNGLoc:
            for d in RNGDvc:
                X[(1, l, d)] = real.addVar(vtype=GRB.INTEGER, lb=LB1[d], ub=UB1[d], name=f'X[1,{l},{d}]')
        for l in RNGLoc:
            for d in RNGDvc:
                X[(2, l, d)] = real.addVar(vtype=GRB.INTEGER, lb=LB2[d], ub=UB2[d], name=f'X[2,{l},{d}]')

        Capital = 0
        for l in RNGLoc:
            # Investment constraint
            real.addConstr(quicksum(X[(1, l, d)] * C[d] for d in RNGDvc) <= Budget1, name='IB')
            # ReInvestment constraint
            real.addConstr(quicksum(X[(2, l, d)] * C[d] for d in RNGDvc) <= Budget2, name='RIB')

            # Formulate capital cost
            Capital += quicksum((X[(1, l, d)] + X[(2, l, d)]) * CO[d] for d in RNGDvc)

        '''Scheduling variables'''
        if True:
            U_E = real.addVars(Y_itgs, vtype=GRB.BINARY, name=f'U_E')  # Charge/discharge binary
            U_G = real.addVars(Y_itgs, vtype=GRB.BINARY, name=f'U_G')  # Import/export binary
            Y_PVES = real.addVars(Y_itgs, name=f'Y_PVES')  # PV to ES (1 before reinvestment, 2 after)
            Y_DGES = real.addVars(Y_itgs, name=f'Y_DGES')  # DE to ES
            Y_GridES = real.addVars(Y_itgs, name=f'Y_GridES')  # Grid to ES
            Y_PVL = real.addVars(Y_itgs, name=f'Y_PV')  # Pv to L
            Y_DGL = real.addVars(Y_itgs, name=f'Y_DGL')  # Dg to L
            Y_ESL = real.addVars(Y_itgs, name=f'Y_ESL')  # ES to L
            Y_GridL = real.addVars(Y_itgs, name=f'Y_GridL')  # Grid to L
            Y_PVCur = real.addVars(Y_itgs, name=f'Y_PVCur')  # PV Curtailed
            Y_DGCur = real.addVars(Y_itgs, name=f'Y_DGCur')  # DG curtailed
            Y_PVGrid = real.addVars(Y_itgs, name=f'Y_DGGrid')  # PV to Grid
            Y_DGGrid = real.addVars(Y_itgs, name=f'Y_PVGrid')  # Dg to Grid
            Y_ESGrid = real.addVars(Y_itgs, name=f'Y_ESGrid')  # ES to Grid
            Y_E = real.addVars(Y_itgs, name=f'Y_E')  # ES level of energy
            Y_LH = real.addVars(Y_itgs, name=f'Y_LH')  # Load served
            Y_LL = real.addVars(Y_itgs, name=f'Y_LL')  # Load lost
            Y_LT = real.addVars(Y_ittgs, name=f'Y_LT')  # Load transferred
        '''Specify Load Demand, PV, Outage Duration for the scenario s'''
        if True:
            # Define the load profiles and PV profiles
            # Define the load profiles and PV profiles
            # Define the load profiles and PV profiles
            L = {(ii, t, g, s): (1 + 0.05 * s) * 20 * Load[f'Month {g}'].iloc[t - 1]
                  for ii in RNGSta for t in RNGTime for g in RNGMonth for s in RNGScen}

            PV = {(t, g, s): (1 + 0.05 * s) * PV_Unit[f'Month {g}'].iloc[t - 1]
                  for t in RNGTime for g in RNGMonth for s in RNGScen}
            Out_Time = {(g, s): 0 for s in RNGScen for g in RNGMonth}
            for s in RNGScen:
                if Scens[s] != 0:
                    for g in RNGMonth:
                        Out_Time[(g, s)] = [OutageStart + j for j in range(int(Scens[s]))]
        '''Scheduling constraints'''
        for s in RNGScen:
            for ii in RNGSta:  # RNGSta = (1, 2)
                for g in RNGMonth:
                    # ES levels
                    real.addConstr(Y_E[(ii, 1, g, s)] == SOC_UB * quicksum(
                        X[(1, l, 1)] + (ii - 1) * X[(2, l, 1)] for l in RNGLoc))
                    # Only use the summation of ES capacity before/after reinvestment for rhs update in ABC BB-D

                    for t in RNGTime:
                        # Limits on energy level in ES
                        real.addConstr(Y_E[(ii, t, g, s)] >= SOC_LB * quicksum(
                            X[(1, l, 1)] + (ii - 1) * X[(2, l, 1)] for l in RNGLoc))

                        real.addConstr(-Y_E[(ii, t, g, s)] >= -SOC_UB * quicksum(
                            X[(1, l, 1)] + (ii - 1) * X[(2, l, 1)] for l in RNGLoc))

                        # PV power decomposition
                        real.addConstr((Y_PVL[(ii, t, g, s)] + Y_PVES[(ii, t, g, s)] +
                                            Y_PVCur[(ii, t, g, s)] + Y_PVGrid[(ii, t, g, s)]) ==
                                           PV[(t, g, s)] * quicksum(
                            X[(1, l, 2)] + (ii - 1) * X[(2, l, 2)] for l in RNGLoc))

                        # DG power decomposition
                        real.addConstr(Y_DGL[(ii, t, g, s)] + Y_DGES[(ii, t, g, s)] +
                                           Y_DGGrid[(ii, t, g, s)] + Y_DGCur[(ii, t, g, s)] ==
                                           quicksum(X[(1, l, 3)] + (ii - 1) * X[(2, l, 3)] for l in RNGLoc))

                        # Assigned load decomposition
                        real.addConstr(Y_LH[(ii, t, g, s)] ==
                                           Eta_i * (Y_ESL[(ii, t, g, s)] + Y_DGL[(ii, t, g, s)] + Y_PVL[(ii, t, g, s)]) +
                                           Y_GridL[(ii, t, g, s)], name='')


                        real.addConstr(Y_LH[(ii, t, g, s)] + Y_LL[(ii, t, g, s)] +
                                           quicksum(Y_LT[(ii, t, to, g, s)] for to in range(t, T + 1)) ==
                                           L[(ii, t, g, s)], name='')

                        if t in DontTran:
                            # Don't allow transfer
                            real.addConstrs(Y_LT[(ii, t, to, g, s)] == 0 for to in range(t, T + 1))
                        else:
                            # Max load transfer
                            real.addConstr(TransMax * L[(ii, t, g, s)] -
                                               quicksum(Y_LT[(ii, t, to, g, s)] for to in range(t, T + 1)) >= 0, name='')

                        # Load transfer and E level
                        real.addConstr(Y_E[(ii, t, g, s)] - quicksum(Y_LT[(ii, to, t, g, s)] for to in range(1, t)) >= 0, name='')

                        # Prohibited transfer to self
                        real.addConstr(Y_LT[(ii, t, t, g, s)] == 0, name='')

                        # ES charging/discharging constraints
                        real.addConstr((UB1[1] + (ii - 1) * UB2[1]) * U_E[(ii, t, g, s)] -
                                           (Y_ESL[(ii, t, g, s)] + Y_ESGrid[(ii, t, g, s)]) >= 0, name='')
                        real.addConstr((UB1[1] + (ii - 1) * UB2[1]) * (1 - U_E[(ii, t, g, s)]) -
                                           (Y_PVES[(ii, t, g, s)] + Y_GridES[(ii, t, g, s)] + Y_DGES[(ii, t, g, s)]) >= 0, name='')

                        real.addConstr(
                            (UB1[1] + UB1[2] + UB1[3] + (ii - 1) * (UB2[1] + UB2[2] + UB2[3])) * U_G[(ii, t, g, s)] -
                            (Y_ESGrid[(ii, t, g, s)] + Y_PVGrid[(ii, t, g, s)] + Y_DGGrid[(ii, t, g, s)]) >= 0,
                            name='')
                        real.addConstr(
                            (UB1[1] + UB1[2] + UB1[3] + (ii - 1) * (UB2[1] + UB2[2] + UB2[3])) * (1 - U_G[(ii, t, g, s)]) -
                            (Y_GridES[(ii, t, g, s)] + Y_GridL[(ii, t, g, s)]) >= 0, name='')

                        # Prohibited transaction with the grid during outage
                        if Out_Time[(g, s)] != 0:
                            for ot in Out_Time[(g, s)]:
                                real.addConstr(Y_GridL[(ii, ot, g, s)] + Y_GridES[(ii, ot, g, s)] == 0, name='')
                                real.addConstr(Y_PVGrid[(ii, ot, g, s)] + Y_ESGrid[(ii, ot, g, s)] +
                                                   Y_DGGrid[(ii, ot, g, s)] == 0, name='')
                                real.addConstrs(U_G[(ii, ot, g, s)] == 0)

                    for t in RNGTimeMinus:
                        # Balance of power flow
                        real.addConstr(Y_E[(ii, t + 1, g, s)] ==
                                           Y_E[(ii, t, g, s)] +
                                           ES_gamma * (Y_PVES[(ii, t, g, s)] + Y_DGES[(ii, t, g, s)] + Eta_c * Y_GridES[(ii, t, g, s)]) -
                                           (Eta_i / ES_gamma) * (Y_ESL[(ii, t, g, s)] + Y_ESGrid[(ii, t, g, s)]), name='')
        '''Costs'''
        if True:
            Total_cost = 0
            for s in RNGScen:
                Costs = 0
                for ii in RNGSta:
                    for g in RNGMonth:
                        for t in RNGTime:
                            # Curtailment cost
                            Costs += PVCurPrice * (Y_PVCur[(ii, t, g, s)] + Y_DGCur[(ii, t, g, s)])
                            # Losing load cost
                            Costs += VoLL * Y_LL[(ii, t, g, s)]
                            # DG cost
                            Costs += DGEffic * (Y_DGL[(ii, t, g, s)] + Y_DGGrid[(ii, t, g, s)] +
                                                Y_DGCur[(ii, t, g, s)] + Y_DGES[(ii, t, g, s)])
                            # Import/Export cost

                            Costs += GridPlus * Y_GridES[(ii, t, g, s)] - \
                                     GridMinus * (Y_PVGrid[(ii, t, g, s)] + Y_ESGrid[(ii, t, g, s)] + Y_DGGrid[(ii, t, g, s)]) - \
                                     GenerPrice * quicksum(X[(1, l, 2)] for l in RNGLoc) * PV[(t, g, s)] - \
                                     LoadPrice * (Y_ESL[(ii, t, g, s)] + Y_DGL[(ii, t, g, s)] + Y_PVL[(ii, t, g, s)])

                            # DRP cost
                            Costs += TransPrice * quicksum(Y_LT[(ii, to, t, g, s)] for to in RNGTime)

                Total_cost += Probs[s] * GenPar * Costs
            real.setObjective(Total_cost + Capital, sense=GRB.MINIMIZE)

        '''Save model data'''
        if True:
            real.write(f'Models/real.mps')
            print(real)


if __name__ == '__main__':
    X_keys = MasterProb()
    for scen in Probs.keys():
        if scen < 3:
            Y_keys, Y_intk = SubProb(scen)
        else:
            break
    with open(f'Models/Indices.pkl', 'wb') as f:
        pickle.dump([X_keys, Y_keys, Y_intk], f)
    f.close()
    # real = DetModel()




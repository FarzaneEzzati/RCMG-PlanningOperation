import pickle
import random
import copy
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import quicksum, GRB
import tqdm 
import warnings
warnings.filterwarnings('ignore')

env = gp.Env()
env.setParam('OutputFlag', 0)

# Preparing Power Outage Scenarios
if True:
    OT = pd.read_csv('Scenarios/Outage/Outage Scenarios - reduced.csv')
    Outage_probs = {i: OT['Probability'].iloc[i-1] for i in range(1, 31)}
    Outage_scens = {i: int(OT['Gamma Scenario'].iloc[i-1]) for i in range(1, 31)}

# Preparing PV scenarios
if True:
    PV_scens = pd.read_csv('Data/PV_profiles.csv')
    PV_probs = {i: 1/30 for i in range(1, 31)}

# Preparing Load Profiles
if True:
    Load_scens = {i: {j: [] for j in range(1, 13)} for i in range(1, 31)}
    Load_probs = {i: 1 for i in range(1, 31)}
    month_counter = 1
    for m in ['JanFeb', 'MarApr', 'MayJun', 'JulAug', 'SepOct', 'NovDec']:
        dw = pd.read_csv(f'Scenarios/Load Demand/HarrisCounty/LoadScenarios-{m}-w.csv')
        de = pd.read_csv(f'Scenarios/Load Demand/HarrisCounty/LoadScenarios-{m}-e.csv')
        cols = dw.columns[2:99]

        for i in range(1, 31):
            Load_probs[i] = Load_probs[i] * (dw['probs'].iloc[i-1] * de['probs'].iloc[i-1])

            dw_single = dw[cols].iloc[i-1]
            de_single = de[cols].iloc[i - 1]
            week_long = pd.concat([dw_single, dw_single, dw_single, dw_single, dw_single, de_single, de_single])
            week_long_hourly = []

            for j in range(1, 169):
                loc = (j-1)*4
                summ = sum(week_long[loc + k] for k in range(4))
                week_long_hourly.append(summ)
            Load_scens[i][month_counter] = week_long_hourly
            Load_scens[i][month_counter+1] = week_long_hourly
        month_counter += 2

# Preparing Load Growth Scenarios
if True:
    AG = pd.read_csv('Scenarios/Load Demand/annual growth scenarios.csv')
    AG_probs = {i: AG['Probability'].iloc[i-1] for i in range(1, 31)}
    AG_scens = {i: AG['Lognorm Scenario'].iloc[i-1] for i in range(1, 31)}

all_probs = {i: Outage_probs[i]*PV_probs[i]*Load_probs[i]*AG_probs[i] for i in Outage_probs}
norm_probs = {i: all_probs[i]/sum(all_probs.values()) for i in Outage_probs}

with open('Data/ScenarioProbabilities.pkl', 'wb') as handle:
    pickle.dump(norm_probs, handle)
handle.close()

# Counts
T = 168  # count of hours per week
LCount = 10  # count of locations
DVCCount = 3  # count of devices
MCount = 12  # count of months
OutageStart = 16  # hour that outage starts


# Ranges
RNGLoc = range(1, LCount + 1)
RNGDvc = range(1, DVCCount + 1)
RNGTime = range(1, T + 1)
RNGTimeMinus = range(1, T)
RNGMonth = range(1, MCount + 1)
RNGScen = list(all_probs.keys())
RNGSta = (1, 2)  # 0: year before reinvestment, 1: year after reinvestment
DontTran = [24 * i + 13 + j for i in range(6) for j in range(7)]  # Each week from hour 1 to 19 pm transfer is prohibited.


# Global parameters
Budget1 = 1000000
Budget2 = Budget1 / 2
Years = 20
ReInvsYear = 10
Operational_Rate = 0.01
PA_Factor = ((1 + Operational_Rate) ** Years - 1) / (Operational_Rate * (1 + Operational_Rate) ** Years)
Labor_Factor = 0.12
C = {1: (1+Labor_Factor)* 151, 2: (1+Labor_Factor)*2780, 3: (1+Labor_Factor)*400}  # order is: [ES, PV, DG]
CO = {i: C[i] * (1 + Operational_Rate * PA_Factor) for i in (1, 2, 3)}
LocationPrice = {i: 2000 for i in RNGLoc}
F = {(i, j): LocationPrice[j] for j in LocationPrice for i in RNGSta}

UB = {(1, 1): 1000, (1, 2): 1000, (1, 3): 400,
      (2, 1): 500, (2, 2): 500, (2, 3): 30}
LB = {(1, 1): 0, (1, 2): 0, (1, 3): 0,
      (2, 1): 0, (2, 2): 0, (2, 3): 0}

ES_gamma = 0.85
DG_gamma = 0.4  # gal/kW
FuelPrice = 3.61  # $/gal
DGEffic = DG_gamma * FuelPrice  # Fuel cost of DG: $/kWh
ES_d = 0.02

zeta = 0.8
GridPlus = 0.1812  # $/kWh
GridMinus = 0.1407
LoadPrice = zeta * GridPlus
PVCurPrice = GridMinus
DGCurPrice = GridMinus + DGEffic

VoLL_sensitivity = 10
VoLL = 1.59 * VoLL_sensitivity * GridPlus 
VoLL_hourly = []
TransMax = 0.3  # %
TransPrice = VoLL * TransMax
for t in RNGTime:
    if t in DontTran:
        VoLL_hourly.append(VoLL)
    else:
        VoLL_hourly.append((1-TransMax) * VoLL)

SOC_UB, SOC_LB = 0.9, 0.1
eta_i = 0.9
GenPar = (365 / 7) / MCount

X_ild = [(i, l, d) for i in RNGSta for l in RNGLoc for d in RNGDvc]
X_il = [(i, l) for i in RNGSta for l in RNGLoc]
Y_itg = [(i, t, g)
         for i in RNGSta for t in RNGTime for g in RNGMonth]
Y_ittg = [(i, t, to, g)
          for i in RNGSta for t in RNGTime for to in RNGTime for g in RNGMonth]

Y_itgs = [(i, t, g, s)
          for i in RNGSta for t in RNGTime for g in RNGMonth for s in RNGScen]
Y_ittgs = [(i, t, to, g, s)
           for i in RNGSta for t in RNGTime for to in RNGTime for g in RNGMonth for s in RNGScen]
Y_1tg = [(1, t, g) for t in RNGTime for g in RNGMonth]
Y_2tg = [(2, t, g) for t in RNGTime for g in RNGMonth]
eta_M = -10000000


def MasterProb():
    master = gp.Model('MasterProb', env=env)
    '''Investment'''
    X = master.addVars(X_ild, vtype=GRB.INTEGER, name='X')
    U = master.addVars(X_il, lb=0, ub=1, vtype=GRB.BINARY, name='U')
    master.update()
    # Investment constraint
    master.addConstr(-quicksum(F[(1, l)] * U[(1, l)] for l in RNGLoc) -
                     quicksum(X[(1, l, d)] * C[d] for l in RNGLoc for d in RNGDvc) >= -Budget1,
                     name='Investment')
    # ReInvestment constraint
    master.addConstr(-quicksum(F[(2, l)] * (U[(2, l)] - U[(1, l)]) for l in RNGLoc) -
                     quicksum(X[(2, l, d)] * C[d] for l in RNGLoc for d in RNGDvc) >= -Budget2,
                     name='Reinvestment')
    # Capacity limit
    master.addConstrs((-quicksum(X[(i, l, d)] for l in RNGLoc) >= -UB[(i, d)]
                      for i in RNGSta for d in RNGDvc), name='Total Capacity Limit')
    master.addConstrs((quicksum(X[(i, l, d)] for l in RNGLoc) >= LB[(i, d)]
                      for i in RNGSta for d in RNGDvc), name='Total Capacity Limit')

    # Allowing capacity
    # Capacity limit
    master.addConstrs((-X[(i, l, d)] >= -UB[(i, d)] * U[(i, l)]
                      for i in RNGSta for l in RNGLoc for d in RNGDvc), name='Capacity Limit')
    master.addConstrs((X[(i, l, d)] >= LB[(i, d)] * U[(i, l)]
                      for i in RNGSta for l in RNGLoc for d in RNGDvc), name='Capacity Limit')

    # Location constraints
    master.addConstrs((U[(2, l)] >= U[(1, l)] for l in RNGLoc), name='Location')

    Capital = quicksum(F[(1, l)] * U[(1, l)] + X[(1, l, d)] * CO[d]
        for l in RNGLoc for d in RNGDvc) + \
              quicksum(F[(2, l)] * (U[(2, l)] - U[(1, l)]) + X[(2, l, d)] * CO[d]
        for l in RNGLoc for d in RNGDvc)

    eta = master.addVar(lb=eta_M, name='eta')
    master.setObjective(Capital + eta, sense=GRB.MINIMIZE)

    # Save master in mps + save data in pickle
    master.update()
    master.write('Models/Master.mps')
    with open('Models/Master_X_Info.pkl', 'wb') as handle:
        pickle.dump(range(len(X_ild)), handle)
    handle.close()



def SubProb(scen):
    sub = gp.Model(env=env)
    if True:
        X = sub.addVars(X_ild, name=f'X')
        Y_PVES = sub.addVars(Y_itg, name=f'Y_PVES')  # PV to ES (1 before reinvestment, 2 after)
        Y_DGES = sub.addVars(Y_itg, name=f'Y_DGES')  # DE to ES
        Y_GridES = sub.addVars(Y_itg, name=f'Y_GridES')  # Grid to ES
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
        Y_LL = sub.addVars(Y_itg, name=f'Y_LL')  # Load lost
        Y_LT = sub.addVars(Y_ittg, name=f'Y_LT')  # Load transferred

    '''Specify Load Demand, PV, Outage Duration for the scenario s'''
    if True:
        # Define the load profiles and PV profiles
        L = {(i, t, g): (1 + (i-1) * AG_scens[scen]/100) * Load_scens[scen][g][t - 1]
             for i in RNGSta for t in RNGTime for g in RNGMonth}

        PV = {(t, g): PV_scens[f'Month {g}'].iloc[t - 1] for t in RNGTime for g in RNGMonth}

        Out_Time = {g: 0 for g in RNGMonth}
        if Outage_scens[scen] != 0:
            if Outage_scens[scen] >= 168-OutageStart:
                for g in RNGMonth:
                    Out_Time[g] = [OutageStart + j for j in range(168-OutageStart+1)]
            else:
                for g in RNGMonth:
                    Out_Time[g] = [OutageStart + j for j in range(int(Outage_scens[scen]))]
    

    '''Scheduling constraints'''
    for i in RNGSta:  # RNGSta = (1, 2)
        for g in RNGMonth:
            # ES levels
            sub.addConstr(Y_E[(i, 1, g)] == SOC_LB * sum((1 - (i - 1) * ES_d) ** ReInvsYear * X[(1, l, 1)] + (i - 1) * X[(2, l, 1)] for l in RNGLoc),
                          name='t1')

            for t in RNGTime:
                # Limits on energy level in ES
                sub.addConstr(
                    Y_E[(i, t, g)] >= SOC_LB * quicksum((1 - (i - 1) * ES_d) ** ReInvsYear * X[(1, l, 1)] + (i - 1) * X[(2, l, 1)] for l in RNGLoc),
                    name='E_LB')

                sub.addConstr(
                    -Y_E[(i, t, g)] >= -SOC_UB * quicksum((1 - (i - 1) * ES_d) ** ReInvsYear * X[(1, l, 1)] + (i - 1) * X[(2, l, 1)] for l in RNGLoc),
                    name='E_UB')

                # PV power decomposition
                sub.addConstr(eta_i * (Y_PVL[(i, t, g)] + Y_PVGrid[(i, t, g)]) +
                              Y_PVCur[(i, t, g)] + Y_PVES[(i, t, g)] ==
                              PV[(t, g)] * quicksum(X[(1, l, 2)] + (i - 1) * X[(2, l, 2)]
                                                    for l in RNGLoc), name='PV')

                # DG power decomposition
                sub.addConstr(eta_i * (Y_DGL[(i, t, g)] + Y_DGGrid[(i, t, g)]) +
                              Y_DGES[(i, t, g)] + Y_DGCur[(i, t, g)] ==
                              quicksum(X[(1, l, 3)] + (i - 1) * X[(2, l, 3)]
                                       for l in RNGLoc), name='DG')


                # Load decomposition
                sub.addConstr(eta_i * (Y_ESL[(i, t, g)] + Y_DGL[(i, t, g)] + Y_PVL[(i, t, g)]) + Y_GridL[(i, t, g)] +
                              Y_LL[(i, t, g)] + quicksum(Y_LT[(i, t, to, g)] for to in range(t, T + 1)) == L[(i, t, g)],
                              name='LoadD')

                if t in DontTran:
                    # Don't allow transfer
                    sub.addConstrs((Y_LT[(i, t, to, g)] == 0 for to in range(t, T + 1)),
                                   name='NoTrans')
                else:
                    # Max load transfer
                    sub.addConstr(TransMax * L[(i, t, g)] -
                                  quicksum(Y_LT[(i, t, to, g)] for to in range(t, T + 1)) >= 0,
                                  name='MaxLoadTrans')

                # Load transfer and E level
                sub.addConstr(Y_E[(i, t, g)] - quicksum(Y_LT[(i, to, t, g)] for to in range(1, t)) >= 0,
                              name='ES Load limit')

                # Prohibited transfer to self
                sub.addConstr(Y_LT[(i, t, t, g)] == 0, name='TransIt')

                # ES charging/discharging constraints
                sub.addConstr(-(Y_ESL[(i, t, g)] + Y_ESGrid[(i, t, g)] +
                                Y_PVES[(i, t, g)] + Y_DGES[(i, t, g)] + Y_GridES[(i, t, g)]) >=
                              -quicksum((1 - (i - 1) * ES_d) ** ReInvsYear * X[(1, l, 2)] + (i - 1) * X[(2, l, 2)] for l in RNGLoc))

                # Prohibited transaction with the grid during outage
                if Out_Time[g] != 0:
                    for ot in Out_Time[g]:
                        sub.addConstr(Y_GridL[(i, ot, g)] + Y_GridES[(i, ot, g)] == 0, name='')
                        sub.addConstr(Y_PVGrid[(i, ot, g)] + Y_ESGrid[(i, ot, g)] +
                                      Y_DGGrid[(i, ot, g)] == 0, name='Outage')

            for t in range(1, T - 1):
                # Balance of power flow
                sub.addConstr(Y_E[(i, t + 1, g)] ==
                              Y_E[(i, t, g)] +
                              ES_gamma * (Y_PVES[(i, t, g)] + Y_DGES[(i, t, g)] + eta_i * Y_GridES[(i, t, g)]) -
                              (eta_i / ES_gamma) * (Y_ESL[(i, t, g)] + Y_ESGrid[(i, t, g)]), name='Balance')
    '''Costs'''
    if True:
        CostInv = PVCurPrice * quicksum(Y_PVCur[itg] + Y_DGCur[itg] for itg in Y_1tg) + \
                  quicksum(VoLL_hourly[itg[1]-1] * Y_LL[itg] for itg in Y_1tg) + \
                  DGEffic * quicksum(Y_DGL[itg] + Y_DGGrid[itg] + Y_DGCur[itg] + Y_DGES[itg] for itg in Y_1tg) + \
                  GridPlus * quicksum(Y_GridES[itg] for itg in Y_1tg) - \
                  GridMinus * quicksum(Y_PVGrid[itg] + Y_ESGrid[itg] + Y_DGGrid[itg] for itg in Y_1tg) - \
                  LoadPrice * quicksum(Y_ESL[itg] + Y_DGL[itg] + Y_PVL[itg] for itg in Y_1tg) + \
                  TransPrice * quicksum(Y_LT[(1, to, t, g)] for to in RNGTime for t in RNGTime for g in RNGMonth)

        CostReInv = PVCurPrice * quicksum(Y_PVCur[itg] + Y_DGCur[itg] for itg in Y_2tg) + \
                  quicksum(VoLL_hourly[itg[1]-1] * Y_LL[itg] for itg in Y_2tg) + \
                  DGEffic * quicksum(Y_DGL[itg] + Y_DGGrid[itg] + Y_DGCur[itg] + Y_DGES[itg] for itg in Y_2tg) + \
                  GridPlus * quicksum(Y_GridES[itg] for itg in Y_2tg) - \
                  GridMinus * quicksum(Y_PVGrid[itg] + Y_ESGrid[itg] + Y_DGGrid[itg] for itg in Y_2tg) - \
                  LoadPrice * quicksum(Y_ESL[itg] + Y_DGL[itg] + Y_PVL[itg] for itg in Y_2tg) + \
                  TransPrice * quicksum(Y_LT[(2, to, t, g)] for to in RNGTime for t in RNGTime for g in RNGMonth)

    total_cost = GenPar * (ReInvsYear * CostInv + (Years - ReInvsYear) * CostReInv)
    sub.setObjective(total_cost, sense=GRB.MINIMIZE)
    sub.update()
    sub.write(f'Models/Sub{scen}.mps')
    AMatrix = sub.getA().todok()
    Constrs = sub.getConstrs() 

    Xkeys = range(len(X_ild))
    possibleTkeys = [(r, x) for r in range(len(Constrs)) for x in Xkeys]
    TMatrix = {key: AMatrix[key] for key in possibleTkeys if key in AMatrix.keys()}
    rVector = {c: Constrs[c].RHS for c in range(len(Constrs))}
    with open(f'Models/Sub{scen}-Tr.pkl', 'wb') as handle:
        pickle.dump([TMatrix, rVector], handle)
    handle.close()


class DetModel:
    def __init__(self):
        real = gp.Model(env=env)
        X = {}
        for l in RNGLoc:
            for d in RNGDvc:
                X[(1, l, d)] = real.addVar(vtype=GRB.INTEGER, lb=LB[d], ub=UB[d], name=f'X[1,{l},{d}]')
        for l in RNGLoc:
            for d in RNGDvc:
                X[(2, l, d)] = real.addVar(vtype=GRB.INTEGER, lb=LB[d], ub=UB[d], name=f'X[2,{l},{d}]')

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
            L = {(i, t, g, s): (1 + 0.05 * s) * 20 * Load_scens[f'Month {g}'].iloc[t - 1]
                 for i in RNGSta for t in RNGTime for g in RNGMonth for s in RNGScen}

            PV = {(t, g, s): (1 + 0.05 * s) * PV_scens[f'Month {g}'].iloc[t - 1]
                  for t in RNGTime for g in RNGMonth for s in RNGScen}
            Out_Time = {(g, s): 0 for s in RNGScen for g in RNGMonth}
            for s in RNGScen:
                if Outage_scens[s] != 0:
                    for g in RNGMonth:
                        Out_Time[(g, s)] = [OutageStart + j for j in range(int(Outage_scens[s]))]
        '''Scheduling constraints'''
        for s in RNGScen:
            for i in RNGSta:  # RNGSta = (1, 2)
                for g in RNGMonth:
                    # ES levels
                    real.addConstr(Y_E[(i, 1, g, s)] == SOC_UB * quicksum(
                        (1-(i-1)*ES_d)**ReInvsYear*X[(1, l, 1)] + (i - 1) * X[(2, l, 1)] for l in RNGLoc))
                    # Only use the summation of ES capacity before/after reinvestment for rhs update in ABC BB-D

                    for t in RNGTime:
                        # Limits on energy level in ES
                        real.addConstr(Y_E[(i, t, g, s)] >= SOC_LB * quicksum(
                            (1-(i-1)*ES_d)**ReInvsYear*X[(1, l, 1)] + (i - 1) * X[(2, l, 1)] for l in RNGLoc))

                        real.addConstr(-Y_E[(i, t, g, s)] >= -SOC_UB * quicksum(
                            (1-(i-1)*ES_d)**ReInvsYear*X[(1, l, 1)] + (i - 1) * X[(2, l, 1)] for l in RNGLoc))

                        # PV power decomposition
                        real.addConstr((Y_PVL[(i, t, g, s)] + Y_PVES[(i, t, g, s)] +
                                        Y_PVCur[(i, t, g, s)] + Y_PVGrid[(i, t, g, s)]) ==
                                       PV[(t, g, s)] * quicksum(
                            X[(1, l, 2)] + (i - 1) * X[(2, l, 2)] for l in RNGLoc))

                        # DG power decomposition
                        real.addConstr(Y_DGL[(i, t, g, s)] + Y_DGES[(i, t, g, s)] +
                                       Y_DGGrid[(i, t, g, s)] + Y_DGCur[(i, t, g, s)] ==
                                       quicksum(X[(1, l, 3)] + (i - 1) * X[(2, l, 3)] for l in RNGLoc))

                        # Assigned load decomposition
                        real.addConstr(Y_LH[(i, t, g, s)] ==
                                       eta_i * (Y_ESL[(i, t, g, s)] + Y_DGL[(i, t, g, s)] + Y_PVL[(i, t, g, s)]) +
                                       Y_GridL[(i, t, g, s)], name='')

                        real.addConstr(Y_LH[(i, t, g, s)] + Y_LL[(i, t, g, s)] +
                                       quicksum(Y_LT[(i, t, to, g, s)] for to in range(t, T + 1)) ==
                                       L[(i, t, g, s)], name='')

                        if t in DontTran:
                            # Don't allow transfer
                            real.addConstrs(Y_LT[(i, t, to, g, s)] == 0 for to in range(t, T + 1))
                        else:
                            # Max load transfer
                            real.addConstr(TransMax * L[(i, t, g, s)] -
                                           quicksum(Y_LT[(i, t, to, g, s)] for to in range(t, T + 1)) >= 0, name='')

                        # Load transfer and E level
                        real.addConstr(Y_E[(i, t, g, s)] - quicksum(Y_LT[(i, to, t, g, s)] for to in range(1, t)) >= 0,
                                       name='')

                        # Prohibited transfer to self
                        real.addConstr(Y_LT[(i, t, t, g, s)] == 0, name='')

                        # ES charging/discharging constraints
                        real.addConstr((UB[1] + (i - 1) * UB[1]) * U_E[(i, t, g, s)] -
                                       (Y_ESL[(i, t, g, s)] + Y_ESGrid[(i, t, g, s)]) >= 0, name='')
                        real.addConstr((UB[1] + (i - 1) * UB[1]) * (1 - U_E[(i, t, g, s)]) -
                                       (Y_PVES[(i, t, g, s)] + Y_GridES[(i, t, g, s)] + Y_DGES[(i, t, g, s)]) >= 0,
                                       name='')

                        real.addConstr(
                            (UB[1] + UB[2] + UB[3] + (i - 1) * (UB[1] + UB[2] + UB[3])) * U_G[(i, t, g, s)] -
                            (Y_ESGrid[(i, t, g, s)] + Y_PVGrid[(i, t, g, s)] + Y_DGGrid[(i, t, g, s)]) >= 0,
                            name='')
                        real.addConstr(
                            (UB[1] + UB[2] + UB[3] + (i - 1) * (UB[1] + UB[2] + UB[3])) * (1 - U_G[(i, t, g, s)]) -
                            (Y_GridES[(i, t, g, s)] + Y_GridL[(i, t, g, s)]) >= 0, name='')

                        # Prohibited transaction with the grid during outage
                        if Out_Time[(g, s)] != 0:
                            for ot in Out_Time[(g, s)]:
                                real.addConstr(Y_GridL[(i, ot, g, s)] + Y_GridES[(i, ot, g, s)] == 0, name='')
                                real.addConstr(Y_PVGrid[(i, ot, g, s)] + Y_ESGrid[(i, ot, g, s)] +
                                               Y_DGGrid[(i, ot, g, s)] == 0, name='')
                                real.addConstrs(U_G[(i, ot, g, s)] == 0)

                    for t in RNGTimeMinus:
                        # Balance of power flow
                        real.addConstr(Y_E[(i, t + 1, g, s)] ==
                                       Y_E[(i, t, g, s)] +
                                       ES_gamma * (Y_PVES[(i, t, g, s)] + Y_DGES[(i, t, g, s)] + eta_i * Y_GridES[
                            (i, t, g, s)]) -
                                       (eta_i / ES_gamma) * (Y_ESL[(i, t, g, s)] + Y_ESGrid[(i, t, g, s)]), name='')
        '''Costs'''
        if True:
            Total_cost = 0
            for s in RNGScen:
                Costs = 0
                for i in RNGSta:
                    for g in RNGMonth:
                        for t in RNGTime:
                            # Curtailment cost
                            Costs += PVCurPrice * (Y_PVCur[(i, t, g, s)] + Y_DGCur[(i, t, g, s)])
                            # Losing load cost
                            Costs += VoLL * Y_LL[(i, t, g, s)]
                            # DG cost
                            Costs += DGEffic * (Y_DGL[(i, t, g, s)] + Y_DGGrid[(i, t, g, s)] +
                                                Y_DGCur[(i, t, g, s)] + Y_DGES[(i, t, g, s)])
                            # Import/Export cost
                            Costs += GridPlus * Y_GridES[(i, t, g, s)] - \
                                     GridMinus * (Y_PVGrid[(i, t, g, s)] + Y_ESGrid[(i, t, g, s)] + Y_DGGrid[(i, t, g, s)]) - \
                                     LoadPrice * (Y_ESL[(i, t, g, s)] + Y_DGL[(i, t, g, s)] + Y_PVL[(i, t, g, s)])

                            # DRP cost
                            Costs += TransPrice * quicksum(Y_LT[(i, to, t, g, s)] for to in RNGTime)

                Total_cost += norm_probs[s] * GenPar * Costs
            real.setObjective(Total_cost + Capital, sense=GRB.MINIMIZE)

        '''Save model data'''
        if True:
            real.write(f'Models/real.mps')
            print(real)


if __name__ == '__main__':
    MasterProb()

    '''for scen in tqdm.tqdm(norm_probs):
        SubProb(scen)
        if scen == 2:
            break'''
    # real = DetModel()





import pickle

import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import quicksum, GRB
import tqdm
import warnings

warnings.filterwarnings('ignore')

env = gp.Env()
env.setParam('OutputFlag', 0)


# Specify the community the model is being built 1: sunnyside, 2:Dove Springs, 3:Rogers Washington
com = 1
com_folder = {1: 'HarrisCounty-SS',
              2: 'TravisCounty-DS',
              3: 'TravisCounty-RW'}

# Community-dependent data
b = 1
UB_dict = {1: {(1, 1): b*300, (1, 2): b*114, (1, 3): b*100,
               (2, 1): b*300, (2, 2): b*396, (2, 3): b*100,
               (3, 1): b*300, (3, 2): b*141, (3, 3): b*100,
               (4, 1): b*300, (4, 2): b*133, (4, 3): b*100,
               (5, 1): b*300, (5, 2): b*246, (5, 3): b*100},

           2: {(1, 1): 600, (1, 2): 318, (1, 3): 100,
               (2, 1): 600, (2, 2): 299, (2, 3): 100,
               (3, 1): 600, (3, 2): 2142, (3, 3): 100,
               (4, 1): 600, (4, 2): 1190, (4, 3): 100},

           3: {(1, 1): 300, (1, 2): 930, (1, 3): 100,
               (2, 1): 300, (2, 2): 40, (2, 3): 100,
               (3, 1): 300, (3, 2): 334, (3, 3): 100}}

LocationPrice = {1: {1: 18164, 2: 62936, 3: 22469, 4: 21467, 5: 39160},
                 2: {1: 87382, 2: 82077, 3: 588080, 4: 326867},
                 3: {1: 285579, 2: 12443, 3: 102558}}

GridPlus_dict = {1: 0.15,
                 2: 0.14,
                 3: 0.14}
GridMinus_dict = {1: 0.12,
                  2: 0.097,
                  3: 0.097}

VoLL_dict = {1: 0.2764,
             2: 0.2567,
             3: 0.5157}

Vuln_dict = {1: 0.59,
             2: 1,
             3: 0}


S = 30
# Preparing Power Outage Scenarios
if True:
    OT = pd.read_csv('Scenarios/Outage/Outage Scenarios - reduced.csv')
    Outage_probs = {i: OT['Probability'].iloc[i] for i in range(S)}
    Outage_scens = {i: int(OT['Gamma Scenario'].iloc[i]) for i in range(S)}

# Preparing PV scenarios
if True:
    PV_scens = {i: {j: [] for j in range(1, 13)} for i in range(S)}
    PV_probs = {i: 1 for i in range(S)}
    month_counter = 1
    for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        dpv = pd.read_csv(f'Scenarios/PV/{com_folder[com]}/PVscenario-{m}.csv')
        cols = dpv.columns[2:27]

        for i in range(S):
            PV_probs[i] = PV_probs[i] * (dpv['probs'].iloc[i])
            dw_single = dpv[cols].iloc[i]
            week_long = pd.concat([dw_single for _ in range(7)])
            PV_scens[i][month_counter] = week_long
        month_counter += 1

# Preparing Load Profiles
if True:
    Load_scens = {i: {j: [] for j in range(1, 13)} for i in range(S)}
    Load_probs = {i: 1 for i in range(S)}
    month_counter = 1
    for m in ['JanFeb', 'MarApr', 'MayJun', 'JulAug', 'SepOct', 'NovDec']:
        dw = pd.read_csv(f'Scenarios/Load Demand/{com_folder[com]}/LoadScenarios-{m}-w.csv')
        de = pd.read_csv(f'Scenarios/Load Demand/{com_folder[com]}/LoadScenarios-{m}-e.csv')
        cols = dw.columns[2:99]
        for i in range(S):
            Load_probs[i] = Load_probs[i] * (dw['probs'].iloc[i] * de['probs'].iloc[i])

            dw_single = dw[cols].iloc[i]
            de_single = de[cols].iloc[i]
            week_long = pd.concat([dw_single, dw_single, dw_single, dw_single, dw_single, de_single, de_single])
            week_long_hourly = []

            j = 0
            while j <= 668:
                summ = sum(week_long[j + k] for k in range(4))
                week_long_hourly.append(summ)
                j += 4

            Load_scens[i][month_counter] = week_long_hourly
            Load_scens[i][month_counter + 1] = week_long_hourly
        month_counter += 2

# Preparing Load Growth Scenarios
if True:
    AG = pd.read_csv('Scenarios/Load Demand/annual growth scenarios.csv')
    AG_probs = {i: AG['Probability'].iloc[i] for i in range(S)}
    AG_scens = {i: AG['Lognorm Scenario'].iloc[i]/100 for i in range(S)}

all_probs = {i: Outage_probs[i] * PV_probs[i] * Load_probs[i] * AG_probs[i] for i in Outage_probs}
norm_probs = {i: all_probs[i] / sum(all_probs.values()) for i in Outage_probs}

with open('Data/ScenarioProbabilities.pkl', 'wb') as handle:
    pickle.dump(norm_probs, handle)
handle.close()


# Counts
T = 168  # count of hours per week
LCount = len(LocationPrice[com])  # count of locations
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

summer_peak = [24 * i + 13 + j for i in range(6) for j in range(8)]
# Each week in summer from hour 1 to 19 pm transfer is prohibited.
winter_peak = np.concatenate(([24 * i + 6 + j for i in range(6) for j in range(5)],
                             [24 * i + 18 + j for i in range(6) for j in range(5)]))

# Each week in winter from hour 6 am to 10 am and 6 pm to 10 pm transfer is prohibited.

DontTrans = {1: winter_peak, 2: winter_peak, 3: winter_peak,
             4: [], 5: [],
             6: summer_peak, 7: summer_peak, 8: summer_peak, 9: summer_peak,
             10: [], 11: [], 12: []}

# Sensitivity Parameters
InvImportance = 0.25 + 0.55*(1 - Vuln_dict[com])
VoLL_sensitivity = 20
TransMax = 0.25
ReInvsYear = 10
Operational_Rate = 0.01
Labor_Factor = 0.12

# Global parameters
Budget1 = 10000000
Budget2 = Budget1 / 2
Years = 20
Interest_Rate = 0.02
PA_Factor1 = ((1 + Interest_Rate) ** Years - 1) / (Interest_Rate * (1 + Interest_Rate) ** Years)
PA_Factor2 = ((1 + Interest_Rate) ** (Years - ReInvsYear) - 1) / (Interest_Rate * (1 + Interest_Rate) ** (Years - ReInvsYear))
PF_Factor = 1 / (1 + Interest_Rate) ** ReInvsYear
C = {1: (1 + Labor_Factor) * 300, 2: (1 + Labor_Factor) * 2780, 3: (1 + Labor_Factor) * 400}  # order is: [ES, PV, DG]
CO1 = {i: C[i] * (1 + Operational_Rate * PA_Factor1) for i in (1, 2, 3)}
CO2 = {i: C[i] * (1 + Operational_Rate * PA_Factor2) for i in (1, 2, 3)}
F = {j: LocationPrice[com][j] for j in LocationPrice[com]}

UB = UB_dict[com]  # Upper bound of devices capacity (location, device)

# Efficiencies and performances
ES_gamma = 0.85
DG_gamma = 0.95  # %
DG_consumption = 0.4  # gal/kW
FuelPrice = 3.61  # $/gal
DGEffic = DG_consumption * FuelPrice  # Fuel cost of DG: $/kWh
ES_d = 0.02

# Electricity Prices
zeta = 0.8  # The parameter specifying what percentage of electricity price be determined as price to sell power to households
GridPlus = GridPlus_dict[com]  # $/kWh (importing price of power from the grid)
GridMinus = GridMinus_dict[com]  # exporting price of power back to grid
LoadPrice = zeta * GridPlus
PVCurPrice = GridMinus
DGCurPrice = GridMinus + DGEffic

VoLL = (2 - VoLL_dict[com]) * VoLL_sensitivity * GridPlus

SOC_UB, SOC_LB = 0.9, 0.1
eta_i = 0.9
GenPar = (365 / 7) / MCount

X_ld = [(l, d)for l in RNGLoc for d in RNGDvc]
X_il = [(i, l) for i in RNGSta for l in RNGLoc]
Y_itg = [(i, t, g)
         for i in RNGSta for t in RNGTime for g in RNGMonth]
Y_ittg = [(i, t, to, g)
          for i in RNGSta for t in RNGTime for to in range(t, T+1) for g in RNGMonth]

Y_itgs = [(i, t, g, s)
          for i in RNGSta for t in RNGTime for g in RNGMonth for s in RNGScen]
Y_ittgs = [(i, t, to, g, s)
           for i in RNGSta for t in RNGTime for to in RNGTime for g in RNGMonth for s in RNGScen]
Y_1tg = [(1, t, g) for t in RNGTime for g in RNGMonth]
Y_2tg = [(2, t, g) for t in RNGTime for g in RNGMonth]
eta_M = -100000000
Xkeys = range(len(X_ld) + LCount)


def MasterProb():
    master = gp.Model('MasterProb', env=env)
    '''Investment'''
    X = master.addVars(X_ld, vtype=GRB.INTEGER, name='X')
    U = master.addVars(RNGLoc, vtype=GRB.BINARY, name='U')
    # Investment constraint
    master.addConstr(quicksum(F[l] * U[l] for l in RNGLoc) +
                     quicksum(X[(l, d)] * C[d] for l in RNGLoc for d in RNGDvc) <= Budget1,
                     name='Investment')
    # Install if only location is open
    master.addConstrs((X[(l, d)] <= UB[(l, d)] * U[l]
                       for l in RNGLoc for d in RNGDvc), name='Location Allowance')

    Capital = quicksum(PA_Factor1 * F[l] * U[l] for l in RNGLoc) +\
              quicksum(X[(l, d)] * CO1[d] for l in RNGLoc for d in RNGDvc)

    eta = master.addVar(lb=eta_M, name='eta')
    master.setObjective(InvImportance * Capital + eta, sense=GRB.MINIMIZE)

    # Save master in mps + save data in pickle
    master.update()
    master.write('Models/Master.mps')
    with open('Models/Master_X_Info.pkl', 'wb') as f:
        pickle.dump(range(len(X_ld)), f)
    f.close()


def SubProb(scen):
    sub = gp.Model(env=env)

    if True:
        X1 = sub.addVars(X_ld, name=f'X1')
        U1 = sub.addVars(RNGLoc, ub=1, name='U1')
        X2 = sub.addVars(X_ld, name=f'X2')
        U2 = sub.addVars(RNGLoc, ub=1, name='U2')
        Y_PVES = sub.addVars(Y_itg, name=f'Y_PVES')  # PV to ES (1 before reinvestment, 2 after)
        Y_DGES = sub.addVars(Y_itg, name=f'Y_DGES')  # DE to ES
        Y_GridES = sub.addVars(Y_itg, name=f'Y_GridES')  # Grid to ES
        Y_PVL = sub.addVars(Y_itg, name=f'Y_PVL')  # Pv to L
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
        L = {(i, t, g): (1 + (i - 1) * AG_scens[scen]) ** ReInvsYear * Load_scens[scen][g][t - 1]
             for i in RNGSta for t in RNGTime for g in RNGMonth}

        PV = {(t, g): 0.25 * PV_scens[scen][g][t - 1] for t in RNGTime for g in RNGMonth}

        Out_Time = {g: 0 for g in RNGMonth}
        if Outage_scens[scen] != 0:
            if Outage_scens[scen] >= 168 - OutageStart:
                for g in RNGMonth:
                    Out_Time[g] = [OutageStart + j for j in range(168 - OutageStart + 1)]
            else:
                for g in RNGMonth:
                    Out_Time[g] = [OutageStart + j for j in range(int(Outage_scens[scen]))]
        VoLL_hourly = {i: {} for i in RNGMonth}
        for g in RNGMonth:
            for tt in RNGTime:
                if tt in Out_Time[g]:
                    if tt in DontTrans[g]:
                        VoLL_hourly[g][tt] = VoLL
                    else:
                        VoLL_hourly[g][tt] = 0.7 * VoLL
                else:
                    VoLL_hourly[g][tt] = 0
    '''Reinvestment Constraints'''
    if True:
        # Investment constraint
        sub.addConstr(quicksum(F[l] * (U2[l] - U1[l]) for l in RNGLoc) +
                      quicksum(X2[(l, d)] * C[d] for l in RNGLoc for d in RNGDvc) <= Budget2,
                         name='Investment')
        # Capacity limit ES
        sub.addConstrs((X2[(l, 1)] + X1[(l, 1)] <= 1.5 * UB[(l, 1)]
                           for l in RNGLoc), name='ES Location Limit')
        # Capacity limit PV
        sub.addConstrs((X2[(l, 2)] + X1[(l, 2)] <= UB[(l, 2)]
                        for l in RNGLoc), name='PV Location Limit')
        # Capacity limit DG
        sub.addConstrs((X2[(l, 3)] + X1[(l, 3)] <= UB[(l, 3)]
                        for l in RNGLoc), name='DG Location Limit')
        # Install devices if location open
        sub.addConstrs((X2[(l, d)] <= U2[l] * UB[(l, d)]
                           for l in RNGLoc for d in RNGDvc), name='Install if Location')
        # Open only once
        sub.addConstrs((U2[l] >= U1[l] for l in RNGLoc), name='One Opening')
    '''Scheduling constraints'''
    for i in RNGSta:  # RNGSta = (1, 2)
        for g in RNGMonth:
            Total_ES = quicksum((1 - (i - 1) * ES_d) ** ReInvsYear * X1[(l, 1)] +
                                (i - 1) * X2[(l, 1)] for l in RNGLoc)
            # First ES levels
            sub.addConstr(Y_E[(i, 1, g)] == SOC_UB * Total_ES, name='t1')

            for t in RNGTime:

                # Calculate total transfer from t
                Total_Transfer_from_t = quicksum(Y_LT[(i, t, to, g)] for to in range(t, T + 1))
                Total_transfer_to_t = quicksum(Y_LT[(i, to, t, g)] for to in range(1, t+1))

                # Limits on energy level in ES
                sub.addConstr(Y_E[(i, t, g)] + Total_transfer_to_t >= SOC_LB * Total_ES, name='E_LB')

                sub.addConstr(Y_E[(i, t, g)] + Total_transfer_to_t <= SOC_UB * Total_ES, name='E_UB')

                # Limits on Charge/Discharge
                sub.addConstr(Y_ESL[(i, t, g)] + Y_ESGrid[(i, t, g)] <= (SOC_UB-SOC_LB) * Total_ES, name='ES_Discharge')

                sub.addConstr(Y_PVES[(i, t, g)] + Y_DGES[(i, t, g)] + Y_GridES[(i, t, g)] <= (SOC_UB-SOC_LB) * Total_ES, name='ES_Charge')

                # PV power decomposition
                sub.addConstr((Y_PVL[(i, t, g)] + Y_PVGrid[(i, t, g)]) +
                              Y_PVCur[(i, t, g)] + Y_PVES[(i, t, g)] ==
                              PV[(t, g)] * quicksum(X1[(l, 2)] + (i - 1) * X2[(l, 2)]
                                                    for l in RNGLoc), name='PV')

                # DG power decomposition
                sub.addConstr((Y_DGL[(i, t, g)] + Y_DGGrid[(i, t, g)]) +
                              Y_DGES[(i, t, g)] + Y_DGCur[(i, t, g)] ==
                              DG_gamma * quicksum(X1[(l, 3)] + (i - 1) * X2[(l, 3)]
                                       for l in RNGLoc), name='DG')

                # Load decomposition
                sub.addConstr(Y_ESL[(i, t, g)] + Y_DGL[(i, t, g)] + Y_PVL[(i, t, g)] + Y_GridL[(i, t, g)] +
                              Y_LL[(i, t, g)] + Total_transfer_to_t - Total_Transfer_from_t == L[(i, t, g)],
                              name='LoadD')

                # if not outage, lost must be zero
                if t not in Out_Time[g]:
                    sub.addConstr(Y_LL[(i, t, g)] == 0, name='NoOutNoLoss')
                    sub.addConstr(Y_GridL[(i, t, g)] >= 0.75*L[(i, t, g)], name='NoOutUseGrid')
                    sub.addConstr(Total_transfer_to_t == 0, name='NoTransToNonOut')

                if t in DontTrans[g]:
                    # Don't allow transfer
                    sub.addConstr(Total_Transfer_from_t == 0, name='NoTrans')
                else:
                    # Max load transfer
                    sub.addConstr(Total_Transfer_from_t <= TransMax * L[(i, t, g)],
                                  name='MaxLoadTrans')

                # Transfer should not exceed ES level
                sub.addConstr(Total_Transfer_from_t <= (Y_E[(i, t, g)] - SOC_LB*Total_ES),
                              name='TransIfPoss')

                # Transfer to self prohibited
                sub.addConstr(Y_LT[(i, t, t, g)] == 0, name='NoTransToSelf')

                # Balance of power flow
                if t != T:
                    sub.addConstr(Y_E[(i, t + 1, g)] ==
                              Y_E[(i, t, g)] - Total_Transfer_from_t +
                              ES_gamma * (Y_PVES[(i, t, g)] + Y_DGES[(i, t, g)] + eta_i * Y_GridES[(i, t, g)]) -
                              (Y_ESL[(i, t, g)] + Y_ESGrid[(i, t, g)]) / (eta_i * ES_gamma), name='Balance')

                # Prohibited transaction with the grid during outage
                if t in Out_Time[g]:
                    sub.addConstr(Y_GridL[(i, t, g)] + Y_GridES[(i, t, g)] +
                                  Y_PVGrid[(i, t, g)] + Y_ESGrid[(i, t, g)] +
                                  Y_DGGrid[(i, t, g)] == 0, name='GridTransaction')
    '''Costs'''
    if True:
        Capital2 = quicksum(PA_Factor2 * F[l] * (U2[l] - U1[l]) for l in RNGLoc) + \
                   quicksum(X2[(l, d)] * CO2[d] for l in RNGLoc for d in RNGDvc)

        CostInv = sum(PVCurPrice * Y_PVCur[itg] + DGCurPrice * Y_DGCur[itg] for itg in Y_1tg) + \
                  sum(VoLL_hourly[itg[2]][itg[1]] * Y_LL[itg] for itg in Y_1tg) + \
                  DGEffic * sum(Y_DGL[itg] + Y_DGGrid[itg] + Y_DGCur[itg] + Y_DGES[itg] for itg in Y_1tg) + \
                  GridPlus * sum(Y_GridES[itg] + Y_GridL[itg] for itg in Y_1tg) - \
                  GridMinus * sum(Y_PVGrid[itg] + Y_ESGrid[itg] + Y_DGGrid[itg] for itg in Y_1tg) - \
                  LoadPrice * sum(Y_ESL[itg] + Y_DGL[itg] + Y_PVL[itg] for itg in Y_1tg)

        CostReInv = sum(PVCurPrice * Y_PVCur[itg] + DGCurPrice * Y_DGCur[itg] for itg in Y_2tg) + \
                    sum(VoLL_hourly[itg[2]][itg[1]] * Y_LL[itg] for itg in Y_2tg) + \
                    DGEffic * sum(Y_DGL[itg] + Y_DGGrid[itg] + Y_DGCur[itg] + Y_DGES[itg] for itg in Y_2tg) + \
                    GridPlus * sum(Y_GridES[itg] + Y_GridL[itg] for itg in Y_2tg) - \
                    GridMinus * sum(Y_PVGrid[itg] + Y_ESGrid[itg] + Y_DGGrid[itg] for itg in Y_2tg) - \
                    LoadPrice * sum(Y_ESL[itg] + Y_DGL[itg] + Y_PVL[itg] for itg in Y_2tg)

    total_cost = InvImportance * Capital2 + GenPar * (ReInvsYear * CostInv + (Years - ReInvsYear) * CostReInv)
    sub.setObjective(total_cost, sense=GRB.MINIMIZE)
    sub.update()
    sub.write(f'Models/Sub{scen}.mps')
    AMatrix = sub.getA().todok()
    Constrs = sub.getConstrs()

    possible_Tr_indices = [(r, x) for r in range(len(Constrs)) for x in Xkeys]
    TMatrix = {key: AMatrix[key] for key in possible_Tr_indices if key in AMatrix.keys()}
    rVector = {c: Constrs[c].RHS for c in range(len(Constrs))}
    with open(f'Models/Sub{scen}-Tr.pkl', 'wb') as f:
        pickle.dump([TMatrix, rVector], f)
    f.close()


if __name__ == '__main__':
    MasterProb()
    for scen in tqdm.tqdm(norm_probs.keys()):
        SubProb(scen)







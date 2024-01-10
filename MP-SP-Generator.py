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
T = 6          # count of hours per week
LCount = 1      # count of locations
DVCCount = 3    # count of devices
MCount = 1   # count of months
HCount = 1     # count of households
OutageStart = 1  # hour that outage starts

# Import data
Scens, Probs, Load, PV_Unit = geneCases(consumer=HCount)

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
C = {1: 600, 2: 2780, 3: 500}  # order is: [ES, PV, DG]
CO = {i: C[i] * (PA_Factor + Operational_Rate) for i in (1, 2, 3)}

UB0 = {1: 100, 2: 100, 3: 60}
LB0 = {1: 20.5, 2: 20.9, 3: 10.1}
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
# Save all ranges for the algorithm part
with open('Data/Ranges.pkl', 'wb') as handle:
    pickle.dump([RNGLoc, RNGDvc, RNGTime, RNGMonth, RNGHouse, RNGScen, RNGSta, Y_tg, Y_htg, Y_ttg], handle)
handle.close()

eta_M = -100

class MasterPro:
    def __init__(self):
        self.master = gp.Model('MasterProb', env=env)
        '''Investment & Reinvestment variables'''
        X = {l: {} for l in RNGSta}
        for l in RNGLoc:
            for d in RNGDvc:
                X[l][(1, d)] = self.master.addVar(lb=LB0[d], ub=UB0[d], name=f'X[{l}][(1,{d})]')
                X[l][(2, d)] = self.master.addVar(lb=LB1[d], ub=UB1[d], name=f'X[{l}][(2,{d})]')
        self.X_keys = range(1, LCount * (2 * DVCCount) + 1)  # starts from 1
        self.X_indcies = {l: [(ii, d) for ii in RNGSta for d in RNGDvc] for l in RNGLoc}
        Capital = 0
        for l in RNGLoc:
            # Investment constraint
            self.master.addConstr(quicksum(X[l][(1, d)] * C[d] for d in RNGDvc) <= Budget1, name='IB')
            # ReInvestment constraint
            self.master.addConstr(quicksum(X[l][(2, d)] * C[d] for d in RNGDvc) <= Budget2, name='RIB')

            # Formulate capital cost
            Capital += quicksum((X[l][(1, d)] + X[l][(2, d)]) * CO[d] for d in RNGDvc)


        # Get A and b to save it
        self.master.update()
        A, b = self.GetAb()  # Checked: correct

        # Save upper and lower bounds
        upper_bounds = {}
        lower_bounds = {}
        for l in RNGLoc:  # Checked: correct
            upper_bounds[l] = {}
            for ii in RNGSta:
                for d in RNGDvc:
                    # Bounds on X decisions before reinvestment
                    if l == 1:
                        bound = -UB0[d]
                    else:
                        bound = -UB1[d]
                    upper_bounds[l][(ii, d)] = bound
        for l in RNGLoc:  # Checked: correct
            lower_bounds[l] = {}
            for ii in RNGSta:
                for d in RNGDvc:
                    # Bounds on X decisions before reinvestment
                    if l == 1:
                        bound = LB0[d]
                    else:
                        bound = LB1[d]
                    lower_bounds[l][(ii, d)] = bound

        eta = self.master.addVar(lb=eta_M, name='eta')
        self.master.setObjective(Capital + eta, sense=GRB.MINIMIZE)

        # Save master in mps + save data in pickle
        self.master.update()
        self.master.write('Models/Master.mps')
        with open(f'Models/Master.pkl', 'wb') as handle:
            pickle.dump([A, b, upper_bounds, lower_bounds], handle)
        handle.close()

    def GetAb(self):
        Coefs = IndexUp(self.master.getA().todok())  # dictionary that indexed up. Increase indexes by one to avoid 0 index
        m = np.max([key[0] for key in Coefs.keys()])  # m only covers the count of constraints that we need
        A_sparse = Coefs
        b_sparse = {}
        ConstrList = self.master.getConstrs()
        for row in range(1, m + 1):
            b_sparse[row] = ConstrList[row - 1].rhs

        # Note that T, W, r are spares matrices in a dictionary format.
        return A_sparse, b_sparse

    def GetIndicesKeys(self):
        return self.X_keys, self.X_indcies

class SubProb:
    def __init__(self, scen):
        self.sub = gp.Model(env=env)

        '''X vars to keep the model accessible for TMatrix'''
        if True:
            x_fixed = {l: self.sub.addVars(X_d, name='Y_X') for l in RNGLoc}
            self.X_keys = range(1, LCount * (2 * DVCCount) + 1)  # starts from 1

        '''Scheduling variables'''
        if True:
            U_E = {i: self.sub.addVars(Y_tg, name=f'U_E{i}') for i in RNGSta}  # Charge/discharge binary
            U_G = {i: self.sub.addVars(Y_tg, name=f'U_G{i}') for i in RNGSta}  # Import/export binary
            Y_PVES = {i: self.sub.addVars(Y_tg, name=f'Y_PVES{i}') for i in RNGSta}  # PV to ES (1 before reinvestment, 2 after)
            Y_DGES = {i: self.sub.addVars(Y_tg, name=f'Y_DGES{i}') for i in RNGSta}  # DE to ES
            Y_GridES = {i: self.sub.addVars(Y_tg, name=f'Y_GridES{i}') for i in RNGSta}  # Grid to ES
            Y_PVL = {i: self.sub.addVars(Y_tg, name=f'Y_PVL{i}') for i in RNGSta}  # Pv to L
            Y_DGL = {i: self.sub.addVars(Y_tg, name=f'Y_DGL{i}') for i in RNGSta}  # Dg to L
            Y_ESL = {i: self.sub.addVars(Y_tg, name=f'Y_ESL{i}') for i in RNGSta}  # ES to L
            Y_GridL = {i: self.sub.addVars(Y_tg, name=f'Y_GridL{i}') for i in RNGSta}  # Grid to L
            Y_PVCur = {i: self.sub.addVars(Y_tg, name=f'Y_PVCur{i}') for i in RNGSta}  # PV Curtailed
            Y_DGCur = {i: self.sub.addVars(Y_tg, name=f'Y_DGCur{i}') for i in RNGSta}  # DG curtailed
            Y_PVGrid = {i: self.sub.addVars(Y_tg, name=f'Y_DGGrid{i}') for i in RNGSta}  # PV to Grid
            Y_DGGrid = {i: self.sub.addVars(Y_tg, name=f'Y_PVGrid{i}') for i in RNGSta}  # Dg to Grid
            Y_ESGrid = {i: self.sub.addVars(Y_tg, name=f'Y_ESGrid{i}') for i in RNGSta}  # ES to Grid
            Y_E = {i: self.sub.addVars(Y_tg, name=f'Y_E{i}') for i in RNGSta}  # ES level of energy
            self.Y_keys = range(1, T * MCount * 2 + 1)  # Key refers to the variable number in its dictionary

            Y_LH = {i: self.sub.addVars(Y_htg, name=f'Y_LH{i}') for i in RNGSta}  # Load served
            Y_LL = {i: self.sub.addVars(Y_htg, name=f'Y_LL{i}') for i in RNGSta}  # Load lost
            Y_LT = {i: self.sub.addVars(Y_ttg, name=f'Y_LT{i}') for i in RNGSta}  # Load transferred
            self.Yh_keys = range(1, T * MCount * HCount * 2 + 1)
            self.Yht_keys = range(1, T * T * MCount * HCount * 2 + 1)

            '''Saving all vars keys in W and T matrix'''
            var_names = ['U_E', 'U_G', 'Y_PVES', 'Y_DGES', 'Y_GridES', 'Y_PVL',
                         'Y_DGL', 'Y_ESL', 'Y_GridL', 'Y_PVCur', 'Y_DGCur',
                         'Y_PVGrid', 'Y_DGGrid', 'Y_ESGrid', 'Y_E', 'Y_LH',
                         'Y_LL', 'Y_LT']
            Y_lenght = T * MCount * 2
            Y_h_lenght = T * MCount * HCount * 2
            Y_t_lenght = T * T * MCount * HCount * 2
            last_key = LCount * (2 * DVCCount)
            XY_keys = {'Y_X': (1, last_key)}
            for name in var_names:
                if name not in ('Y_LH', 'Y_LL', 'Y_LT'):
                    XY_keys[name] = (last_key+1, last_key + Y_lenght)
                    last_key += Y_lenght
                elif name != 'Y_LT':
                    XY_keys[name] = (last_key + 1, last_key + Y_h_lenght)
                    last_key += Y_h_lenght
                else:
                    XY_keys[name] = (last_key + 1, last_key + Y_t_lenght)
                    last_key += Y_t_lenght
            self.XY_keys = XY_keys
        '''Specify Load Demand, PV, Outage Duration for the scenario s'''
        if True:
            # Define the load profiles and PV profiles
            L1 = {(h, t, g): Load[h - 1][f'Month {g}'].iloc[t - 1]
                  for h in RNGHouse for t in RNGTime for g in RNGMonth}
            L2 = {(h, t, g): 1.3 * Load[h - 1][f'Month {g}'].iloc[t - 1]
                  for h in RNGHouse for t in RNGTime for g in RNGMonth}
            L = {1: L1, 2: L2}

            PV = {(t, g): PV_Unit[f'Month {g}'].iloc[t - 1]
                  for t in RNGTime for g in RNGMonth}

            Out_Time = {g: 0 for g in RNGMonth}
            if Scens[scen] != 0:
                for g in RNGMonth:
                    Out_Time[g] = [OutageStart + j for j in range(int(Scens[scen]))]

        '''Scheduling constraints'''
        for ii in RNGSta:  # RNGSta = (1, 2)
            for g in RNGMonth:
                # ES levels
                self.sub.addConstr(Y_E[ii][(1, g)] == SOC_UB * quicksum(x_fixed[l][(1, 1)] + (ii - 1) * x_fixed[l][(2, 1)] for l in RNGLoc))
                # Only use the summation of ES capacity before/after reinvestment for rhs update in ABC BB-D

                for t in RNGTime:
                    # Limits on energy level in ES
                    self.sub.addConstr(Y_E[ii][(t, g)] >= SOC_LB * quicksum(x_fixed[l][(1, 1)] + (ii - 1) * x_fixed[l][(2, 1)] for l in RNGLoc))


                    self.sub.addConstr(-Y_E[ii][(t, g)] >= -SOC_UB * quicksum(x_fixed[l][(1, 1)] + (ii - 1) * x_fixed[l][(2, 1)] for l in RNGLoc))

                    # PV power decomposition
                    self.sub.addConstr((Y_PVL[ii][(t, g)] + Y_PVES[ii][(t, g)] +
                                        Y_PVCur[ii][(t, g)] + Y_PVGrid[ii][(t, g)]) ==
                                        PV[(t, g)] * quicksum(x_fixed[l][(1, 2)] + (ii - 1) * x_fixed[l][(2, 2)] for l in RNGLoc))

                    # DG power decomposition
                    self.sub.addConstr(Y_DGL[ii][(t, g)] + Y_DGES[ii][(t, g)] +
                                    Y_DGGrid[ii][(t, g)] + Y_DGCur[ii][(t, g)] ==
                                    quicksum(x_fixed[l][(1, 3)] + (ii - 1) * x_fixed[l][(2, 3)] for l in RNGLoc))


                    # Assigned load decomposition
                    self.sub.addConstr(quicksum(Y_LH[ii][(h, t, g)] for h in RNGHouse) ==
                                    Eta_i * (Y_ESL[ii][(t, g)] + Y_DGL[ii][(t, g)] + Y_PVL[ii][(t, g)]) +
                                    Y_GridL[ii][(t, g)], name='')

                    for h in RNGHouse:
                        # Load decomposition
                        self.sub.addConstr(Y_LH[ii][(h, t, g)] + Y_LL[ii][(h, t, g)] +
                                        quicksum(Y_LT[ii][(t, to, g)] for to in range(t, T + 1)) ==
                                        L[ii][(h, t, g)], name='')

                    if t in DontTran:
                        # Don't allow transfer
                        self.sub.addConstrs(Y_LT[ii][(t, to, g)] == 0 for to in range(t, T + 1))
                    else:
                        # Max load transfer
                        self.sub.addConstr(TransMax * np.sum([L[ii][(h, t, g)] for h in RNGHouse]) -
                                           quicksum(Y_LT[ii][(t, to, g)] for to in range(t, T + 1)) >= 0, name='')

                    # Load transfer and E level
                    self.sub.addConstr(Y_E[ii][(t, g)] - quicksum(Y_LT[ii][(to, t, g)] for to in range(1, t)) >= 0,  name='')

                    # Prohibited transfer to self
                    self.sub.addConstr(Y_LT[ii][(t, t, g)] == 0, name='')

                    # ES charging/discharging constraints
                    self.sub.addConstr((UB0[1] + (ii - 1) * UB1[1]) * U_E[ii][(t, g)] -
                                       (Y_ESL[ii][(t, g)] + Y_ESGrid[ii][(t, g)]) >= 0, name='')
                    self.sub.addConstr((UB0[1] + (ii - 1) * UB1[1]) * (1 - U_E[ii][(t, g)]) -
                                       (Y_PVES[ii][(t, g)] + Y_GridES[ii][(t, g)] + Y_DGES[ii][(t, g)]) >= 0, name='')

                    self.sub.addConstr((UB0[1] + UB0[2] + UB0[3] + (ii - 1) * (UB1[1] + UB1[2] + UB1[3])) * U_G[ii][(t, g)] -
                                       (Y_ESGrid[ii][(t, g)] + Y_PVGrid[ii][(t, g)] + Y_DGGrid[ii][(t, g)]) >= 0,
                                       name='')
                    self.sub.addConstr((UB0[1] + UB0[2] + UB0[3] + (ii - 1) * (UB1[1] + UB1[2] + UB1[3])) * (1 - U_G[ii][(t, g)]) -
                                       (Y_GridES[ii][(t, g)] + Y_GridL[ii][(t, g)]) >= 0, name='')

                    # Prohibited transaction with the grid during outage
                    if Out_Time[g] != 0:
                        for ot in Out_Time[g]:
                            self.sub.addConstr(Y_GridL[ii][(ot, g)] + Y_GridES[ii][(ot, g)] == 0, name='')
                            self.sub.addConstr(Y_PVGrid[ii][(ot, g)] + Y_ESGrid[ii][(ot, g)] +
                                            Y_DGGrid[ii][(ot, g)] == 0, name='')

                for t in RNGTimeMinus:
                    # Balance of power flow
                    self.sub.addConstr(Y_E[ii][(t + 1, g)] -
                                       (Y_E[ii][(t, g)] +
                                        ES_gamma * (Y_PVES[ii][(t, g)] + Y_DGES[ii][(t, g)] + Eta_c * Y_GridES[ii][(t, g)]) -
                                        (Eta_i / ES_gamma) * (Y_ESL[ii][(t, g)] + Y_ESGrid[ii][(t, g)])
                                        ) == 0, name='')

        '''Get T, W, r and save it'''
        TMatrix, WMatrix, rMatrix = self.GetSaveTWr()

        '''Saving Bounds'''
        if True:
            '''Upper bounds'''
            upper_bounds = {}

            Binary_U_bound = {ii: {tg: -1 for tg in Y_tg} for ii in RNGSta}
            ES_U_bound = {ii: {tg: -(UB0[1] + (ii - 1) * UB1[1]) for tg in Y_tg} for ii in RNGSta}
            PV_U_bound = {ii: {tg: -(UB0[2] + (ii - 1) * UB1[2]) for tg in Y_tg} for ii in RNGSta}
            DG_U_bound = {ii: {tg: -(UB0[3] + (ii - 1) * UB1[3]) for tg in Y_tg} for ii in RNGSta}
            L_total_U_bound = {ii: {tg: -sum(L[ii][(h, tg[0], tg[1])] for h in RNGHouse) for tg in Y_tg} for ii in RNGSta}
            L_U_bound = {ii: {htg: -L[ii][(htg[0], htg[1], htg[2])] for htg in Y_htg} for ii in RNGSta}
            L_t_U_bound = {ii: {ttg: -sum(L[ii][(h, ttg[0], ttg[2])] for h in RNGHouse) for ttg in Y_ttg} for ii in RNGSta}

            # U_E
            upper_bounds['U_E'] = Binary_U_bound

            # U_G
            upper_bounds['U_G'] = Binary_U_bound

            # Y_PVES
            upper_bounds['Y_PVES'] = PV_U_bound

            # Y_DGES
            upper_bounds['Y_DGES'] = DG_U_bound

            # Y_GridES
            upper_bounds['Y_GridES'] = ES_U_bound

            # Y_PVL
            upper_bounds['Y_PVL'] = PV_U_bound

            # Y_DGL
            upper_bounds['Y_DGL'] = DG_U_bound

            # Y_ESL
            upper_bounds['Y_ESL'] = ES_U_bound

            # Y_GridL
            upper_bounds['Y_GridL'] = L_total_U_bound

            # Y_PVCur
            upper_bounds['Y_PVCur'] = PV_U_bound

            # Y_DGCur
            upper_bounds['Y_DGL'] = DG_U_bound

            # Y_PVGrid
            upper_bounds['Y_PVGrid'] = PV_U_bound

            # Y_DGGrid
            upper_bounds['Y_DGCur'] = DG_U_bound

            # Y_ESGrid
            upper_bounds['Y_ESGrid'] = ES_U_bound

            # E
            upper_bounds['Y_E'] = ES_U_bound

            # Y_LH
            upper_bounds['Y_LH'] = L_U_bound

            # Y_LL
            upper_bounds['Y_LL'] = L_U_bound

            # Y_LT
            upper_bounds['Y_LT'] = L_t_U_bound

            '''End of upper bounds'''

            '''Lower bounds'''
            lower_bounds = {}
            Binary_L_bound = {ii: {tg: 0 for tg in Y_tg} for ii in RNGSta}
            ES_L_bound = {ii: {tg: LB0[1] + (ii - 1) * LB1[1] for tg in Y_tg} for ii in RNGSta}
            PV_L_bound = {ii: {tg: 0 for tg in Y_tg} for ii in RNGSta}
            DG_L_bound = {ii: {tg: 0 for tg in Y_tg} for ii in RNGSta}
            L_total_L_bound = {ii: {tg: 0 for tg in Y_tg} for ii in RNGSta}
            L_L_bound = {ii: {htg: 0 for htg in Y_htg} for ii in RNGSta}
            L_t_L_bound = {ii: {ttg: 0 for ttg in Y_ttg} for ii in RNGSta}
            # U_E
            lower_bounds['U_E'] = Binary_L_bound

            # U_G
            lower_bounds['U_G'] = Binary_L_bound

            # Y_PVES
            lower_bounds['Y_PVES'] = PV_L_bound

            # Y_DGES
            lower_bounds['Y_DGES'] = DG_L_bound

            # Y_GridES
            lower_bounds['Y_GridES'] = ES_L_bound

            # Y_PVL
            lower_bounds['Y_PVL'] = PV_L_bound

            # Y_DGL
            lower_bounds['Y_DGL'] = DG_L_bound

            # Y_ESL
            lower_bounds['Y_ESL'] = ES_L_bound

            # Y_GridL
            lower_bounds['Y_GridL'] = L_total_L_bound

            # Y_PVCur
            lower_bounds['Y_PVCur'] = PV_L_bound

            # Y_DGCur
            lower_bounds['Y_DGL'] = DG_L_bound

            # Y_PVGrid
            lower_bounds['Y_PVGrid'] = PV_L_bound

            # Y_DGGrid
            lower_bounds['Y_DGCur'] = DG_L_bound

            # Y_ESGrid
            lower_bounds['Y_ESGrid'] = ES_L_bound

            # E
            lower_bounds['Y_E'] = ES_L_bound

            # Y_LH
            lower_bounds['Y_LH'] = L_L_bound

            # Y_LL
            lower_bounds['Y_LL'] = L_L_bound

            # Y_LT
            lower_bounds['Y_LT'] = L_t_L_bound
            '''End of lower bounds'''

        '''Costs'''
        if True:
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
                                GenerPrice * quicksum(x_fixed[l][(1, 2)] for l in RNGLoc) * PV[(t, g)] - \
                                LoadPrice * quicksum(Y_LH[ii][(h, t, g)] for h in RNGHouse)

                        # DRP cost
                        Costs += TransPrice * quicksum(Y_LT[ii][(to, t, g)] for to in RNGTime)

            total_cost = Probs[scen] * GenPar * Costs
            self.sub.setObjective(total_cost, sense=GRB.MINIMIZE)
            self.sub.update()

        '''Save model data'''
        if True:
            self.sub.write(f'Models/Sub{scen}.mps')
            self.x_fixed = x_fixed
            with open(f'Models/Sub{scen}.pkl', 'wb') as f:
                pickle.dump([TMatrix, WMatrix, rMatrix, upper_bounds, lower_bounds], f)
            f.close()
        print(self.sub)
    def GetSaveTWr(self):
        Coefs = IndexUp(self.sub.getA().todok())  # dictionary that indexed up
        # Increase indexes by one to avoid 0 index
        m = np.max([key[0] for key in Coefs.keys()])  # m only covers the count of constraints that we need
        T_sparse = {}
        W_sparse = copy.copy(Coefs)
        r_sparse = {}
        ConstrList = self.sub.getConstrs()
        for row in range(1, m + 1):
            for xkey in self.X_keys:
                if (row, xkey) in Coefs:
                    T_sparse[(row, xkey)] = Coefs[(row, xkey)]
                    del W_sparse[(row, xkey)]

            r_sparse[row] = ConstrList[row - 1].rhs

        # Note that T, W, r are spares matrices in a dictionary format.
        return T_sparse, W_sparse, r_sparse

    def FixXSolve(self, x):
        for ii in RNGSta:
            for d in RNGDvc:
                self.sub.addConstr(self.x_fixed[(ii, d)] == x[(ii, d)], name=f'XFix[{ii}, {d}]')

        self.sub.optimize()
        FixXObj = self.sub.ObjVal

        for ii in RNGSta:
            for d in RNGDvc:
                self.sub.remove(self.sub.getConstrByName(f'XFix[{ii}, {d}]'))
        self.sub.update()
        return FixXObj

    def GetIndicesKeys(self):
        return self.XY_keys

if __name__ == '__main__':
    mp = MasterPro()
    sp1 = SubProb(1)
    sp2 = SubProb(2)
    Xkeys, Xindcies = mp.GetIndicesKeys()
    XYkeys = sp1.GetIndicesKeys()
    with open(f'Models/Indices.pkl', 'wb') as f:
        pickle.dump([Xkeys, Xindcies, XYkeys], f)
    f.close()



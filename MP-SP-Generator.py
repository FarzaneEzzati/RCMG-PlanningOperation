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
T = 10    # count of hours per week
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
C = {1: 260, 2: 244, 3: 50}  # order is: [ES, PV, DG]
CO = {i: C[i] * (PA_Factor + Operational_Rate) for i in (1, 2, 3)}

UB1 = {1: 100.1, 2: 100.1, 3: 60.1}
LB1 = {1: 1, 2: 1, 3: 1}
UB2 = {1: 50.1, 2: 40.1, 3: 30.1}
LB2 = {1: 0, 2: 0, 3: 0}

alpha, beta, zeta = 0.5, 0.2, 0.9
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

X_ild = [(ii, l, d) for ii in RNGSta for l in RNGLoc for d in RNGDvc]
X_id = [(ii, d) for ii in RNGSta for d in RNGDvc]
Y_itg = [(ii, t, g)
        for ii in RNGSta for t in RNGTime for g in RNGMonth]
Y_ihtg = [(ii, h, t, g)
         for ii in RNGSta for h in RNGHouse for t in RNGTime for g in RNGMonth]
Y_ittg = [(ii, t, to, g)
         for ii in RNGSta for t in RNGTime for to in RNGTime for g in RNGMonth]

Y_itgs = [(ii, t, g, s)
        for ii in RNGSta for t in RNGTime for g in RNGMonth for s in RNGScen]
Y_ihtgs = [(ii, h, t, g, s)
         for ii in RNGSta for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen]
Y_ittgs = [(ii, t, to, g, s)
         for ii in RNGSta for t in RNGTime for to in RNGTime for g in RNGMonth for s in RNGScen]

# Save all ranges for the algorithm part
with open('Data/Ranges.pkl', 'wb') as handle:
    pickle.dump([RNGLoc, RNGDvc, RNGTime, RNGMonth, RNGHouse, RNGScen, RNGSta, Y_itg, Y_ihtg, Y_ittg], handle)
handle.close()

eta_M = -1000000


class MasterPro:
    def __init__(self):
        self.master = gp.Model('MasterProb', env=env)
        '''Investment & Reinvestment variables'''
        X = {}
        for l in RNGLoc:
            for d in RNGDvc:
                X[(1, l, d)] = self.master.addVar(lb=LB1[d], ub=UB1[d], name=f'X[1,{l},{d}]')
        for l in RNGLoc:
            for d in RNGDvc:
                X[(2, l, d)] = self.master.addVar(lb=LB2[d], ub=UB2[d], name=f'X[2,{l},{d}]')
        self.X_keys = range(1, LCount * (2 * DVCCount) + 1)  # starts from 1
        self.X_indcies = X_ild
        Capital = 0
        for l in RNGLoc:
            # Investment constraint
            self.master.addConstr(quicksum(X[(1, l, d)] * C[d] for d in RNGDvc) <= Budget1, name='IB')
            # ReInvestment constraint
            self.master.addConstr(quicksum(X[(2, l, d)] * C[d] for d in RNGDvc) <= Budget2, name='RIB')

            # Formulate capital cost
            Capital += quicksum((X[(1, l, d)] + X[(2, l, d)]) * CO[d] for d in RNGDvc)


        # Get A and b to save it
        self.master.update()
        A, b = self.GetAb()  # Checked: correct

        # Save upper and lower bounds
        upper_bounds = {}
        lower_bounds = {}
        for ild in X_ild:
            if ild[0] == 1:
                upper_bounds[X[ild].index+1] = UB1[ild[2]]
                lower_bounds[X[ild].index+1] = LB1[ild[2]]
            else:
                upper_bounds[X[ild].index+1] = UB2[ild[2]]
                lower_bounds[X[ild].index+1] = LB2[ild[2]]
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


class SubProb:
    def __init__(self, scen):
        self.sub = gp.Model(env=env)

        '''X vars to keep the model accessible for TMatrix'''
        if True:
            x_fixed = self.sub.addVars(X_ild, name='X')
            self.X_keys = range(1, LCount * (2 * DVCCount) + 1)  # starts from 1
        '''Scheduling variables'''
        if True:
            U_E = self.sub.addVars(Y_itg, ub=1, name=f'U_E')  # Charge/discharge binary
            U_G = self.sub.addVars(Y_itg, ub=1, name=f'U_G')  # Import/export binary
            Y_PVES = self.sub.addVars(Y_itg, name=f'Y_PVES')  # PV to ES (1 before reinvestment, 2 after)
            Y_DGES = self.sub.addVars(Y_itg, name=f'Y_DGES')  # DE to ES
            Y_GridES = self.sub.addVars(Y_itg, name=f'Y_GridES')   # Grid to ES
            Y_PVL = self.sub.addVars(Y_itg, name=f'Y_PV')  # Pv to L
            Y_DGL = self.sub.addVars(Y_itg, name=f'Y_DGL')  # Dg to L
            Y_ESL = self.sub.addVars(Y_itg, name=f'Y_ESL')  # ES to L
            Y_GridL = self.sub.addVars(Y_itg, name=f'Y_GridL')  # Grid to L
            Y_PVCur = self.sub.addVars(Y_itg, name=f'Y_PVCur')  # PV Curtailed
            Y_DGCur = self.sub.addVars(Y_itg, name=f'Y_DGCur')  # DG curtailed
            Y_PVGrid = self.sub.addVars(Y_itg, name=f'Y_DGGrid')  # PV to Grid
            Y_DGGrid = self.sub.addVars(Y_itg, name=f'Y_PVGrid')  # Dg to Grid
            Y_ESGrid = self.sub.addVars(Y_itg, name=f'Y_ESGrid')  # ES to Grid
            Y_E = self.sub.addVars(Y_itg, name=f'Y_E')  # ES level of energy
            Y_LH = self.sub.addVars(Y_ihtg, name=f'Y_LH')  # Load served
            Y_LL = self.sub.addVars(Y_ihtg, name=f'Y_LL')  # Load lost
            Y_LT = self.sub.addVars(Y_ittg, name=f'Y_LT')  # Load transferred
            self.sub.update()
            '''Saving all vars keys in W and T matrix'''
            self.Y_keys = range(self.X_keys[-1] + 1, len(self.sub.getVars()) + 1)
            '''Obtain the index for integer variables in the second stage'''
            self.int_var_keys = []
            for var in self.sub.getVars():
                if ('U_E' in var.VarName, 'U_G' in var.VarName) != (False, False):
                    self.int_var_keys.append(var.index + 1)
        '''Specify Load Demand, PV, Outage Duration for the scenario s'''
        if True:
            # Define the load profiles and PV profiles
            L = {(ii, h, t, g): 10 * Load[h - 1][f'Month {g}'].iloc[t - 1]
                  for ii in RNGSta for h in RNGHouse for t in RNGTime for g in RNGMonth}

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
                self.sub.addConstr(Y_E[(ii, 1, g)] == SOC_UB * quicksum(x_fixed[(1, l, 1)] + (ii - 1) * x_fixed[(2, l, 1)] for l in RNGLoc))
                # Only use the summation of ES capacity before/after reinvestment for rhs update in ABC BB-D

                for t in RNGTime:
                    # Limits on energy level in ES
                    self.sub.addConstr(Y_E[(ii, t, g)] >= SOC_LB * quicksum(x_fixed[(1, l, 1)] + (ii - 1) * x_fixed[(2, l, 1)] for l in RNGLoc))


                    self.sub.addConstr(-Y_E[(ii, t, g)] >= -SOC_UB * quicksum(x_fixed[(1, l, 1)] + (ii - 1) * x_fixed[(2, l, 1)] for l in RNGLoc))

                    # PV power decomposition
                    self.sub.addConstr((Y_PVL[(ii, t, g)] + Y_PVES[(ii, t, g)] +
                                        Y_PVCur[(ii, t, g)] + Y_PVGrid[(ii, t, g)]) ==
                                        PV[(t, g)] * quicksum(x_fixed[(1, l, 2)] + (ii - 1) * x_fixed[(2, l, 2)] for l in RNGLoc))

                    # DG power decomposition
                    self.sub.addConstr(Y_DGL[(ii, t, g)] + Y_DGES[(ii, t, g)] +
                                    Y_DGGrid[(ii, t, g)] + Y_DGCur[(ii, t, g)] ==
                                    quicksum(x_fixed[(1, l, 3)] + (ii - 1) * x_fixed[(2, l, 3)] for l in RNGLoc))


                    # Assigned load decomposition
                    self.sub.addConstr(quicksum(Y_LH[(ii, h, t, g)] for h in RNGHouse) ==
                                    Eta_i * (Y_ESL[(ii, t, g)] + Y_DGL[(ii, t, g)] + Y_PVL[(ii, t, g)]) +
                                    Y_GridL[(ii, t, g)], name='')

                    for h in RNGHouse:
                        # Load decomposition
                        self.sub.addConstr(Y_LH[(ii, h, t, g)] + Y_LL[(ii, h, t, g)] +
                                        quicksum(Y_LT[(ii, t, to, g)] for to in range(t, T + 1)) ==
                                        L[(ii, h, t, g)], name='')

                    if t in DontTran:
                        # Don't allow transfer
                        self.sub.addConstrs(Y_LT[(ii, t, to, g)] == 0 for to in range(t, T + 1))
                    else:
                        # Max load transfer
                        self.sub.addConstr(TransMax * np.sum([L[(ii, h, t, g)] for h in RNGHouse]) -
                                           quicksum(Y_LT[(ii, t, to, g)] for to in range(t, T + 1)) >= 0, name='')

                    # Load transfer and E level
                    self.sub.addConstr(Y_E[(ii, t, g)] - quicksum(Y_LT[(ii, to, t, g)] for to in range(1, t)) >= 0,  name='')

                    # Prohibited transfer to self
                    self.sub.addConstr(Y_LT[(ii, t, t, g)] == 0, name='')

                    # ES charging/discharging constraints
                    self.sub.addConstr((UB1[1] + (ii - 1) * UB2[1]) * U_E[(ii, t, g)] -
                                       (Y_ESL[(ii, t, g)] + Y_ESGrid[(ii, t, g)]) >= 0, name='')
                    self.sub.addConstr((UB1[1] + (ii - 1) * UB2[1]) * (1 - U_E[(ii, t, g)]) -
                                       (Y_PVES[(ii, t, g)] + Y_GridES[(ii, t, g)] + Y_DGES[(ii, t, g)]) >= 0, name='')

                    self.sub.addConstr((UB1[1] + UB1[2] + UB1[3] + (ii - 1) * (UB2[1] + UB2[2] + UB2[3])) * U_G[(ii, t, g)] -
                                       (Y_ESGrid[(ii, t, g)] + Y_PVGrid[(ii, t, g)] + Y_DGGrid[(ii, t, g)]) >= 0,
                                       name='')
                    self.sub.addConstr((UB1[1] + UB1[2] + UB1[3] + (ii - 1) * (UB2[1] + UB2[2] + UB2[3])) * (1 - U_G[(ii, t, g)]) -
                                       (Y_GridES[(ii, t, g)] + Y_GridL[(ii, t, g)]) >= 0, name='')

                    # Prohibited transaction with the grid during outage
                    if Out_Time[g] != 0:
                        for ot in Out_Time[g]:
                            self.sub.addConstr(Y_GridL[(ii, ot, g)] + Y_GridES[(ii, ot, g)] == 0, name='')
                            self.sub.addConstr(Y_PVGrid[(ii, ot, g)] + Y_ESGrid[(ii, ot, g)] +
                                            Y_DGGrid[(ii, ot, g)] == 0, name='')

                for t in RNGTimeMinus:
                    # Balance of power flow
                    self.sub.addConstr(Y_E[(ii, t + 1, g)] ==
                                       Y_E[(ii, t, g)] +
                                        ES_gamma * (Y_PVES[(ii, t, g)] + Y_DGES[(ii, t, g)] + Eta_c * Y_GridES[(ii, t, g)]) -
                                        (Eta_i / ES_gamma) * (Y_ESL[(ii, t, g)] + Y_ESGrid[(ii, t, g)]), name='')
        '''Saving Bounds'''
        if True:
            '''Upper bounds'''
            Binary_U_bound = {itg: 1 for itg in Y_itg}
            ES_U_bound = {itg: UB1[1] + (itg[0] - 1) * UB2[1] for itg in Y_itg}
            PV_U_bound = {itg: UB1[2] + (itg[0] - 1) * UB2[2] for itg in Y_itg}
            DG_U_bound = {itg: UB1[3] + (itg[0] - 1) * UB2[3] for itg in Y_itg}
            L_total_U_bound = {itg: sum(L[(itg[0], h, itg[1], itg[2])] for h in RNGHouse) for itg in Y_itg}
            L_U_bound = {ihtg: L[ihtg] for ihtg in Y_ihtg}
            L_t_U_bound = {ittg: sum(L[(ittg[0], h, ittg[1], ittg[3])] for h in RNGHouse) for ittg in Y_ittg}
            upper_bounds = {}
            last_key = self.X_keys[-1]
            # U_E
            for value in Binary_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # U_G
            for value in Binary_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_PVES
            for value in PV_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_DGES
            for value in DG_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_GridES
            for value in ES_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_PVL
            for value in PV_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_DGL
            for value in DG_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_ESL
            for value in ES_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_GridL
            for value in L_total_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_PVCur
            for value in PV_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_DGCur
            for value in DG_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_PVGrid
            for value in PV_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_DGGrid
            for value in DG_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_ESGrid
            for value in ES_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # E
            for value in ES_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_LH
            for value in L_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_LL
            for value in L_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value

            # Y_LT
            for value in L_t_U_bound.values():
                last_key += 1
                upper_bounds[last_key] = value
            '''End of upper bounds'''

            '''Lower bounds'''
            Binary_L_bound = {itg: 0 for itg in Y_itg}
            ES_L_bound = {itg: 0 for itg in Y_itg}
            PV_L_bound = {itg: 0 for itg in Y_itg}
            DG_L_bound = {itg: 0 for itg in Y_itg}
            L_total_L_bound = {itg: 0 for itg in Y_itg}
            L_L_bound = {ihtg: 0 for ihtg in Y_ihtg}
            L_t_L_bound = {ittg: 0 for ittg in Y_ittg}

            lower_bounds = {}
            last_key = self.X_keys[-1]
            # U_E
            for value in Binary_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # U_G
            for value in Binary_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_PVES
            for value in PV_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_DGES
            for value in DG_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_GridES
            for value in ES_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_PVL
            for value in PV_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_DGL
            for value in DG_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_ESL
            for value in ES_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_GridL
            for value in L_total_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_PVCur
            for value in PV_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_DGCur
            for value in DG_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_PVGrid
            for value in PV_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_DGGrid
            for value in DG_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_ESGrid
            for value in ES_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # E
            for value in ES_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_LH
            for value in L_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_LL
            for value in L_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value

            # Y_LT
            for value in L_t_L_bound.values():
                last_key += 1
                lower_bounds[last_key] = value
            '''End of lower bounds'''
        '''Costs'''
        if True:
            Costs = 0
            for ii in RNGSta:
                for g in RNGMonth:
                    for t in RNGTime:
                        # Curtailment cost
                        Costs += PVCurPrice * (Y_PVCur[(ii, t, g)] + Y_DGCur[(ii, t, g)])
                        # Losing load cost
                        Costs += quicksum(VoLL[h] * Y_LL[(ii, h, t, g)] for h in RNGHouse)
                        # DG cost
                        Costs += DGEffic * (Y_DGL[(ii, t, g)] + Y_DGGrid[(ii, t, g)] +
                                           Y_DGCur[(ii, t, g)] + Y_DGES[(ii, t, g)])
                        # Import/Export cost

                        Costs += GridPlus * Y_GridES[(ii, t, g)] - \
                                GridMinus * (Y_PVGrid[(ii, t, g)] + Y_ESGrid[(ii, t, g)] + Y_DGGrid[(ii, t, g)]) - \
                                GenerPrice * quicksum(x_fixed[(1, l, 2)] for l in RNGLoc) * PV[(t, g)] - \
                                LoadPrice * (Y_ESL[(ii, t, g)] + Y_DGL[(ii, t, g)] + Y_PVL[(ii, t, g)])

                        # DRP cost
                        Costs += TransPrice * quicksum(Y_LT[(ii, to, t, g)] for to in RNGTime)

            total_cost = Probs[scen] * GenPar * Costs
            self.sub.setObjective(total_cost, sense=GRB.MINIMIZE)
            self.sub.update()
            '''Get T, W, r and save it'''
            TMatrix, WMatrix, rMatrix = self.GetTWr()
        '''Save model data'''
        if True:
            self.sub.write(f'Models/Sub{scen}.mps')
            with open(f'Models/Sub{scen}.pkl', 'wb') as f:
                pickle.dump([TMatrix, WMatrix, rMatrix, upper_bounds, lower_bounds], f)
            f.close()
    def GetTWr(self):
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


class DetModel:
    def __init__(self):
        self.real = gp.Model(env=env)
        X = {}
        for l in RNGLoc:
            for d in RNGDvc:
                X[(1, l, d)] = self.real.addVar(lb=LB1[d], ub=UB1[d], name=f'X[1,{l},{d}]')
        for l in RNGLoc:
            for d in RNGDvc:
                X[(2, l, d)] = self.real.addVar(lb=LB2[d], ub=UB2[d], name=f'X[2,{l},{d}]')

        Capital = 0
        for l in RNGLoc:
            # Investment constraint
            self.real.addConstr(quicksum(X[(1, l, d)] * C[d] for d in RNGDvc) <= Budget1, name='IB')
            # ReInvestment constraint
            self.real.addConstr(quicksum(X[(2, l, d)] * C[d] for d in RNGDvc) <= Budget2, name='RIB')

            # Formulate capital cost
            Capital += quicksum((X[(1, l, d)] + X[(2, l, d)]) * CO[d] for d in RNGDvc)

        '''Scheduling variables'''
        if True:
            U_E = self.real.addVars(Y_itgs, vtype=GRB.BINARY, name=f'U_E')  # Charge/discharge binary
            U_G = self.real.addVars(Y_itgs, vtype=GRB.BINARY, name=f'U_G')  # Import/export binary
            Y_PVES = self.real.addVars(Y_itgs, name=f'Y_PVES')  # PV to ES (1 before reinvestment, 2 after)
            Y_DGES = self.real.addVars(Y_itgs, name=f'Y_DGES')  # DE to ES
            Y_GridES = self.real.addVars(Y_itgs, name=f'Y_GridES')  # Grid to ES
            Y_PVL = self.real.addVars(Y_itgs, name=f'Y_PV')  # Pv to L
            Y_DGL = self.real.addVars(Y_itgs, name=f'Y_DGL')  # Dg to L
            Y_ESL = self.real.addVars(Y_itgs, name=f'Y_ESL')  # ES to L
            Y_GridL = self.real.addVars(Y_itgs, name=f'Y_GridL')  # Grid to L
            Y_PVCur = self.real.addVars(Y_itgs, name=f'Y_PVCur')  # PV Curtailed
            Y_DGCur = self.real.addVars(Y_itgs, name=f'Y_DGCur')  # DG curtailed
            Y_PVGrid = self.real.addVars(Y_itgs, name=f'Y_DGGrid')  # PV to Grid
            Y_DGGrid = self.real.addVars(Y_itgs, name=f'Y_PVGrid')  # Dg to Grid
            Y_ESGrid = self.real.addVars(Y_itgs, name=f'Y_ESGrid')  # ES to Grid
            Y_E = self.real.addVars(Y_itgs, name=f'Y_E')  # ES level of energy
            Y_LH = self.real.addVars(Y_ihtgs, name=f'Y_LH')  # Load served
            Y_LL = self.real.addVars(Y_ihtgs, name=f'Y_LL')  # Load lost
            Y_LT = self.real.addVars(Y_ittgs, name=f'Y_LT')  # Load transferred
        '''Specify Load Demand, PV, Outage Duration for the scenario s'''
        if True:
            # Define the load profiles and PV profiles
            # Define the load profiles and PV profiles
            # Define the load profiles and PV profiles
            L = {(ii, h, t, g, s): 10 * Load[h - 1][f'Month {g}'].iloc[t - 1]
                  for ii in RNGSta for h in RNGHouse for t in RNGTime for g in RNGMonth for s in RNGScen}

            PV = {(t, g, s): PV_Unit[f'Month {g}'].iloc[t - 1]
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
                    self.real.addConstr(Y_E[(ii, 1, g, s)] == SOC_UB * quicksum(
                        X[(1, l, 1)] + (ii - 1) * X[(2, l, 1)] for l in RNGLoc))
                    # Only use the summation of ES capacity before/after reinvestment for rhs update in ABC BB-D

                    for t in RNGTime:
                        # Limits on energy level in ES
                        self.real.addConstr(Y_E[(ii, t, g, s)] >= SOC_LB * quicksum(
                            X[(1, l, 1)] + (ii - 1) * X[(2, l, 1)] for l in RNGLoc))

                        self.real.addConstr(-Y_E[(ii, t, g, s)] >= -SOC_UB * quicksum(
                            X[(1, l, 1)] + (ii - 1) * X[(2, l, 1)] for l in RNGLoc))

                        # PV power decomposition
                        self.real.addConstr((Y_PVL[(ii, t, g, s)] + Y_PVES[(ii, t, g, s)] +
                                            Y_PVCur[(ii, t, g, s)] + Y_PVGrid[(ii, t, g, s)]) ==
                                           PV[(t, g, s)] * quicksum(
                            X[(1, l, 2)] + (ii - 1) * X[(2, l, 2)] for l in RNGLoc))

                        # DG power decomposition
                        self.real.addConstr(Y_DGL[(ii, t, g, s)] + Y_DGES[(ii, t, g, s)] +
                                           Y_DGGrid[(ii, t, g, s)] + Y_DGCur[(ii, t, g, s)] ==
                                           quicksum(X[(1, l, 3)] + (ii - 1) * X[(2, l, 3)] for l in RNGLoc))

                        # Assigned load decomposition
                        self.real.addConstr(quicksum(Y_LH[(ii, h, t, g, s)] for h in RNGHouse) ==
                                           Eta_i * (Y_ESL[(ii, t, g, s)] + Y_DGL[(ii, t, g, s)] + Y_PVL[(ii, t, g, s)]) +
                                           Y_GridL[(ii, t, g, s)], name='')

                        for h in RNGHouse:
                            # Load decomposition
                            self.real.addConstr(Y_LH[(ii, h, t, g, s)] + Y_LL[(ii, h, t, g, s)] +
                                               quicksum(Y_LT[(ii, t, to, g, s)] for to in range(t, T + 1)) ==
                                               L[(ii, h, t, g, s)], name='')

                        if t in DontTran:
                            # Don't allow transfer
                            self.real.addConstrs(Y_LT[(ii, t, to, g, s)] == 0 for to in range(t, T + 1))
                        else:
                            # Max load transfer
                            self.real.addConstr(TransMax * np.sum([L[(ii, h, t, g, s)] for h in RNGHouse]) -
                                               quicksum(Y_LT[(ii, t, to, g, s)] for to in range(t, T + 1)) >= 0, name='')

                        # Load transfer and E level
                        self.real.addConstr(Y_E[(ii, t, g, s)] - quicksum(Y_LT[(ii, to, t, g, s)] for to in range(1, t)) >= 0, name='')

                        # Prohibited transfer to self
                        self.real.addConstr(Y_LT[(ii, t, t, g, s)] == 0, name='')

                        # ES charging/discharging constraints
                        self.real.addConstr((UB1[1] + (ii - 1) * UB2[1]) * U_E[(ii, t, g, s)] -
                                           (Y_ESL[(ii, t, g, s)] + Y_ESGrid[(ii, t, g, s)]) >= 0, name='')
                        self.real.addConstr((UB1[1] + (ii - 1) * UB2[1]) * (1 - U_E[(ii, t, g, s)]) -
                                           (Y_PVES[(ii, t, g, s)] + Y_GridES[(ii, t, g, s)] + Y_DGES[(ii, t, g, s)]) >= 0, name='')

                        self.real.addConstr(
                            (UB1[1] + UB1[2] + UB1[3] + (ii - 1) * (UB2[1] + UB2[2] + UB2[3])) * U_G[(ii, t, g, s)] -
                            (Y_ESGrid[(ii, t, g, s)] + Y_PVGrid[(ii, t, g, s)] + Y_DGGrid[(ii, t, g, s)]) >= 0,
                            name='')
                        self.real.addConstr(
                            (UB1[1] + UB1[2] + UB1[3] + (ii - 1) * (UB2[1] + UB2[2] + UB2[3])) * (1 - U_G[(ii, t, g, s)]) -
                            (Y_GridES[(ii, t, g, s)] + Y_GridL[(ii, t, g, s)]) >= 0, name='')

                        # Prohibited transaction with the grid during outage
                        if Out_Time[(g, s)] != 0:
                            for ot in Out_Time[(g, s)]:
                                self.real.addConstr(Y_GridL[(ii, ot, g, s)] + Y_GridES[(ii, ot, g, s)] == 0, name='')
                                self.real.addConstr(Y_PVGrid[(ii, ot, g, s)] + Y_ESGrid[(ii, ot, g, s)] +
                                                   Y_DGGrid[(ii, ot, g, s)] == 0, name='')

                    for t in RNGTimeMinus:
                        # Balance of power flow
                        self.real.addConstr(Y_E[(ii, t + 1, g, s)] ==
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
                            Costs += quicksum(VoLL[h] * Y_LL[(ii, h, t, g, s)] for h in RNGHouse)
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
            self.real.setObjective(Total_cost + Capital, sense=GRB.MINIMIZE)

        '''Save model data'''
        if True:
            self.real.write(f'Models/real.mps')


if __name__ == '__main__':
    mp = MasterPro()
    sp1 = SubProb(1)
    sp2 = SubProb(2)
    with open(f'Models/Indices.pkl', 'wb') as f:
        pickle.dump([mp.X_keys, mp.X_indcies, sp1.Y_keys, sp1.int_var_keys], f)
    f.close()
    real = DetModel()




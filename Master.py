import pickle
import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import quicksum, GRB
import warnings
from tqdm import tqdm
import gc  # garbage collecter
import psutil  # to monitor memory
from time import time
from Methods import *
from Data import ModelsData

class Master:
    def __init__(self, data: ModelsData):
        master_env = gp.Env()
        master_env.setParam('OutputFlag', 1)
        ############################## First Stage Model
        l_index = data.l_index
        d_index = data.d_index
        master = gp.Model('MasterProb', env=master_env)
        self.X_I = master.addVars(data.L, data.D, vtype=GRB.INTEGER, name='X')
        self.U_I = master.addVars(data.L, vtype=GRB.BINARY, name='U')
        self.eta = master.addVar(ub=float('inf'), lb=data.M, vtype=GRB.CONTINUOUS, name='eta')
        ##### Investment costs and constraint
        capital_cost = sum(data.F[l] * self.U_I[l] 
                           for l in l_index) +\
                       sum(self.X_I[l, d] * data.C[d] 
                           for l in l_index for d in d_index)
        operation_cost = sum(self.X_I[l, d] * data.O[d] 
                             for l in l_index for d in d_index)
        master.addConstr((1 - data.sv_subsidy_rate) * capital_cost <= data.Budget_I, name='budget')
        master.addConstrs(
            (self.X_I[l, d] <= data.device_ub[l, d] * self.U_I[l] for l in l_index for d in d_index), name='ub')
        #### Objective
        total_cost = (1 - data.sv_subsidy_rate) * capital_cost + \
                     data.N * data.to_year * operation_cost + \
                     self.eta
        master.setObjective(total_cost, sense=GRB.MINIMIZE)
        ##### Save master in mps + save data in pickle
        master.update()
        master.write('Models/Master.mps')
        self.model = master


sep_env = gp.Env()
sep_env.setParam('Method', 0)
sep_env.setParam('OutputFlag', 0)
sep_env.setParam('NumericFocus', 2)
class Separation:
    def __init__(self, data: ModelsData, scenario_index):
        ############################## Indices
        g_index = data.g_index
        l_index = data.l_index
        i_index = data.i_index
        t_index = data.t_index
        d_index = data.d_index
        I, G, T, L, D = data.I, data.G, data.T, data.L, data.D

        scenario_combo = data.Sc[scenario_index]

        sep = gp.Model(f'Separation {scenario_index}', env=sep_env)
        self.X_I = sep.addMVar((L, D), name=f'X_I')
        self.U_I = sep.addMVar(L, name='U_I')
        self.X_E = sep.addMVar((L, D), name=f'X_E')
        self.Y_PVES = sep.addMVar((I, G, T), name=f'Y_PVES')
        self.Y_DGES = sep.addMVar((I, G, T), name=f'Y_DGES')
        self.Y_GridES = sep.addMVar((I, G, T), name=f'Y_GridES')
        self.Y_PVL = sep.addMVar((I, G, T), name=f'Y_PVL')
        self.Y_DGL = sep.addMVar((I, G, T), name=f'Y_DGL')
        self.Y_ESL = sep.addMVar((I, G, T), name=f'Y_ESL')
        self.Y_GridL = sep.addMVar((I, G, T), name=f'Y_GridL')
        self.Y_PVCur = sep.addMVar((I, G, T), name=f'Y_PVCur')
        self.Y_DGCur = sep.addMVar((I, G, T), name=f'Y_DGCur')
        self.Y_PVGrid = sep.addMVar((I, G, T), name=f'Y_DGGrid')
        self.Y_DGGrid = sep.addMVar((I, G, T), name=f'Y_PVGrid')
        self.Y_ESGrid = sep.addMVar((I, G, T), name=f'Y_ESGrid')
        self.Y_E = sep.addMVar((I, G, T), name=f'Y_E')
        self.Y_LSh = sep.addMVar((I, G, T), name=f'Y_LSh')
        self.Y_I = sep.addMVar((I, G, T), name='Y_I')
        self.Y_LT = sep.addMVar((I, G, T, T), name=f'Y_LT')
        sep.update()

        ###### Prepare Load, Outage, and Load Growth
        self.Load = np.array([(1 + i * data.LG[scenario_combo[2]]/100) ** data.n *
                         data.LD[:, :, scenario_combo[1]] for i in i_index])
        self.Load = self.Load.reshape(I, 4, 3, T).mean(axis=2)  # [I, 4, T]
        Outage = []
        for g in g_index:
            outage_duration = max(0, int(data.OT[g, scenario_combo[0]]))
            if outage_duration != 0:
                if outage_duration >= 168 - (data.outage_start + 1):
                    Outage.append(range(data.outage_start, data.T))
                else:
                    Outage.append(range(data.outage_start, data.outage_start + outage_duration))
        ###### Expansion Constraints
        capital_cost_E = gp.quicksum(self.X_E[:, d].sum() * data.C[d] for d in d_index)
        operation_cost_E = gp.quicksum(self.X_E[:, d].sum() * data.O[d] for d in d_index)
        sep.addConstr(
            (1 - data.sv_subsidy_rate) * capital_cost_E <= data.Budget_E, name='Budget Limit')


        ###### Capacity limit
        sep.addConstrs(
            (self.X_E[l, d] + self.X_I[l, d] <= data.device_ub[l, d] for l in l_index for d in d_index),
              name='ES Location Limit')
        sep.addConstrs(
            (self.X_E[l, d] <= data.device_ub[l, d] * self.U_I[l] for l in l_index for d in d_index),
            name='install if open')


        ###### Operation Constraints
        sep_constrs = []
        after_degradation = (1 - data.degrad_rate) ** data.n
        for i in range(I):
            available_es = self.X_E[:, 0].sum() + after_degradation * self.X_I[:, 0].sum() if i != 0 \
                else self.X_I[:,0].sum()
            available_pv = self.X_I[:, 1].sum() + i * self.X_E[:, 1].sum()
            available_dg = self.X_I[:, 2].sum() + i * self.X_E[:, 2].sum()
            for g in g_index:
                sep_constrs.append(self.Y_E[i, g, data.outage_start] == data.es_soc_ub * available_es)
                sep_constrs.append(self.Y_E[i, g, 0] == 0.5 * available_es)
                for t in t_index:
                    if t in Outage[g]:
                        trans_to_t = gp.quicksum(self.Y_LT[i, g, to, t] for to in range(max(0, t - data.trans_period), t))
                    else:
                        trans_to_t = 0
                    trans_from_t = gp.quicksum(self.Y_LT[i, g, t, to] for to in range(t + 1, min(T, t + data.trans_period + 1)))
                    pv_dg_grid_to_es = self.Y_PVES[i, g, t] + self.Y_DGES[i, g, t] + self.Y_GridES[i, g, t]
                    es_to_grid_load = self.Y_ESL[i, g, t] + self.Y_ESGrid[i, g, t]
                    es_level_plus_trans = self.Y_E[i, g, t] + trans_to_t

                    sep_constrs.append(
                        es_level_plus_trans >= data.es_soc_lb * available_es)
                    sep_constrs.append(
                        es_level_plus_trans <= data.es_soc_ub * available_es)
                    sep_constrs.append(
                        pv_dg_grid_to_es <= (data.es_soc_ub - data.es_soc_lb) * available_es)
                    sep_constrs.append(
                        es_to_grid_load <= (data.es_soc_ub - data.es_soc_lb) * available_es)

                    pv_usage = self.Y_PVL[i, g, t] + self.Y_PVGrid[i, g, t] +\
                               self.Y_PVCur[i, g, t] + self.Y_PVES[i, g, t]
                    sep_constrs.append(pv_usage <= data.PV[g, t] * available_pv)

                    dg_usage = self.Y_DGL[i, g, t] + self.Y_DGGrid[i, g, t] + \
                               self.Y_DGES[i, g, t] + self.Y_DGCur[i, g, t]
                    sep_constrs.append(dg_usage <= data.dg_effi * available_dg)

                    means_of_load = self.Y_ESL[i, g, t] + self.Y_DGL[i, g, t] + \
                                    self.Y_PVL[i, g, t] + self.Y_GridL[i, g, t] +  \
                                    self.Y_LSh[i, g, t] + trans_to_t - trans_from_t
                    sep_constrs.append(means_of_load == self.Load[i, g, t])

                    # Outage and Trans times
                    sep_constrs.append(self.Y_LT[i, g, t, t] == 0)

                    if t not in Outage[g]:
                        sep_constrs.append(self.Y_LSh[i, g, t] == 0)
                        sep_constrs.append(trans_to_t == 0)
                    else:
                        grid_trade = self.Y_GridL[i, g, t] + self.Y_GridES[i, g, t] + \
                                     self.Y_PVGrid[i, g, t] + self.Y_ESGrid[i, g, t] + self.Y_DGGrid[i, g, t]
                        sep_constrs.append(grid_trade == 0)

                        sep_constrs.append(trans_from_t <= data.drp * self.Load[i, g, t])

                    if t in data.no_trans[g]:
                        sep_constrs.append(trans_from_t == 0)
                    else:
                        allowed_to_trans = self.Y_E[i, g, t] - data.es_soc_lb * available_es
                        sep_constrs.append(trans_from_t <= allowed_to_trans)
                    # Balance
                    if t < T-1:
                        sep_constrs.append(
                            self.Y_E[i, g, t + 1] == self.Y_E[i, g, t] -
                            trans_from_t + trans_to_t +
                            data.es_effi * pv_dg_grid_to_es -
                            data.es_eta * es_to_grid_load / data.es_effi)
                    sep_constrs.append(
                        self.Y_I[i, g, t] == trans_from_t)
        sep.addConstrs((c for c in sep_constrs))


        ###### Costs
        Costs = [0, 0]
        for i in range(I):
            # Cache repeated sums
            sum_LSh = self.Y_LSh[i].sum()
            sum_PVCur = self.Y_PVCur[i].sum()
            sum_DGCur = self.Y_DGCur[i].sum()
            sum_GridImport = self.Y_GridES[i].sum()
            sum_GridExport = self.Y_ESGrid[i].sum() + self.Y_PVGrid[i].sum() + self.Y_DGGrid[i].sum()
            sum_DG = self.Y_DGES[i].sum() + self.Y_DGL[i].sum() + self.Y_DGGrid[i].sum()
            sum_Load = self.Y_ESL[i].sum() + self.Y_PVL[i].sum() + self.Y_DGL[i].sum()
            sum_I = self.Y_I[i].sum()

            # Final expression assembly
            self.C_LSh = data.e_sheding * sum_LSh
            self.C_Cur = data.e_cur_pv * sum_PVCur + data.e_cur_dg * sum_DGCur
            self.C_IE = data.e_grid_import * sum_GridImport - data.e_grid_export * sum_GridExport
            self.C_F = data.dg_fule_cost * sum_DG
            self.C_R = -data.e_load * sum_Load
            self.C_IN = -data.e_drp * sum_I

            Costs[i] = self.C_LSh + self.C_Cur + self.C_IE + self.C_F + self.C_R + self.C_IN
        # Force scalars to be float
        total_cost = capital_cost_E + \
                     (data.N - data.n) * data.to_year * operation_cost_E + \
                     data.to_year * (data.n * Costs[0] + (data.N - data.n) * Costs[1])
        sep.setObjective(total_cost, sense=GRB.MINIMIZE)

        ###### Store A, T, r
        sep.update()
        self.AMatrix = sep.getA()  # [constraints, vars]
        Constrs = sep.getConstrs()
        self.TMatrix = self.AMatrix[:, 0:len(data.x_keys)]
        self.rVector = np.array(sep.getAttr("RHS", Constrs))
        self.model = sep

        sep.write(f'Models/sep {scenario_index}.lp')





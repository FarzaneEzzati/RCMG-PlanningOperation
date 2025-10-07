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

class Separation:
    def __init__(self, data: ModelsData, scenario_index, env):
        ############################## Indices
        g_index = data.g_index
        l_index = data.l_index
        i_index = data.i_index
        t_index = data.t_index
        d_index = data.d_index
        I, G, T, L, D = data.I, data.G, data.T, data.L, data.D

        scenario_combo = data.Sc[scenario_index]

        sep = gp.Model(f'Separation {scenario_index}', env=env)
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
        self.Y_LT = sep.addMVar((I, G, T), name=f'Y_LT')
        self.ES = sep.addMVar(I, name='ES')
        self.PV = sep.addMVar(I, name='PV')
        self.DG = sep.addMVar(I, name='DG')
        self.Load = sep.addMVar((I, G, T), name='Load')

        self.C_LSh = sep.addMVar(I, name='C_LSh')
        self.C_Cur = sep.addMVar(I, name='C_Cur')
        self.C_F = sep.addMVar(I, name='C_F')
        self.C_Import = sep.addMVar(I, name='C_Import')
        self.C_Export = sep.addMVar(I, name='C_Export')
        self.C_R = sep.addMVar(I, name='C_R')
        self.C_IN = sep.addMVar(I, name='C_IN')
        self.C_OB = sep.addMVar(I, name='C_OB')  # outage benefit
        self.capital_cost_E = sep.addVar(name='capital')
        self.operation_cost_E = sep.addVar(name='operation')

        sep.update()

        ###### Prepare Load, Outage, and Load Growth
        load_gr = data.LG[scenario_combo[2]]
        load_demand = np.array([data.LD[:, :, scenario_combo[1]] for i in i_index])
        self.load_demand = load_demand.reshape(I, 4, 3, T).mean(axis=2)  # [I, 4, T]
        self.Outage = []
        for g in g_index:
            outage_duration = max(0, int(data.OT[g, scenario_combo[0]]))
            if outage_duration != 0:
                if outage_duration >= 168 - (data.outage_start + 1):
                    self.Outage.append(range(data.outage_start, data.T))
                else:
                    self.Outage.append(range(data.outage_start, data.outage_start + outage_duration))
        ###### Expansion Constraints
        sep_constrs, constrs_names = [], []

        sep_constrs.append(-self.X_E[:, 1].sum() >= -4 * self.X_E[:, 0].sum())
        constrs_names.append('pv_if_es')

        sep_constrs.append(-self.X_E[:, 2].sum() >= -self.X_E[:, 1].sum())
        constrs_names.append('dg_if_pv')

        sep_constrs.append(-self.X_E[:, 2].sum() >= -self.X_E[:, 0].sum())
        constrs_names.append('dg_if_es')

        sep_constrs.append(self.capital_cost_E == gp.quicksum(self.X_E[:, d].sum() * data.C[d] for d in d_index))
        constrs_names.append('capital_cost')

        sep_constrs.append(self.operation_cost_E == gp.quicksum(self.X_E[:, d].sum() * data.O[d] for d in d_index))
        constrs_names.append('operation_cost')

        sep_constrs.append(-self.capital_cost_E >= -data.Budget_E)
        constrs_names.append('Budget_Limit')

        ###### Linking constraints
        self.linking_names = []
        for l in l_index:
            for d in d_index:
                name = f'device_ub_install[{l},{d}]'
                sep.addConstr(-self.X_E[l, d] - self.X_I[l, d] + 1.25 * data.device_ub[l, d] * self.U_I[l] >= 0, name=name)
                self.linking_names.append(name)

                name = f'minor_expansion[{l},{d}]'
                sep.addConstr(-self.X_E[l, d] + self.X_I[l, d] >= 0, name=name)
                self.linking_names.append(name)

        dg_ub = 1.25 * data.device_ub[0, 2]
        name = f'dg_ub_install'
        sep.addConstr(-self.X_I[:, 2].sum() - self.X_E[:, 2].sum() + dg_ub >= 0, name=name)
        self.linking_names.append(name)

        for i in range(I):
            degraded = 1 if i == 0 else (1 - data.degrad_rate) ** data.n

            name = f'available_es[{i}]'
            sep.addConstr(self.ES[i] == i * self.X_E[:, 0].sum() + degraded * self.X_I[:, 0].sum(), name=name)
            self.linking_names.append(name)

            name = f'available_pv[{i}]'
            sep.addConstr(self.PV[i] == i * self.X_E[:, 1].sum() + self.X_I[:, 1].sum(), name=name)
            self.linking_names.append(name)

            name =f'available_dg[{i}]'
            sep.addConstr(self.DG[i] == i * self.X_E[:, 2].sum() + degraded * self.X_I[:, 2].sum(), name=name)
            self.linking_names.append(name)
        sep.update()
        self.linking_constrs = [sep.getConstrByName(c) for c in self.linking_names]

        ###### Operation Constraints
        self.power_at_outage = [0, 0]
        for i in range(I):
            load_growth = (1 + i * data.n * load_gr)
            for g in g_index:
                sep_constrs.append(self.Y_E[i, g, data.outage_start] == data.es_soc_ub * self.ES[i])
                constrs_names.append('es_at_outage')

                sep_constrs.append(self.Y_E[i, g, 0] == data.es_soc_ub * self.ES[i])
                constrs_names.append('es_at_0')

                for t in t_index:
                    pv_dg_grid_to_es = self.Y_PVES[i, g, t] + self.Y_DGES[i, g, t] + self.Y_GridES[i, g, t]
                    es_to_grid_load = self.Y_ESL[i, g, t] + self.Y_ESGrid[i, g, t]

                    sep_constrs.append(
                        self.Y_E[i, g, t] >= data.es_soc_lb * self.ES[i])
                    constrs_names.append('min_level')

                    sep_constrs.append(
                        -self.Y_E[i, g, t] >= -data.es_soc_ub * self.ES[i])
                    constrs_names.append('max_level')

                    sep_constrs.append(
                        -pv_dg_grid_to_es >= -(data.es_soc_ub - data.es_soc_lb) * self.ES[i])
                    constrs_names.append('charge_max')

                    sep_constrs.append(
                        -es_to_grid_load >= -(data.es_soc_ub - data.es_soc_lb) * self.ES[i])
                    constrs_names.append('discharge_max')

                    name = 'pv_usage'
                    pv_usage = self.Y_PVL[i, g, t] + self.Y_PVGrid[i, g, t] +\
                               self.Y_PVCur[i, g, t] + self.Y_PVES[i, g, t]
                    sep.addConstr(-pv_usage + data.PV[g, t] * self.PV[i] >= 0, name=f'pv_usage[{i},{g},{t}]')
                    constrs_names.append('pv_usage')

                    dg_usage = self.Y_DGL[i, g, t] + self.Y_DGGrid[i, g, t] + \
                               self.Y_DGES[i, g, t] + self.Y_DGCur[i, g, t]
                    sep_constrs.append(-dg_usage + data.dg_effi * self.DG[i] >= 0)
                    constrs_names.append('dg_usage')

                    means_of_load = self.Y_ESL[i, g, t] + self.Y_LT[i, g, t] +\
                                    self.Y_DGL[i, g, t] + self.Y_PVL[i, g, t] + \
                                    self.Y_GridL[i, g, t] + self.Y_LSh[i, g, t]
                    sep_constrs.append(means_of_load == self.Load[i, g, t])
                    constrs_names.append('load_means')

                    sep.addConstr(-self.Y_LT[i, g, t] + data.drp * self.Load[i, g, t] >= 0,
                                  name=f'max_trans_load[{i},{g},{t}]')

                    if t not in self.Outage[g]:
                        sep_constrs.append(self.Y_LSh[i, g, t] == 0)
                        constrs_names.append('zero_shed')

                    else:
                        grid_trade = self.Y_GridL[i, g, t] + self.Y_GridES[i, g, t] + \
                                     self.Y_PVGrid[i, g, t] + self.Y_ESGrid[i, g, t] + \
                                     self.Y_DGGrid[i, g, t]
                        sep_constrs.append(grid_trade == 0)
                        constrs_names.append('zero_trade')

                        self.power_at_outage[i] += self.Y_ESL[i, g, t] + \
                                                   self.Y_PVL[i, g, t]

                    if t in data.no_trans[g]:
                        sep_constrs.append(self.Y_LT[i, g, t] == 0)
                        constrs_names.append('zero_trans_at_no_trans')
                    # Balance
                    if t < T-1:
                        sep_constrs.append(
                            self.Y_E[i, g, t + 1] == self.Y_E[i, g, t] +
                            data.es_effi * pv_dg_grid_to_es -
                            data.es_eta * es_to_grid_load / data.es_effi)
                        constrs_names.append('es_level_change')
                    # Load for sensitivity
                    sep.addConstr(self.Load[i, g, t] == load_growth * self.load_demand[i, g, t],
                                  name=f'load[{i},{g},{t}]')

        sep.addConstrs((c for c in sep_constrs))
        self.constrs_names = constrs_names
        ###### Costs
        self.Costs = [0, 0]
        for i in range(I):
            # Cache repeated sums
            sum_LSh = sum(self.Y_LSh[i, g] + 0.5 * self.Y_LT[i, g] for g in data.g_index)
            sum_PVCur = self.Y_PVCur[i].sum()
            sum_DGCur = self.Y_DGCur[i].sum()
            sum_GridImport = self.Y_GridES[i].sum()
            sum_GridExport = self.Y_PVGrid[i].sum()
            sum_DG = self.Y_DGES[i].sum() + self.Y_DGL[i].sum() + self.Y_DGGrid[i].sum()
            sum_Load = self.Y_ESL[i].sum() + self.Y_PVL[i].sum() + self.Y_DGL[i].sum()
            sum_DRP = self.Y_LT[i].sum()

            # Final expression assembly
            e_sheding = data.e_sheding * (1 + i * load_gr) ** data.n
            sep.addConstr(self.C_LSh[i] == gp.quicksum(e_sheding * sum_LSh), name=f'C_LSh{i}')
            sep.addConstr(self.C_Cur[i] == data.e_cur_pv * sum_PVCur + data.e_cur_dg * sum_DGCur, name=f'C_Cur{i}')
            sep.addConstr(self.C_Import[i] == data.e_grid_import * sum_GridImport, name=f'C_Import{i}')
            sep.addConstr(self.C_Export[i] == data.e_grid_export * sum_GridExport, name=f'C_Export{i}')
            sep.addConstr(self.C_F[i] == data.dg_fuel_cost * sum_DG, name=f'C_F{i}')
            sep.addConstr(self.C_R[i] == data.e_load * sum_Load, name=f'C_R{i}')
            sep.addConstr(self.C_IN[i] == data.e_drp * sum_DRP, name=f'C_In{i}')
            sep.addConstr(self.C_OB[i] == 0.2 * (1 + data.wtp) * data.e_grid_import * self.power_at_outage[i], name=f'C_OB{i}')

            self.Costs[i] = self.C_LSh[i] + self.C_Cur[i] + self.C_Import[i] + self.C_F[i] - \
                            self.C_R[i] - self.C_Export[i] - self.C_IN[i] - self.C_OB[i]

        # Force scalars to be float
        total_cost = self.capital_cost_E + \
                     data.to_year * (data.N - data.n) * self.operation_cost_E + \
                     data.to_year * (data.n * self.Costs[0] + (data.N - data.n) * self.Costs[1])
        sep.setObjective(total_cost, sense=GRB.MINIMIZE)

        sep.update()
        self.constrs = sep.getConstrs()
        self.model = sep
        #sep.write(f'Models/sep {scenario_index}.lp')


def save_solutions(f: Separation, scenario, mg_id, name):
    # Create a dictionary directly from f's variables and costs
    vars_to_save = {
        'y_pves': f.Y_PVES.x,
        'y_dges': f.Y_DGES.x,
        'y_grides': f.Y_GridES.x,
        'y_pvl': f.Y_PVL.x,
        'y_dgl': f.Y_DGL.x,
        'y_esl': f.Y_ESL.x,
        'y_gridl': f.Y_GridL.x,
        'y_pvcur': f.Y_PVCur.x,
        'y_dgcur': f.Y_DGCur.x,
        'y_pvgrid': f.Y_PVGrid.x,
        'y_dggrid': f.Y_DGGrid.x,
        'y_esgrid': f.Y_ESGrid.x,
        'y_e': f.Y_E.x,
        'y_lsh': f.Y_LSh.x,
        'y_lt': f.Y_LT.x,
        'load': f.Load.x
    }
    costs_to_save = {
        'capital': f.capital_cost_E.x,
        'operation': f.operation_cost_E.x,
        'c_lsh': f.C_LSh.x,
        'c_cur': f.C_Cur.x,
        'c_import': f.C_Import.x,
        'c_export': f.C_Import.x,
        'c_f': f.C_F.x,
        'c_r': f.C_R.x,
        'c_in': f.C_IN.x
    }
    pars_to_save = {
        'load': f.load_demand
    }
    # Save to a pickle file
    with open(name, 'wb') as file:
        pickle.dump([vars_to_save, costs_to_save, pars_to_save], file)





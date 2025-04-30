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

warnings.filterwarnings('ignore')

env = gp.Env()
env.setParam('OutputFlag', 0)


def build_models(mg_id):
    ############################## Read Data
    general_dict, location_dict, OT, LD, LG, PV, Sc, Pr = read_data(mg_id)
    ############################## Set Parameters
    I = 2
    T = 168  # count of hours
    L = len(location_dict.keys())  # count of locations
    D = 3  # count of devices
    G = 12  # count of months
    S = len(Pr)

    # Peak Hours
    summer_peak = [23 * i + 12 + j for i in range(6) for j in range(8)]  # 13-19PM
    winter_peak = np.concatenate(([23 * i + 5 + j for i in range(6) for j in range(5)],
                                  [23 * i + 17 + j for i in range(6) for j in range(5)]))  # 6-10AM, 6-10PM
    no_trans = {0: winter_peak, 1: winter_peak, 2: winter_peak,
                3: [], 4: [],
                5: summer_peak, 6: summer_peak, 7: summer_peak, 8: summer_peak,
                9: [], 10: [], 11: []}
    trans_period = 6
    ###### General Data
    Budget_I = general_dict['Budget_I']
    Budget_E = general_dict['Budget_E']
    C = np.array([general_dict[key] for key in ['C_es', 'C_pv', 'C_dg']])
    e_grid_import = general_dict['e_grid_import']
    e_grid_export = general_dict['e_grid_export']
    wtp = general_dict['wtp']
    sv = general_dict['sv']
    subsidy_rate = general_dict['subsidy_rate']
    total_subsidy = general_dict['total_subsidy']
    sv_subsidy_rate = sv * subsidy_rate
    ###### Device Efficiencies
    es_soc_ub, es_soc_lb = 0.9, 0.1
    es_eta = 0.9
    es_effi = 0.90
    dg_effi = 0.95
    dg_fuel_usage = general_dict['dg_efficiency(gal/kw)']
    fuel_price = general_dict['fuel($/gal)']
    dg_fule_cost = dg_fuel_usage * fuel_price  # Fuel cost of DG: $/kWh
    degrad_rate = 0.02
    ###### Electricity Prices
    e_load_rate = general_dict[
        'e_load_rate']  # The parameter specifying what percentage of electricity price be determined as price to sell power to households
    e_load = e_load_rate * e_grid_import
    e_sheding = (1 + wtp) * e_grid_import
    e_cur_pv = e_grid_export
    e_cur_dg = e_grid_export + dg_fule_cost
    ###### Demand Response
    drp = 0.25
    e_drp = drp * e_load
    ###### General Parameter
    outage_start = 15
    to_year = (365 / 7) / 12
    M = -1e10
    N = 20  # Life Time
    n = 10  # Expansion year
    labor_rate = 0.12
    operation_rate = 0.1
    O = C * operation_rate  # operational cost
    C = C * (1 + labor_rate)  # unit cost with labor
    F = np.array([value['cost']
                  for _, value in location_dict.items()])  # location cost
    device_ub = np.array([[value['es_max'], value['pv_max'], value['dg_max']]
                          for _, value in location_dict.items()])  # [n_locations, n_devices]
    ############################## Indices
   
    g_index = range(G)
    l_index = range(L)
    i_index = (0, 1)
    t_index = range(T)
    d_index = range(D)
    ############################## First Stage Model
    master = gp.Model('MasterProb', env=env)
    X_I = master.addVars(L, D, vtype=GRB.INTEGER, name='X')
    U_I = master.addVars(L, vtype=GRB.BINARY, name='U')
    eta = master.addVar(ub=float('inf'), lb=M, vtype=GRB.CONTINUOUS, name='eta')
    
    ##### Investment constraint
    capital_cost = sum(F[l] * U_I[l] for l in range(L)) + sum(X_I[l, d] * C[d] for l in l_index for d in d_index)
    operation_cost = sum(X_I[l, d] * O[d] for l in l_index for d in d_index)
    
    master_constrts = {}
    master_constrts[f'budget'] = ((1 - sv_subsidy_rate) * capital_cost <= Budget_I)
    for l in l_index:
        for d in d_index:
            master_constrts[f'X_I[{l},{d}]'] = (X_I[l, d] <= device_ub[l, d] * U_I[l])
    master_constrts[f'subsidy'] = (sv_subsidy_rate * capital_cost <= total_subsidy)
    master.addConstrs((c for c in master_constrts.values()))
    master.setObjective((1 - sv_subsidy_rate) * capital_cost + operation_cost + eta, sense=GRB.MINIMIZE)

    ##### Save master in mps + save data in pickle
    master.update()
    master.write('Models/Master.mps')
    x_keys = range(L * D)
    with open('Models/Master.pkl', 'wb') as f:
        pickle.dump([x_keys, L], f)


    ############################## Second Stage Models
    subproblems = []
    TMatrices = []
    rVectors = []
    Loads = []
    for scenario_index in tqdm(range(len(Sc))):
        s = Sc[scenario_index]
        sub = gp.Model(env=env)
        X_I = sub.addMVar((L, D), lb=0, name=f'X_I')
        U_I = sub.addMVar(L, lb=0, ub=1, name='U_I')
        X_E = sub.addMVar((L, D), name=f'X_E')
        Y_PVES = sub.addMVar((I, G, T), name=f'Y_PVES')
        Y_DGES = sub.addMVar((I, G, T), name=f'Y_DGES')
        Y_GridES = sub.addMVar((I, G, T), name=f'Y_GridES')
        Y_PVL = sub.addMVar((I, G, T), name=f'Y_PVL')
        Y_DGL = sub.addMVar((I, G, T), name=f'Y_DGL')
        Y_ESL = sub.addMVar((I, G, T), name=f'Y_ESL')
        Y_GridL = sub.addMVar((I, G, T), name=f'Y_GridL')
        Y_PVCur = sub.addMVar((I, G, T), name=f'Y_PVCur')
        Y_DGCur = sub.addMVar((I, G, T), name=f'Y_DGCur')
        Y_PVGrid = sub.addMVar((I, G, T), name=f'Y_DGGrid')
        Y_DGGrid = sub.addMVar((I, G, T), name=f'Y_PVGrid')
        Y_ESGrid = sub.addMVar((I, G, T), name=f'Y_ESGrid')
        Y_E = sub.addMVar((I, G, T), name=f'Y_E')
        Y_LSh = sub.addMVar((I, G, T), name=f'Y_LSh')
        Y_I = sub.addMVar((I, G, T), name='Y_I')
        Y_LT = sub.addMVar((I, G, T, T), name=f'Y_LT')
        sub.update()

        ###### Prepare Load, Outage, and Load Growth
        Load = np.array([(1 + i * LG[s[2]]/100) ** n * LD[:, :, s[1]] for i in (0, 1)])
        Outage = {}
        for g in g_index:
            outage_duration = max(0, int(OT[g, s[0]]))
            if outage_duration != 0:
                if outage_duration >= 168 - (outage_start + 1):
                    Outage[g] = range(outage_start, T)
                else:
                    Outage[g] = range(outage_start, outage_start + outage_duration)
        ###### Expansion Constraints
        capital_cost_E = gp.quicksum(X_E[:, d].sum() * C[d] for d in d_index)
        operation_cost_E = gp.quicksum(X_E[:, d].sum() * O[d] for d in d_index)
        sub.addConstr(
            (1 - sv_subsidy_rate) * capital_cost_E <= Budget_E, name='Budget Limit')
        sub.addConstr(
            sv_subsidy_rate * capital_cost_E <= 0.5 * total_subsidy, name='Total Subsidy')
        ###### Capacity limit
        sub.addConstrs(
            (X_E[l, d] + X_I[l, d] <= device_ub[l, d] for l,d in np.ndindex(L, D)),
              name='ES Location Limit')
        ###### Operation Constraints
        after_degradation = (1 - degrad_rate) ** n

        sub_constrs = []
        for i in range(I):
            available_es = i * after_degradation * X_I[:, 0].sum() + i * X_E[l, 0].sum()
            available_pv = X_I[:, 1].sum() + i * X_E[:, 1].sum()
            available_dg = (X_I[:, 2].sum() + i * X_E[:, 2].sum()) * dg_effi
            for g in g_index:
                start = time()
                a = time()
                for t in t_index:
                    if t in Outage[g]:
                        trans_to_t = gp.quicksum(Y_LT[i, g, to, t] for to in range(max(0, t - trans_period), t))
                    else:
                        trans_to_t = 0
                    trans_from_t = gp.quicksum(Y_LT[i, g, t, to] for to in range(t + 1, min(T, t + trans_period + 1)))
                    pv_dg_grid_to_es = Y_PVES[i, g, t] + Y_DGES[i, g, t] + Y_GridES[i, g, t]
                    es_to_grid_load = Y_ESL[i, g, t] + Y_ESGrid[i, g, t]
                    es_level_plus_trans = Y_E[i, g, t] + trans_to_t

                    sub_constrs.append(
                        es_level_plus_trans >= es_soc_lb * available_es)
                    sub_constrs.append(
                        es_level_plus_trans <= es_soc_ub * available_es)
                    sub_constrs.append(
                        pv_dg_grid_to_es <= (es_soc_ub - es_soc_lb) * available_es)
                    sub_constrs.append(
                        es_to_grid_load <= (es_soc_ub - es_soc_lb) * available_es)

                    pv_usage = Y_PVL[i, g, t] + Y_PVGrid[i, g, t] +\
                               Y_PVCur[i, g, t] + Y_PVES[i, g, t]
                    sub_constrs.append(pv_usage <= PV[g, t] * available_pv)

                    dg_usage = Y_DGL[i, g, t] + Y_DGGrid[i, g, t] + \
                               Y_DGES[i, g, t] +Y_DGCur[i, g, t]
                    sub_constrs.append(dg_usage <= available_dg)

                    means_of_load = Y_ESL[i, g, t] + Y_DGL[i, g, t] + \
                                    Y_PVL[i, g, t] + Y_GridL[i, g, t] +  \
                                    Y_LSh[i, g, t] + trans_to_t - trans_from_t
                    sub_constrs.append(means_of_load == Load[i, g, t])

                    # Outage and Trans times
                    sub_constrs.append(Y_LT[i, g, t, t] == 0)

                    if t not in Outage[g]:
                        sub_constrs.append(Y_LSh[i, g, t] == 0)
                        sub_constrs.append(trans_to_t == 0)
                    else:
                        grid_trade = Y_GridL[i, g, t] + Y_GridES[i, g, t] + \
                                     Y_PVGrid[i, g, t] + Y_ESGrid[i, g, t] + Y_DGGrid[i, g, t]
                        sub_constrs.append(grid_trade == 0)

                        sub_constrs.append(trans_from_t <= drp * Load[i, g, t])

                    if t in no_trans[g]:
                        sub_constrs.append(trans_from_t == 0)
                    else:
                        allowed_to_trans = Y_E[i, g, t] - es_soc_lb * available_es
                        sub_constrs.append(trans_from_t <= allowed_to_trans)
                    # Balance
                    if t < T-1:
                        sub_constrs.append(
                            Y_E[i, g, t + 1] == Y_E[i, g, t] -
                            trans_from_t +
                            es_effi * pv_dg_grid_to_es -
                            es_eta * es_to_grid_load / es_effi)
                    sub_constrs.append(
                        Y_I[i, g, t] == trans_from_t)
        sub.addConstrs((c for c in sub_constrs))
        ###### Costs
        Costs = [0, 0]
        for i in range(I):
            # Cache repeated sums
            sum_LSh = Y_LSh[i].sum()
            sum_PVCur = Y_PVCur[i].sum()
            sum_DGCur = Y_DGCur[i].sum()
            sum_GridImport = Y_GridES[i].sum() + Y_GridL[i].sum()
            sum_GridExport = Y_ESGrid[i].sum() + Y_PVGrid[i].sum() + Y_DGGrid[i].sum()
            sum_DG = Y_DGES[i].sum() + Y_DGL[i].sum() + Y_DGGrid[i].sum()
            sum_Load = Y_ESL[i].sum() + Y_PVL[i].sum() + Y_DGL[i].sum()
            sum_I = Y_I[i].sum()

            # Final expression assembly
            C_LSh = e_sheding * sum_LSh
            C_Cur = e_cur_pv * sum_PVCur + e_cur_dg * sum_DGCur
            C_IE = e_grid_import * sum_GridImport - e_grid_export * sum_GridExport
            C_F = dg_fule_cost * sum_DG
            C_R = -e_load * sum_Load
            C_IN = -e_drp * sum_I

            Costs[i] = C_LSh + C_Cur + C_IE + C_F + C_R + C_IN
        # Force scalars to be float
        pr = Pr[scenario_index]
        total_cost = pr * (capital_cost_E + operation_cost_E * (N - n) + to_year * (n * Costs[0] + (N - n) * Costs[1]))
        sub.setObjective(total_cost, sense=GRB.MINIMIZE)

        sub.update()

        AMatrix = sub.getA()  # [constraints, vars]
        Constrs = sub.getConstrs()
        TMatrix = AMatrix[:, 0:len(x_keys)]
        rVector = np.array([sub.getAttr("RHS", Constrs)])
        sub.write(f'Models/Sub{scenario_index}.lp')
        with open(f'Models/Sub{scenario_index}-Tr.pkl', 'wb') as f:
            pickle.dump([TMatrix, rVector], f)
        with open(f'Models/Sub{scenario_index}-info.pkl', 'wb') as f:
            pickle.dump([Load, Outage, e_grid_import, e_load, C], f)
        ##### Store
        subproblems.append(sub)
        TMatrices.append(TMatrix)
        rVectors.append(rVector)
        Loads.append(Load)
    ############################## Memory Monitor
    mem = psutil.virtual_memory()
    print(psutil.virtual_memory())
    print(f"Used: {mem.percent}% | Available: {mem.available / 1e9:.2f} GB")

    return master, subproblems, TMatrices, rVectors, Pr, Loads



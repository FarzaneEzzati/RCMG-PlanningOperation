import pickle

import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import quicksum, GRB
from itertools import product
import warnings
from tqdm import tqdm
import gc  # garbage collecter
import psutil  # to monitor memory
from time import time

warnings.filterwarnings('ignore')

env = gp.Env()
env.setParam('OutputFlag', 0)

""" reduce uncertainty to only outage duration and load growth, independent"""


def read_data(mg_id):
    ############################## Import Data
    location_df = pd.read_csv(f'Data/location_{mg_id}.csv')
    general_df = pd.read_csv(f'Data/data_{mg_id}.csv')
    general_dict = {c: general_df[c].iloc[0] for c in general_df.columns}
    location_dict = {l: dict(zip(location_df.columns[1:], location_df.iloc[l].values[1:])) for l in
                     location_df['location']}

    ############################## Power Outage Scenarios
    OT = pd.read_csv('Scenarios/Outage/Outage_Scenarios_reduced.csv', usecols=['Gamma_Scenario', 'Normalized'])
    OT_prob = OT['Normalized']
    OT_scen = np.tile(OT['Gamma_Scenario'], (12, 1))  # [12, OT_count]
    OT_count = OT_scen.shape[1]
    ############################## PV
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    PV_scen = []
    for m in months:
        df = pd.read_csv(f'Scenarios/PV/{mg_id}/PVscenario-{m}.csv')
        week_scen = np.tile(df[df.columns[2:]].iloc[0], (1, 7))
        PV_scen.append(week_scen)
    PV_scen = np.array(PV_scen).squeeze(1)
    ############################## Load scenarios
    joint_months = ['JanFeb', 'MarApr', 'MayJun', 'JulAug', 'SepOct', 'NovDec']
    LD_scen = []
    for m in joint_months:
        df_w = pd.read_csv(f'Scenarios/Load Demand/{mg_id}/LoadScenarios-{m}-w.csv')
        df_e = pd.read_csv(f'Scenarios/Load Demand/{mg_id}/LoadScenarios-{m}-e.csv')
        weekday_scen = np.tile(df_w[df_w.columns[1:]], (1, 5))
        weekend_scen = np.tile(df_e[df_e.columns[1:]], (1, 2))
        weeklong_scen = np.concatenate((weekday_scen, weekend_scen), axis=1)
        num_scens = weeklong_scen.shape[0]
        weeklong_scen = weeklong_scen.reshape((num_scens, 168, 4)).sum(axis=2)
        LD_scen.append(weeklong_scen)
        LD_scen.append(weeklong_scen)
    LD_prob = np.array(df_w['probs'])
    LD_scen = np.transpose(np.array(LD_scen), (0, 2, 1))  # [12, 168, LD_count]
    LD_count = LD_scen.shape[-1]
    ############################## Load Growth Scenarios
    LG = pd.read_csv('Scenarios/Load Demand/annual growth scenarios.csv')
    LG_scen = LG['Lognorm Scenario'] / 100
    LG_prob = LG['Probability']  # [LG_count]
    LG_count = LG_scen.shape[0]
    ############################## Combination Scenarios
    combination_scen = np.array(list(product(range(OT_count), range(LD_count), range(LG_count))))
    combination_prob = np.array([OT_prob[s[0]] * LD_prob[s[1]] * LG_prob[s[2]]
                                 for s in combination_scen])
    with open('Data/Probabilities.pkl', 'wb') as handle:
        pickle.dump(combination_prob, handle)

    return general_dict, location_dict, \
        OT_scen, LD_scen, LG_scen, PV_scen, \
        combination_scen, combination_prob


def build_model(mg_id):
    ############################## Read Data
    general_dict, location_dict, OT, LD, LG, PV, Sc, Pr = read_data(mg_id)
    ############################## Set Parameters
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
    M = -10000
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
    ld_index = np.ndindex(L, D)
    gt_index = np.ndindex(G, T)
    gt_minus1_index = np.ndindex(G, T - 1)
    g_index = range(G)
    l_index = range(L)
    i_index = (0, 1)
    t_list = range(T)
    ############################## First Stage Model
    master = gp.Model('MasterProb', env=env)
    X_I = master.addMVar((L, D), vtype=GRB.INTEGER, name='X')
    U_I = master.addMVar(L, vtype=GRB.BINARY, name='U')
    eta = master.addVar(ub=float('inf'), lb=M, vtype=GRB.CONTINUOUS, name='eta')
    ##### Investment constraint
    capital_cost = sum(F[l] * U_I[l] for l in range(L)) + sum(X_I[l, d] * C[d] for l, d in ld_index)
    operation_cost = sum(X_I[l, d] * O[d] for l, d in ld_index)
    master.addConstr(
        (1 - sv_subsidy_rate) * capital_cost <= Budget_I, name='Investment')
    master.addConstrs(
        (X_I[l, d] <= device_ub[l, d] * U_I[l] for l, d in ld_index), name='Location Allowance')
    master.addConstr(
        sv_subsidy_rate * capital_cost <= total_subsidy, name='Total Subsidy')
    master.setObjective((1 - sv_subsidy_rate) * capital_cost + operation_cost + eta, sense=GRB.MINIMIZE)
    ##### Save master in mps + save data in pickle
    master.update()
    master.write('Models/Master.mps')
    X_I_keys = range(L * D + L)
    with open('Models/Master_X_Info.pkl', 'wb') as f:
        pickle.dump(X_I_keys, f)
    ##### Free up memory
    master.dispose()
    del master
    del X_I, U_I, eta, capital_cost, operation_cost
    gc.collect()

    ############################## Second Stage Models
    for scenario_index in tqdm(range(len(Sc))):
        s = Sc[scenario_index]
        sub = gp.Model(env=env)
        X_I = sub.addMVar((L, D), name=f'X_I')
        U_I = sub.addVars(L, ub=1, name='U_I')
        sub.update()
        X_E = sub.addMVar((L, D), name=f'X_I')
        U_E = sub.addVars(L, ub=1, name='U_E')
        Y_PVES = {i: sub.addMVar((G, T), name=f'Y_PVES[{i}]') for i in i_index}
        Y_DGES = {i: sub.addMVar((G, T), name=f'Y_DGES[{i}]') for i in i_index}
        Y_GridES = {i: sub.addMVar((G, T), name=f'Y_GridES[{i}]') for i in i_index}
        Y_PVL = {i: sub.addMVar((G, T), name=f'Y_PVL[{i}]') for i in i_index}
        Y_DGL = {i: sub.addMVar((G, T), name=f'Y_DGL[{i}]') for i in i_index}
        Y_ESL = {i: sub.addMVar((G, T), name=f'Y_ESL[{i}]') for i in i_index}
        Y_GridL = {i: sub.addMVar((G, T), name=f'Y_GridL[{i}]') for i in i_index}
        Y_PVCur = {i: sub.addMVar((G, T), name=f'Y_PVCur[{i}]') for i in i_index}
        Y_DGCur = {i: sub.addMVar((G, T), name=f'Y_DGCur[{i}]') for i in i_index}
        Y_PVGrid = {i: sub.addMVar((G, T), name=f'Y_DGGrid[{i}]') for i in i_index}
        Y_DGGrid = {i: sub.addMVar((G, T), name=f'Y_PVGrid[{i}]') for i in i_index}
        Y_ESGrid = {i: sub.addMVar((G, T), name=f'Y_ESGrid[{i}]') for i in i_index}
        Y_E = {i: sub.addMVar((G, T), name=f'Y_E[{i}]') for i in i_index}
        Y_LS = {i: sub.addMVar((G, T), name=f'Y_LS[{i}]') for i in i_index}
        Y_I = {i: sub.addMVar((G, T), name='Y_I[{i}]') for i in i_index}
        Y_LT = {i: sub.addMVar((G, T, T), name=f'Y_LT[{i}]') for i in i_index}
        ###### Prepare Load, Outage, and Load Growth
        Load = np.array([(1 + i * LG[s[2]]) ** n * LD[:, :, s[1]] for i in (0, 1)])
        Outage = {}
        for g in g_index:
            outage_duration = max(0, int(OT[g, s[0]]))
            if outage_duration != 0:
                if outage_duration >= 168 - (outage_start + 1):
                    Outage[g] = range(outage_start, T)
                else:
                    Outage[g] = range(outage_start, outage_start + outage_duration)
        ###### Expansion Constraints
        capital_cost_E = sum(F[l] * (U_E[l] - U_I[l]) for l in range(L)) + gp.quicksum(X_E[l, d] * C[d] for l, d in ld_index)
        operation_cost_E = gp.quicksum(X_E[l, d] * O[d] for l, d in ld_index)
        sub.addConstr(capital_cost_E <= Budget_E, name='E_Investment')
        ###### Capacity limit
        sub.addConstr(X_E + X_I <= device_ub, name='ES Location Limit')
        sub.addConstrs((X_E[l, d] <= U_E[l] * device_ub[l, d] for l, d in ld_index),
            name='Install E')
        sub.addConstrs((U_E[l] >= U_I[l] for l in l_index), name='One Opening')
        ###### Operation Constraints
        after_degradation = (1 - degrad_rate) ** n
        for i in (0, 1):
            available_es = quicksum(i * after_degradation * X_I[l, 0] + i * X_E[l, 0] for l in l_index)
            sub.addConstr(Y_E[i][:, 0] == es_soc_ub * available_es,
                          name='Et0')
            sub.addConstrs(
                (Y_E[i][g, t] + gp.quicksum(Y_LT[i][g, 1:, t]) >= es_soc_lb * available_es for g, t in gt_index),
                name='E_LB')
            sub.addConstrs(
                (Y_E[i][g, t] + gp.quicksum(Y_LT[i][g, :t, t]) <= es_soc_ub * available_es
                 for g, t in gt_index),
                name='E_LB')
            sub.addConstr(Y_ESL[i] + Y_ESGrid[i] <= (es_soc_ub - es_soc_lb) * available_es,
                name='ES_Discharge')
            sub.addConstr(Y_PVES[i] + Y_DGES[i] + Y_GridES[i] <= (es_soc_ub - es_soc_lb) * available_es,
                name='ES_Charge')
            sub.addConstr(Y_PVL[i] + Y_PVGrid[i] + Y_PVCur[i] + Y_PVES[i] == PV * (gp.quicksum(X_I[:, 1]) + i * gp.quicksum(X_E[:, 1])),
                name='PV')
            sub.addConstr(Y_DGL[i] + Y_DGGrid[i] + Y_DGES[i] + Y_DGCur[i] == dg_effi * (gp.quicksum(X_I[:, 2]) + i * gp.quicksum(X_E[:, 2])),
                name='DG')
            sub.addConstrs(
                (Y_ESL[i][g, t] + Y_DGL[i][g, t] + Y_PVL[i][g, t] + Y_GridL[i][g, t] + Y_LS[i][g, t] +
                 gp.quicksum(Y_LT[i][g, :t, t]) - gp.quicksum(Y_LT[i][g, t, t+1:]) == Load[i][g, t] for g, t in gt_index),
                name='LoadD')
            sub.addConstrs(
                (Y_LS[i][g, t] == 0 for g, t in gt_index if t not in Outage[g]),
                name='NoOutNoLoss')
            sub.addConstrs(
                (Y_GridL[i][g, t] >= 0.75 * Load[i][g, t] for g, t in gt_index if t not in Outage[g]),
                name='NoOutUseGrid')
            sub.addConstrs(
                (gp.quicksum(Y_LT[i][g, to, t] for to in range(1, t)) == 0 for g, t in gt_index if
                 t not in Outage[g]),
                name='NoTransToNonOut')
            sub.addConstrs(
                (gp.quicksum(Y_LT[i][g, t, to] for to in range(t + 1, T)) == 0 for g, t in gt_index if
                 t in no_trans[g]),
                name='NoTrans')
            sub.addConstrs(
                (gp.quicksum(Y_LT[i][g, t, t+1:]) <= drp * Load[i][g, t]
                 for g, t in gt_index if t in Outage[g]),
                name='MaxLoadTrans')
            sub.addConstrs(
                (gp.quicksum(Y_LT[i][g, t, t+1:]) <= Y_E[i][g, t] - es_soc_lb * available_es
                 for g, t in gt_index),
                name='TransIfPoss')
            sub.addConstr(gp.quicksum(Y_LT[i][g, t, t] for g, t in gt_index) == 0,
                          name='NoTransToSelf')
            sub.addConstrs(
                (Y_E[i][g, t + 1] == Y_E[i][g, t] -
                 gp.quicksum(Y_LT[i][g, t, t+1:]) +
                 es_effi * (Y_PVES[i][g, t] + Y_DGES[i][g, t] + es_eta * Y_GridES[i][g, t]) -
                 es_eta * (Y_ESL[i][g, t] + Y_ESGrid[i][g, t]) / es_effi for g, t in gt_minus1_index),
                name='Balance')
            sub.addConstrs(
                (Y_GridL[i][g, t] + Y_GridES[i][g, t] +
                 Y_PVGrid[i][g, t] + Y_ESGrid[i][g, t] + Y_DGGrid[i][g, t] == 0
                 for g in g_index for t in Outage[g]),
                name='GridTransaction')
            sub.addConstrs(
                (Y_I[i][g, t] == e_drp * gp.quicksum(Y_LT[i][g, t, t+1:]) for g, t in gt_minus1_index),
                name='Incentive')
        ###### Costs
        Costs = [0, 0]
        for i in (0, 1):
            C_LS = e_sheding * gp.quicksum(Y_LS[i][g, t] for g, t in gt_index)
            C_Cur = e_cur_pv * gp.quicksum(Y_PVCur[i][g, t] for g, t in gt_index) + \
                    e_cur_dg * gp.quicksum(Y_DGCur[i][g, t] for g, t in gt_index)
            C_IE = e_grid_import * (gp.quicksum(Y_GridES[i][g, t] + Y_GridL[i][g, t] for g, t in gt_index)) - \
                   e_grid_export * (gp.quicksum(Y_ESGrid[i][g, t] + Y_PVGrid[i][g, t] + Y_DGGrid[i][g, t]
                                                for g, t in gt_index))
            C_F = dg_fule_cost * (gp.quicksum(Y_DGES[i][g, t] + Y_DGL[i][g, t] + Y_DGGrid[i][g, t]
                                              for g, t in gt_index))
            C_R = - e_load * (gp.quicksum(Y_ESL[i][g, t] + Y_PVL[i][g, t] + Y_DGL[i][g, t]
                                          for g, t in gt_index))
            C_IN = - gp.quicksum(Y_I[i][g, t] for g, t in gt_index)
            Costs[i] = C_LS + C_Cur + C_IE + C_F + C_R + C_IN

        # Force scalars to be float
        pr = float(Pr[scenario_index])
        n = float(n)
        N = float(N)
        ty = float(to_year)
        total_cost = pr * (capital_cost_E + operation_cost_E * (N - n) + ty * (n * Costs[0] + (N - n) * Costs[1]))
        sub.setObjective(total_cost, sense=GRB.MINIMIZE)

        sub.update()

        start = time()
        AMatrix = sub.getA()  # [constraints, vars]
        Constrs = sub.getConstrs()
        TMatrix = AMatrix[:, 0:len(X_I_keys)]
        rVector = np.array(sub.getAttr("RHS", Constrs))
        sub.write(f'Models/Sub{scenario_index}.mps')
        with open(f'Models/Sub{scenario_index}-Tr.pkl', 'wb') as f:
            pickle.dump([TMatrix, rVector, X_I_keys], f)
        ##### Free up memory
        sub.dispose()
        del sub
        del X_E, U_E
        del Y_PVES, Y_DGES, Y_GridES
        del Y_PVL, Y_DGL, Y_ESL, Y_GridL
        del Y_PVCur, Y_DGCur
        del Y_PVGrid, Y_DGGrid, Y_ESGrid
        del Y_E, Y_LS, Y_I, Y_LT
        del Load, Outage
        del Costs, C_IE, C_Cur, C_R, C_F, C_IN, C_LS
        del AMatrix, TMatrix, rVector, Constrs
        del capital_cost_E, operation_cost_E
        gc.collect()
    ############################## Memory Monitor
    mem = psutil.virtual_memory()
    print(psutil.virtual_memory())
    print(f"Used: {mem.percent}% | Available: {mem.available / 1e9:.2f} GB")

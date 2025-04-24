import pickle
import gurobipy as gp
from gurobipy import Model
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from typing import Dict

env_master = gp.Env()
env_sub = gp.Env()
env_sub.setParam('OutputFlag', 0)
env_sub.setParam('DualReductions', 0)
env_sub.setParam('InfUnbdInfo', 1)


'''class BendersCut:
    def __init__(self, models, probs, TMatrices, rVectors, x_keys):
        self.models = models
        self.probs = probs
        self.TMatrices = TMatrices
        self.rVectors = rVectors
        self.x_keys = x_keys

    def __call__(self, model, where):
        if where == gp.GRB.Callback.MIPSOL:
            X = model.cbGetSolution(model._vars)
            # Get solutions in subproblems, calculate e and E
            E, e = GetPIs(optimal_x=X, models=self.models, probs=self.probs,
                          TMatrices=self.TMatrices, rVectors=self.rVectors,
                          x_keys=self.x_keys)
            # Add a cut based on the solution # For example, adding a simple cut:
            model.cbLazy(model._vars[-1] + sum(model._vars[x] * E_x for E_x in E) >= e)
'''

def LoadSubProblems():
    # Load scenario probabilities
    with open('Data/Probabilities.pkl', 'rb') as handle:
        scenarios, probs, max_outage_scenario  = pickle.load(handle)

    # Open subproblems
    models, TMatrices, rVectors = [], [], []
    print('Loading Subproblems')
    with tqdm(range(len(probs)), desc='Loading') as tbar:
        for s in range(len(probs)):
            models.append(gp.read(f'Models/Sub{s}.mps', env=env_sub))
            with open(f'Models/Sub{s}-Tr.pkl', 'rb') as handle:
                TMatrix, rVector = pickle.load(handle)
            TMatrices.append(TMatrix)
            rVectors.append(rVector)
            tbar.update(1)
    TMatrices = np.array(TMatrices)
    rVectors = np.array(rVectors)

    return models, probs, TMatrices, rVectors



def GetPIs(optimal_x, models, probs, TMatrices, rVectors, x_keys):
    duals = []  # Dictionary of dual multipliers
    s_range = range(len(probs))
    n_xs = len(x_keys)
    #### Optimize first
    for s, f in enumerate(models):
        vars = f.getVars()
        for x in x_keys:
            vars[x].UB = optimal_x[x]
            vars[x].LB = optimal_x[x]
        f.update()
        f.optimize()
        if f.status == 2:
            duals.append(np.array([c.Pi for c in f.getConstrs()]))
    duals = np.array(duals)

    # Calculate two values e and E for Benders
    duals_r = np.array([np.dot(duals[s], rVectors[s]) for s in s_range])   # [S]
    e = np.average(duals_r, weights=probs)
    E = []
    for x_key in x_keys:
        Temp1 = np.array([np.dot(TMatrices[s][:, x_key], duals[s]) for s in s_range]) # [S, C]
        E.append(np.average(Temp1, weights=probs))
    return E, e



def BendersCut(model, where, models, probs, TMatrices, rVectors, x_keys):
    if where == gp.GRB.Callback.MIPSOL:
        X = model.cbGetSolution(model._vars)
        # Get solutions in subproblems, calculate e and E
        E, e = GetPIs(optimal_x=X, models=models, probs=probs,
                          TMatrices=TMatrices, rVectors=rVectors,
                          x_keys=x_keys)
        # Add a cut based on the solution # For example, adding a simple cut:
        model.cbLazy(model._vars[-1] + sum(model._vars[x] * E_x for E_x in E) >= e)


def solve_with_BD_BandB(models, probs, TMatrices, rVectors):
    ##### Solve master by callback
    master = gp.read('Models/Master.mps', env=env_master)
    with open(f'Models/Master.pkl', 'rb') as handle:
        x_keys, num_locations = pickle.load(handle)
    master._vars = master.getVars()
    master.Params.LazyConstraints = 1
    master.Params.LogFile = "Models/master_log.log"
    master.Params.DegenMoves = 0
    master.optimize(lambda model, where: BendersCut(model, where, models, probs, TMatrices, rVectors, x_keys))

    ##### Get optimal x
    optimal_x = np.array([x.x for x in master.getVars()]) # Save optimal solution of master problem
    optimal_obj = master.ObjVal
    return optimal_x, optimal_obj, x_keys, num_locations


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
    ############################## Max of Outage
    max_outage_index = np.argmax(OT_scen)
    max_outage_scenario_index = next(i for i, s in enumerate(combination_scen) if s[0] == max_outage_index)
    with open('Data/Probabilities.pkl', 'wb') as handle:
        pickle.dump([combination_scen, combination_prob, max_outage_scenario_index], handle)

    return general_dict, location_dict, \
        OT_scen, LD_scen, LG_scen, PV_scen, \
        combination_scen, combination_prob

def solve_subproblems(sub_modles, optimal_x_u, x_keys, probs, num_locations):
    ### Get sub vars
    x_u_scenarios = []
    for sub_model in sub_models:
        sub_vars = sub_model.getVars()
        for x_key in x_keys:
            sub_vars[x_key].UB = optimal_x_u[x_key]
            sub_vars[x_key].LB = optimal_x_u[x_key]
        sub_model.update()
        sub_model.optimize()
        sub_optimal_vars = sub_model.getVars()[:len(x_keys)]
        x_u_scenarios.append([var.x for var in sub_optimal_vars])
    ### Get average
    average_x_u = np.average(np.array(x_u_scenarios), axis=0, weights=probs)
    ### Reshape
    average_x = average_x_u[:num_locations].reshape((num_locations, 3))  # [num_l, 3]
    average_u = average_x_u[num_locations:].reshape((num_locations, 1))  # [num_l, 1]
    x_u_E = np.concatenate((average_x, average_u), axis=1)

    optimal_x = optimal_x_u[:num_locations].reshape((num_locations, 3))  # [num_l, 3]
    optimal_u = optimal_x_u[num_locations:].reshape((num_locations, 1))  # [num_l, 1]
    x_u_I = np.concatenate((optimal_x, optimal_u), axis=1)
    ### Store
    pd.DataFrame(x_u_I, index=False, columns=['ES', 'PV', 'DG', 'U']).to_csv(f'(MG{mg_id})X_I.csv')
    pd.DataFrame(x_u_E, index=False, columns=['ES', 'PV', 'DG', 'U']).to_csv(f'(MG{mg_id})X_E.csv')
    return x_u_I, x_u_E

def get_report(models, x_u_I, x_u_E):
    with open('Data/Probabilities.pkl', 'rb') as handle:
        scenarios, probs, max_outage_scenario = pickle.load(handle)
    S = len(probs)
    G = 12
    T = 168
    outage_start = 15

    #  Resilience Metrics
    Phi_metric, Lambda_metric, E_metric = np.zeros(S), np.zeros(S), np.zeros(S)
    bill_saving = np.zeros(S)
    for s in range(S):
        # Open Load and Outage hours for each scenario
        with open(f'Models/Sub{s}-info.pkl', 'wb') as f:
            Load, Outage_hrs, e_grid, e_load, C = pickle.load(f)
        # Get Load Shedding vars
        Y_LSh = np.array([[[models[s].getVarByName(f'Y_LSh[{i}][{g},{t}]').x for t in range(168)]
                  for g in range(12)]
                 for i in (0, 1)])  # [2, 12, 168]
        # Calculate metrics
        for g in range(G):
            outage_end = outage_start + len(Outage_hrs[g])
            # Calculate Phi metric
            did_shed = Y_LSh[0, g, outage_start:outage_end] >= 0.75 * Load[0, g, outage_start:outage_end]
            T_fail = np.argmax(~did_shed) if not did_shed.all() else len(did_shed)
            did_shed = Y_LSh[1, g, outage_start:outage_end] >= 0.75 * Load[1, g, outage_start:outage_end]
            T_fail += np.argmax(~did_shed) if not did_shed.all() else len(did_shed)
            Phi_g = T_fail / (2 * Outage_hrs[g])
            Phi_metric[s] += Phi_g / G
            # Calculate Lambda metric
            L_0 = sum(Load[0, g, outage_start:outage_end] - Y_LSh[0, g, outage_start:outage_end]) /\
                  sum(Load[0, g, outage_start:outage_end])
            L_1 = sum(Load[1, g, outage_start:outage_end] - Y_LSh[1, g, outage_start:outage_end]) /\
                  sum(Load[1, g, outage_start:outage_end])
            Lambda_metric[s] += 0.5 * (L_0 + L_1) / G
            # Calculate E metric
            E_0 = sum(Y_LSh[0, g, outage_start:outage_end] >= 0.75 * Load[0, g, outage_start:outage_end])
            E_1 = sum(Y_LSh[1, g, outage_start:outage_end] >= 0.75 * Load[1, g, outage_start:outage_end])
            E_metric[s] += (0.5 * (E_0 + E_1) / Outage_hrs[g]) / G
        # Calculate bill saving
        Y_ESL = np.array([[[models[s].getVarByName(f'Y_ESL[{i}][{g},{t}]').x for t in range(T)]
                  for g in range(G)] for i in (0, 1)])
        Y_DGL = np.array([[[models[s].getVarByName(f'Y_DGL[{i}][{g},{t}]').x for t in range(T)]
                  for g in range(G)] for i in (0, 1)])
        Y_PVL = np.array([[[models[s].getVarByName(f'Y_PVL[{i}][{g},{t}]').x for t in range(T)]
                  for g in range(G)] for i in (0, 1)])
        es_serving = np.sum((Y_ESL[:, g, :outage_start] + Y_ESL[:, g, outage_end:]).sum() for g in range(G))
        dg_serving = np.sum((Y_DGL[:, g, :outage_start] + Y_DGL[:, g, outage_end:]).sum() for g in range(G))
        pv_serving = np.sum((Y_PVL[:, g, :outage_start] + Y_PVL[:, g, outage_end:]).sum() for g in range(G))
        mg_bill = (es_serving + pv_serving + dg_serving) * e_load
        grid_bill = (np.sum((Load[:, g, :outage_start] + Load[:, g, outage_end:]).sum() for g in range(G)))*e_grid
        bill_saving[s] = (grid_bill - mg_bill) / grid_bill

    # Expected metrics
    Phi_avg = np.average(Phi_metric, weights=probs)
    Lambda_avg = np.average(Lambda_metric, weights=probs)
    E_avg = np.average(E_metric, weights=probs)
    # Expected bill savings
    bill_saving_avg = np.average(bill_saving, weights=probs)
    # Expected recourse
    costs_avg = np.sum([models[s].ObjVal for s in range(S)])
    # First year investment
    investment = np.dot(np.sum(x_u_I[:, :3], axis=0), C)
    expansion = np.dot(np.sum(x_u_E[:, :3], axis=0), C)

    #  Save reports
    report = {'Investment': investment,
              'Reinvestment': expansion,
              'Avg Recourse': costs_avg,
              'Bill Saving': bill_saving_avg,
              'Phi metric': Phi_avg,
              'Lambda metric': Lambda_avg,
              'E metric': E_avg
              }
    pd.DataFrame(report, index=[0]).to_csv('Report.csv')


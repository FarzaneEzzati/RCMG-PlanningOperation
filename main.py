import pickle
import gurobipy as gp
import pandas as pd
from tqdm import tqdm
import numpy as np
env = gp.Env()
env2 = gp.Env()
env2.setParam('OutputFlag', 0)
env.setParam('DualReductions', 0)


# Open data required
with open('Data/ScenarioProbabilities.pkl', 'rb') as handle:
    Probs = pickle.load(handle)
handle.close()
Probs = {i: Probs[i] for i in range(20)}
# Probs = [1/2, 1/2]
with open('Models/Master_X_Info.pkl', 'rb') as handle:
    Xkeys = pickle.load(handle)
handle.close()


# Open subproblems
SP, TMatrices, rVectors = {}, {}, {}
print('Load Subproblems')
for scen in tqdm(range(len(Probs))):
    SP[scen] = gp.read(f'Models/Sub{scen}.mps', env=env2)
    with open(f'Models/Sub{scen}-Tr.pkl', 'rb') as handle:
        TMatrix, rVector = pickle.load(handle)
    handle.close()
    TMatrices[scen] = TMatrix
    rVectors[scen] = rVector


def GetPIs(X_star):
    PIs = {}  # The dictionary to save dual multipliers of subproblems
    for s in SP:  # Fix x in subproblems and solve
        first_vars = SP[s].getVars()
        for x in Xkeys:
            first_vars[x].UB = X_star[x]
            first_vars[x].LB = X_star[x]
        SP[s].optimize()
        if SP[s].status == 2:
            PIs[s] = [c.Pi for c in SP[s].getConstrs()]

    # Calculate two values e and E for Benders
    pir = [Probs[s] * sum(PIs[s][row] * rVectors[s][row] for row in range(len(rVectors[s]))) for s in range(len(Probs))]
    e = sum(pir)
    E = [[0 for x in Xkeys] for _ in range(len(Probs))]
    for s in range(len(Probs)):
        for key in TMatrices[s].keys():
            E[s][key[1]] += PIs[s][key[0]] * TMatrices[s][key]
    E = [sum(E[s][x] * Probs[s] for s in range(len(Probs))) for x in Xkeys]
    return E, e  


def BendersCut(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        X = model.cbGetSolution(model._vars)

        # Get solutions in subproblems, calculate e and E
        E, e = GetPIs(X)

        # Add a cut based on the solution # For example, adding a simple cut:
        model.cbLazy(model._vars[-1] + sum(model._vars[x] * E[x] for x in Xkeys) >= e)

        

if __name__ == '__main__':

    master = gp.read('Models/Master.mps', env=env)
    master._vars = master.getVars()
    master.Params.LazyConstraints = 1
    master.Params.MIPGap = 0.05
    master.Params.TimeLimit = 1000
    master.Params.LogFile = "master_log.log"
    master.Params.DegenMoves = 0
    master.optimize(BendersCut)

    # Reporting
    from ModelsGenerator import X_ild, C, Load_scens, AG_scens, Outage_scens, DontTrans, Y_itg, Y_ittg,\
        RNGTime, LoadPrice, GridPlus, RNGSta, RNGMonth, ReInvsYear, eta_i
    X_values = [x.x for x in master.getVars()]
    total_cost = master.ObjVal
    optimal_solution = {}
    counter = 0
    for ild in X_ild:
        optimal_solution[(ild[0], ild[1], ild[2])] = X_values[counter]
        counter += 1
    pd.DataFrame(optimal_solution, index=[0]).to_csv('OptimalSolution.csv')


    #  Solve subproblems for optimal x found
    for scen in SP.keys():
        first_vars_optimal = SP[scen].getVars()
        for x in Xkeys:
            first_vars_optimal[x].UB = X_values[x]
            first_vars_optimal[x].LB = X_values[x]
        SP[scen].update()
        SP[scen].optimize()


    print('Reporting started')
    LoadLost = sum(Probs[i] * sum(SP[i].getVarByName(f'Y_LL[{itg[0]},{itg[1]},{itg[2]}]').x
                                  for itg in Y_itg) for i in SP.keys())
    LoadServed = sum(Probs[i] * sum(SP[i].getVarByName(f'Y_ESL[{itg[0]},{itg[1]},{itg[2]}]').x +
                                    SP[i].getVarByName(f'Y_DGL[{itg[0]},{itg[1]},{itg[2]}]').x +
                                    SP[i].getVarByName(f'Y_PVL[{itg[0]},{itg[1]},{itg[2]}]').x
                                    for itg in Y_itg)for i in SP.keys())
    LoadTrans = sum(Probs[i] * sum(SP[i].getVarByName(f'Y_LT[{ittg[0]},{ittg[1]},{ittg[2]},{ittg[3]}]').x
                                   for ittg in Y_ittg) for i in SP.keys())
    GridLoad = sum(Probs[i] * sum(SP[i].getVarByName(f'Y_GridL[{itg[0]},{itg[1]},{itg[2]}]').x
                                  for itg in Y_itg)for i in SP.keys())
    TotalLoad = LoadLost + LoadServed + LoadTrans + GridLoad
    GridExport = sum(Probs[i] * sum(SP[i].getVarByName(f'Y_ESGrid[{itg[0]},{itg[1]},{itg[2]}]').x +
                                    SP[i].getVarByName(f'Y_PVGrid[{itg[0]},{itg[1]},{itg[2]}]').x +
                                    SP[i].getVarByName(f'Y_DGGrid[{itg[0]},{itg[1]},{itg[2]}]').x
                                    for itg in Y_itg)for i in SP.keys())

    #  Resilience Metrics
    LoadFail = [0 for s in SP.keys()]
    LoadServedOutage, TotalLoadOutage = [], []
    LoadServedNoTrans, TotalLoadNoTrans = [], []
    LoadServedNoOutage, TotalLoadNoOutage = [], []
    for scen in SP.keys():
        if Outage_scens[scen] >= 168 - 16:
            outage_hours = range(16, 169)
        else:
            outage_hours = range(16, 16 + Outage_scens[scen] + 1)

        for oh in outage_hours:
            if SP[scen].getVarByName(f'Y_LL[1,{oh},8]').x != 0:
                LoadFail[scen] = oh - 16
                break
        LoadServedOutage.append(sum(SP[scen].getVarByName(f'Y_ESL[{i},{t},{g}]').x +
                                    SP[scen].getVarByName(f'Y_DGL[{i},{t},{g}]').x +
                                    SP[scen].getVarByName(f'Y_PVL[{i},{t},{g}]').x
                                    for i in RNGSta for t in outage_hours for g in RNGMonth))
        TotalLoadOutage.append(sum((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear * Load_scens[scen][g][t-1]
                                   for i in RNGSta for t in outage_hours for g in RNGMonth))

        LoadServedNoTrans.append(sum(SP[scen].getVarByName(f'Y_ESL[{i},{t},{g}]').x +
                                    SP[scen].getVarByName(f'Y_DGL[{i},{t},{g}]').x +
                                    SP[scen].getVarByName(f'Y_PVL[{i},{t},{g}]').x
                                    for i in RNGSta for g in (1, 2, 3, 6, 7, 8, 9) for t in outage_hours if t in DontTrans[g]))
        TotalLoadNoTrans.append(sum((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear * Load_scens[scen][g][t-1]
                                    for i in RNGSta for g in (1, 2, 3, 6, 7, 8, 9) for t in outage_hours if t in DontTrans[g]))

        LoadServedNoOutage.append(sum(SP[scen].getVarByName(f'Y_ESL[{i},{t},{g}]').x +
                                    SP[scen].getVarByName(f'Y_DGL[{i},{t},{g}]').x +
                                    SP[scen].getVarByName(f'Y_PVL[{i},{t},{g}]').x
                                    for i in RNGSta for t in RNGTime if t not in outage_hours for g in RNGMonth))
        TotalLoadNoOutage.append(sum((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear *  Load_scens[scen][g][t - 1]
                                    for i in RNGSta for t in RNGTime if t not in outage_hours for g in RNGMonth))

    Robustness = 1 - eta_i * sum([Probs[s] * LoadFail[s]/Outage_scens[s] for s in SP.keys()])
    Redundancy =  eta_i * sum([Probs[s] * LoadServedOutage[s]/TotalLoadOutage[s] for s in SP.keys()])
    Resourcefullness =  eta_i * sum([Probs[s] * LoadServedNoTrans[s]/TotalLoadNoTrans[s] for s in SP.keys()])
    Bill1 = sum([Probs[s] * TotalLoadNoOutage[s] * GridPlus for s in SP.keys()])
    Bill2 = sum([Probs[s] * (eta_i * LoadServedNoOutage[s] * LoadPrice +
                            (TotalLoadNoOutage[s] - eta_i * LoadServedNoOutage[s]) * GridPlus)
                for s in SP.keys()])
    #  Other Reports
    report =   {'Investment': sum(C[ild[2]] * optimal_solution[ild] for ild in X_ild if ild[0] == 1),
                'Reinvestment': sum(C[ild[2]] * optimal_solution[ild] for ild in X_ild if ild[0] == 2),
                'Avg Recourse': sum(Probs[s] * SP[s].ObjVal for s in SP.keys()),
                'Load Lost%': LoadLost / TotalLoad,
                'Load Served%': LoadServed / TotalLoad,
                'Load Transferred%': LoadTrans / TotalLoad,
                'Grid Load%': GridLoad / TotalLoad,
                'Grid Exported': GridExport,
                'Bill Before': Bill1,
                'Bill After': Bill2,
                'Robustness': Robustness,
                'Redundancy': Redundancy,
                'Resourcefulness': Resourcefullness}
    pd.DataFrame(report, index=[0]).to_csv('Report.csv')




    # Previous code: No use
    '''k, t, t_last = 1, {}, 0  # Iteration number
    epsilon1 = 0.1   # Stopping tolerance

    print_format = "{:<5} {:<10} {:<15} {:<20} {:<20} {:<8}"
    print(print_format.format('Itr', 'TreeSize', 'NodesExplored', 'LowerBound', 'UpperBound', 'OptimalityGap'))

    Gap = 100
    while Gap > epsilon1:
        if len(Tree) > 0:
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Branching Till Integer Found <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            while len(Tree) > 0:
                t_bar = DictMin(Treev)
                if Treev[t_bar] > V_incumbent:
                    del Tree[t_bar]
                    del Treev[t_bar]
                    del Treex[t_bar]
                else:
                    if Treev[t_bar] > v_relaxation:
                        v_relaxation = Treev[t_bar]
                        v_list.append(v_relaxation)
                    not_int_keys = XInt(Treex[t_bar])
                    if not not_int_keys:
                        t[k] = t_bar
                        break
                    # Node splitting
                    split_key = random.choice(not_int_keys)
                    split_value = Treex[t_bar][split_key]
                    for sense in ('u', 'l'):
                        t_last += 1
                        temp_node = MasterProb(Tree[t_bar].master.copy(), Xkeys)
                        temp_node.AddSplit(split_key, split_value, sense)
                        temp_solution = temp_node.Solve()

                        if temp_solution != 'None':
                            Tree[t_last], Treex[t_last], Treev[t_last] = temp_node, temp_solution[0], temp_solution[1]

                    del Tree[t_bar]
                    del Treev[t_bar]
                    del Treex[t_bar]
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Benders' Cut Generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            PIs, ObjVals = {}, {}
            for sp in SP:
                PIs[sp], ObjVals[sp] = SP[sp].SolveForBenders(Treex[t[k]])

            Tree[t[k]].AddBendersCut(PIs, SP, TMatrices, rVectors, Probs)
            master_solution = Tree[t[k]].Solve()

            if master_solution[1] > v_relaxation:
                v_relaxation = master_solution[1]

            if master_solution[0] == Treex[t[k]]:
                if master_solution[1] < V_incumbent:
                    X_star = master_solution[0]
                    ObjVal_star = master_solution[1]
                    V_incumbent = master_solution[1]
                    V_list.append(V_incumbent)

                    nodes_to_remove = []
                    for key in Tree.keys():
                        if Treev[key] > V_incumbent:
                            nodes_to_remove.append(key)
                    for n in nodes_to_remove:
                        del Tree[n]
                        del Treex[n]
                        del Treev[n]

                del Tree[t[k]]
                del Treex[t[k]]
                del Treev[t[k]]
            else:
                if master_solution[1] < V_incumbent:
                    Treex[t[k]], Treev[t[k]] = master_solution[0], master_solution[1]
                else:
                    del Tree[t[k]]
                    del Treex[t[k]]
                    del Treev[t[k]]


            if V_incumbent * v_relaxation < 0:
                Gap = 1000
            else:
                Gap = 100 * (V_incumbent - v_relaxation) / abs(V_incumbent)

            print(print_format.format(k, len(Tree), t_last+1, v_relaxation, V_incumbent, Gap))
            k += 1
        else:
            break'''








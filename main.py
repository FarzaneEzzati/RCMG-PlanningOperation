import pickle
import gurobipy as gp
import pandas as pd
from tqdm import tqdm
import copy
from ModelsGenerator import Xkeys, X_ld, C, Load_scens, AG_scens, Outage_scens, DontTrans, Y_itg, Y_ittg, \
    RNGTime, LoadPrice, GridPlus, RNGSta, RNGMonth, ReInvsYear, eta_i

env = gp.Env()
env2 = gp.Env()
env2.setParam('OutputFlag', 0)

# Open data required
with open('Data/ScenarioProbabilities.pkl', 'rb') as handle:
    Probs = pickle.load(handle)
handle.close()
Probs = {i: Probs[i] for i in range(30)}

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
        vars = SP[s].getVars()
        for x in Xkeys:
            vars[x].UB = X_star[x]
            vars[x].LB = X_star[x]
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
    # Solve master problem by callback
    master = gp.read('Models/Master.mps', env=env)
    master._vars = master.getVars()
    master.Params.LazyConstraints = 1
    '''master.Params.MIPGap = 0.00002
    master.Params.TimeLimit = 2000'''
    master.Params.LogFile = "master_log.log"
    master.Params.DegenMoves = 0
    master.optimize(BendersCut)

    # Reporting
    X_values = [x.x for x in master.getVars()] # Save optimal solution of master problem
    total_cost = master.ObjVal # the objective value of master problem
    X1, counter = {}, 0  # start changing solution format from a list to dictionary to find ild
    for ld in X_ld:
        X1[(ld[0], ld[1])] = X_values[counter]
        counter += 1

    #  Solve subproblems for optimal x found
    X2 = {ld: 0 for ld in X_ld}
    for scen in SP.keys():
        vars_optimal = SP[scen].getVars()
        for x in Xkeys:
            vars_optimal[x].UB = X_values[x]
            vars_optimal[x].LB = X_values[x]
        SP[scen].update()
        SP[scen].optimize()
        for ld in X_ld:
            X2[ld] += Probs[scen] * SP[scen].getVarByName(f'X2[{ld[0]},{ld[1]}]').x
    pd.DataFrame(X1, index=[0]).to_csv('X1.csv') # master problem solution save
    pd.DataFrame(X2, index=[0]).to_csv('X2.csv') # subproblem solution save

    print('Reporting started')
    #  Resilience Metrics
    RobList = []
    LSOList, LOList = [], []  # Load Served in Outage List, Load in Outage List
    LSnTList, LnTList = [], []  # Load Served when no Transfer List
    LSnOList, LnOList = [], []  # Load Served when no Outage List
    LLOList = []  # Load Lost when Outage
    LTOList = []  # Load Transed when Outage
    ImportList = []
    for scen in SP.keys():
        if Outage_scens[scen] >= 168 - 16:
            outage_hours = range(16, 169)
        else:
            outage_hours = range(16, 16 + Outage_scens[scen] + 1)

        AllLoadFails, AllOutages = 0, 0
        AllLoadTrans = 0
        for i in RNGSta:
            for g in RNGMonth:
                Fail = copy.copy(Outage_scens[scen])
                for oh in outage_hours[:len(outage_hours)-2]:
                    yll1 = SP[scen].getVarByName(f'Y_LL[{i},{oh},{g}]').x >=\
                           0.75 * ((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][oh-1]
                    yll2 = SP[scen].getVarByName(f'Y_LL[{i},{oh+1},{g}]').x >= \
                          0.75 * ((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][oh]

                    if (yll1, yll2) == (True, True):
                        Fail = oh - 16
                        break
                AllLoadFails += Fail
                AllOutages += Outage_scens[scen]

                for oh in outage_hours:
                    AllLoadTrans += sum(SP[scen].getVarByName(f'Y_LT[{i},{oh},{tt},{g}]').x
                                        for tt in range(oh, outage_hours[-1]+1))
        RobList.append(AllLoadFails/AllOutages)

        LOList.append(sum(((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][t - 1]
                                  for i in RNGSta for t in outage_hours for g in RNGMonth))

        LLOList.append(sum(SP[scen].getVarByName(f'Y_LL[{i},{t},{g}]').x
                                  for i in RNGSta for t in outage_hours for g in RNGMonth))

        LSOList.append(sum(SP[scen].getVarByName(f'Y_ESL[{i},{t},{g}]').x +
                          SP[scen].getVarByName(f'Y_DGL[{i},{t},{g}]').x +
                          SP[scen].getVarByName(f'Y_PVL[{i},{t},{g}]').x
                                    for i in RNGSta for t in outage_hours for g in RNGMonth))

        LTOList.append(AllLoadTrans)

        LSnTList.append(sum(SP[scen].getVarByName(f'Y_ESL[{i},{t},{g}]').x +
                                     SP[scen].getVarByName(f'Y_DGL[{i},{t},{g}]').x +
                                     SP[scen].getVarByName(f'Y_PVL[{i},{t},{g}]').x
                                     for i in RNGSta for g in (1, 2, 3, 6, 7, 8, 9) for t in outage_hours if
                                     t in DontTrans[g]))

        LnTList.append(sum(((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][t - 1]
                                    for i in RNGSta for g in (1, 2, 3, 6, 7, 8, 9) for t in outage_hours if
                                    t in DontTrans[g]))

        LSnOList.append(sum(SP[scen].getVarByName(f'Y_ESL[{i},{t},{g}]').x +
                                      SP[scen].getVarByName(f'Y_DGL[{i},{t},{g}]').x +
                                      SP[scen].getVarByName(f'Y_PVL[{i},{t},{g}]').x
                                      for i in RNGSta for t in RNGTime if t not in outage_hours for g in RNGMonth))
        LnOList.append(sum(((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][t - 1]
                                     for i in RNGSta for t in RNGTime if t not in outage_hours for g in RNGMonth))

    Robustness = sum([Probs[s] * RobList[s] for s in SP.keys()])
    Redundancy = eta_i * sum([Probs[s] * LSOList[s] / LOList[s] for s in SP.keys()])
    Resourcefullness = eta_i * sum([Probs[s] * LSnTList[s] / LnTList[s] for s in SP.keys()])
    Bill1 = sum([Probs[s] * LnOList[s] * GridPlus for s in SP.keys()])
    Bill2 = sum([Probs[s] * (eta_i * LSnOList[s] * LoadPrice + (LnOList[s] - eta_i * LSnOList[s]) * GridPlus)
                 for s in SP.keys()])
    GridExport = sum(Probs[scen] * sum(SP[scen].getVarByName(f'Y_ESGrid[{itg[0]},{itg[1]},{itg[2]}]').x +
                                    SP[scen].getVarByName(f'Y_PVGrid[{itg[0]},{itg[1]},{itg[2]}]').x +
                                    SP[scen].getVarByName(f'Y_DGGrid[{itg[0]},{itg[1]},{itg[2]}]').x
                                    for itg in Y_itg) for scen in SP.keys())
    GridImport = sum([Probs[scen] * (LnOList[scen] - eta_i * LSnOList[scen]) for scen in SP.keys()])
    GridImportPerc = sum(Probs[scen] * (LnOList[scen] - eta_i * LSnOList[scen])/LnOList[scen] for scen in SP.keys())
    #  Save reports
    report = {'Investment': sum(C[ld[1]] * X1[ld] for ld in X_ld),
              'Reinvestment': sum(C[ld[1]] * X2[ld] for ld in X_ld),
              'Avg Recourse': sum(Probs[scen] * SP[scen].ObjVal for scen in SP.keys()),
              'Load Lost%': sum(Probs[scen] * LLOList[scen]/LOList[scen] for scen in SP.keys()),
              'Load Served%': sum(Probs[scen] * LSOList[scen]/LOList[scen] for scen in SP.keys()),
              'Load Transferred%': sum(Probs[scen] * LTOList[scen]/LOList[scen] for scen in SP.keys()),
              'Grid Load%': GridImportPerc,
              'Grid Exported': GridExport,
              'Grid Imported': GridImport,
              'Bill Before': Bill1,
              'Bill After': Bill2,
              'Robustness': Robustness,
              'Redundancy': Redundancy,
              'Resourcefulness': Resourcefullness,
              'ES1': sum(X1[ld] for ld in X_ld if ld[1] == 1),
              'PV1': sum(X1[ld] for ld in X_ld if ld[1] == 2),
              'DG1': sum(X1[ld] for ld in X_ld if ld[1] == 3),
              'ES2': sum(X2[ld] for ld in X_ld if ld[1] == 1),
              'PV2': sum(X2[ld] for ld in X_ld if ld[1] == 2),
              'DG2': sum(X2[ld] for ld in X_ld if ld[1] == 3)
              }
    pd.DataFrame(report, index=[0]).to_csv('Report.csv')

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
    # Previous code: No use

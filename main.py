import pickle
import pandas as pd
import numpy as np
from MasterSubProbs import MasterProb, SubProb
import random
from static_functions import DictMin, IndexUp
import gurobipy as gp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
env = gp.Env()
env2 = gp.Env()
env2.setParam('OutputFlag', 0)
env.setParam('DualReductions', 0)

with open('Data/ScenarioProbabilities.pkl', 'rb') as handle:
    Probs = pickle.load(handle)
handle.close()
Probs = [Probs[1], Probs[2], Probs[3], Probs[4] ]

with open('Models/Master_X_Info.pkl', 'rb') as handle:
    Xkeys = pickle.load(handle)
handle.close()




SP, TMatrices, rVectors = {}, {}, {}
print('Load Subproblems')
for scen in tqdm(range(4)):
    SP[scen] = gp.read(f'Models/Sub{scen+1}.mps', env=env2)
    with open(f'Models/Sub{scen+1}-Tr.pkl', 'rb') as handle:
        TMatrix, rVector = pickle.load(handle)
    handle.close()
    TMatrices[scen] = TMatrix
    rVectors[scen] = rVector

    '''AMatrix = s_model.getA().todok()
    Constrs = s_model.getConstrs() 

    possibleTkeys = [(r, x) for r in range(len(Constrs)) for x in Xkeys]
    TMatrices.append({key: AMatrix[key] for key in possibleTkeys if key in AMatrix.keys()})
    rVectors.append({c: Constrs[c].RHS for c in range(len(Constrs))})'''


def GetPIs(X_values):
    PIs = {}
    for scen in SP:
        vars = SP[scen].getVars()
        for x in Xkeys:
            vars[x].UB = X_values[x]
            vars[x].LB = X_values[x]
        SP[scen].update()
        SP[scen].optimize()

        if SP[scen].status == 2:
            constrs = SP[scen].getConstrs()
            PIs[scen] = [constrs[c].Pi for c in range(len(constrs))]

    pir = [Probs[scen] * sum(PIs[scen][row] * rVectors[scen][row] for row in range(len(rVectors[scen])))
        for scen in range(len(Probs))]
    e = sum(pir)
    E = [[0 for x in Xkeys] for _ in range(len(Probs))]
    for scen in range(len(Probs)):
        for key in TMatrices[scen].keys():
            #print(key)
            E[scen][key[1]] += PIs[scen][key[0]] * TMatrices[scen][key]
    E = [sum(E[scen][x] * Probs[scen] for scen in range(len(Probs))) for x in Xkeys]
    return E, e  


def mycallback(model, where): 
    if where == gp.GRB.Callback.MIPSOL:
        X = model.cbGetSolution(model._vars)
        # Get solutions in subproblems, calculate e and E
        E, e = GetPIs(X)
        # Add a cut based on the solution # For example, adding a simple cut: 
        model.cbLazy(model._vars[-1] + sum(model._vars[x] * E[x] for x in Xkeys) >= e)

        

if __name__ == '__main__':
    test = False
    if test:
        pass
    else:
        master = gp.read('Models/Master.mps', env=env)
        master._vars = master.getVars()
        master.Params.LazyConstraints = 1
        master.Params.MIPGap = 0.05
        master.Params.TimeLimit = 600
        start = time.time()
        master.optimize(mycallback)
        finish = time.time()
        optimal_solution = [x.x for x in master.getVars()]
        total_cost = master.ObjVal
        print(optimal_solution)




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








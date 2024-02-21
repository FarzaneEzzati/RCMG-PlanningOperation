import pickle

import pandas as pd

from MasterSubProbs import MasterProb, SubProb
import random
from static_functions import DictMin, IndexUp
import gurobipy as gp
import matplotlib.pyplot as plt
env = gp.Env()
env.setParam('OutputFlag', 0)
env.setParam('DualReductions', 0)


def XInt(x):
    not_integer_keys = []
    for key in x.keys():
        if x[key] != int(x[key]):
            not_integer_keys.append(key)
    return not_integer_keys


if __name__ == '__main__':
    with open('Models/Master_X_Info.pkl', 'rb') as handle:
        Xkeys = pickle.load(handle)
    handle.close()

    with open('Data/ScenarioProbabilities.pkl', 'rb') as handle:
        Probs = pickle.load(handle)
    handle.close()
    # temporary
    Probs = {1: Probs[1], 2: Probs[2]}

    m_model = gp.read('Models/Master.mps', env=env)
    master_model = MasterProb(m_model, Xkeys)
    master_solution = master_model.Solve()

    X, ObjVal = master_solution[0], master_solution[1]
    X_star, ObjVal_star = None, None
    v_relaxation, V_incumbent = -10e10, 10e10

    v_list, V_list = [], []
    Tree, Treev, Treex = {0: master_model}, {0: ObjVal}, {0: X}

    SP = {}
    for scen in Probs:
        s_model = gp.read(f'Models/Sub{scen}.mps', env=env)
        SP[scen] = SubProb(s_model)

    k, t, t_last = 1, {}, 0  # Iteration number
    epsilon1 = 0.1   # Stopping tolerance
    epsilon2 = 0.1
    cw = 20
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

            Tree[t[k]].AddBendersCut(PIs, SP, Probs)
            master_solution = Tree[t[k]].Solve()
            if master_solution[0] == Treex[t[k]]:
                if master_solution[1] < V_incumbent:
                    X_star = master_solution[0]
                    ObjVal_star = master_solution[1]
                    V_incumbent = master_solution[1]
                    V_list = V_incumbent
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

            print(print_format.format(k, len(Tree), t_last, v_relaxation, V_incumbent, Gap))
            k += 1
        else:
            break
    print(f'Optimal Solution is: {X_star}')
    pd.DataFrame({'Lower Bound': v_list,
                 'Upper Bound': V_list}).to_csv('BendersUpperLower.csv')







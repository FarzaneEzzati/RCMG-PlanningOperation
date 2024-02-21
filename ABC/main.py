from MasterSub import MasterProb, SubProb, Probs
from SubBB import SubBB
from CGLP import CGLP
import pickle
import random
import copy
from static_functions import DictMin, IndexUp, XInt, YInt
import gurobipy as gp
import matplotlib.pyplot as plt
env = gp.Env()
env.setParam('OutputFlag', 0)
env.setParam('DualReductions', 0)


if __name__ == '__main__':
    D, max_cut = 4, 10
    m_model = gp.read('Models/Master.mps', env=env)
    root_model = MasterProb(m_model)
    root_solution = root_model.Solve()
    X, ObjVal = root_solution[0], root_solution[1]
    X_star = None
    Y_star = None
    v, V = -float('inf'), float('inf')
    v_list = []
    T1, T1_v, T1_x = {0: root_model}, {0: ObjVal}, {0: X}

    SP = {}
    for scen in Probs:
        s_model = gp.read(f'Models/Sub{scen}.mps', env=env)
        SP[scen] = SubProb(s_model, X_keys, Y_keys, Y_int_keys)

    k, t, t_last = 1, {}, 0  # Iteration number

    Gz = {0: {sp: [] for sp in Probs}}
    gz = 0
    epsilon = 0.1   # Stopping tolerance
    print(f'V: {V:0.2f} ====== v: {v:0.2f}')
    while (k < 10, V-v > epsilon, len(T1) > 0) == (True, True, True):
        print(f'\n{10*">"} Iteration {k}')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Branching Till Integer Found <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        while True:
            t_bar = DictMin(T1_v)
            if T1_v[t_bar] > v:
                v = T1_v[t_bar]
                v_list.append(v)
            non_int = XInt(T1_x[t_bar])
            if not non_int:
                t[k] = t_bar
                break
            # Node splitting
            split_key = random.choice(non_int)
            split_value = T1_x[t_bar][split_key]
            node_1, x_keys_1 = T1[t_bar].ReturnModel()
            for sense in ('u', 'l'):
                t_last += 1
                temp_node = MasterProb(node_1.copy(), copy.copy(x_keys_1))
                temp_node.AddSplit(split_key, split_value, sense)
                temp_output = temp_node.Solve()

                if temp_output != 'None':
                    T1[t_last], T1_x[t_last], T1_v[t_last] = temp_node, temp_output[0], temp_output[1]

                    SP[t_last] = {}
                    Gz[t_last] = copy.copy(Gz[t_bar])
                    for sp in Probs:
                        s_model, s_x, s_y, s_yitk = SP[t_bar][sp].ReturnModel()
                        SP[t_last][sp] = SubProb(s_model.copy(), s_x, s_y, s_yitk)
            del T1[t_bar]
            del T1_v[t_bar]
            del T1_x[t_bar]
            del SP[t_bar]
        print(f'BB Tree has {len(T1)} node(s). Node {t[k]} is selected')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Benders' Cut Generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        SP_y, SP_v, SP_bt, SP_ut, SP_lt = {}, {}, {}, {}, {}

        for sp in SP[t[k]]:
            print(f'Convexifying SP {sp}:', end=' ')
            gen_cut, do_cglp, max_cuts = True, True, 0
            nie = None
            y_d, v_d, bt_d, ut_d, lt_d = SP[t[k]][sp].SolveForBenders(T1_x[t[k]])
            if not YInt({y: y_d[y] for y in SP[t[k]][sp].Y_int_keys}):
                gen_cut = False
            else:
                T2 = SubBB(SP[t[k]][sp], y_d, v_d, T1_x[t[k]], D)
                cglp_class = CGLP(T1[t[k]], SP[t[k]][sp], T2, T1_x[t[k]], y_d)

                while (do_cglp, gen_cut, max_cuts <= 15) == (True, True, True):
                    max_cuts += 1
                    cglp_class.cglp.optimize()
                    if cglp_class.cglp.ObjVal <= 0:
                        print('Faced CGLP <= 0')
                        nie = random.choice([key for key in SP[t[k]][sp].Y_int_keys if y_d[key] != int(y_d[key])])
                        cglp_class.cglp.dispose()
                        do_cglp = False
                    else:
                        PI1 = {x: cglp_class.PI1[x].x for x in SP[t[k]][sp].X}
                        PI2 = {y: cglp_class.PI2[y].x for y in SP[t[k]][sp].Y}
                        PI0 = cglp_class.PI0.x
                        gz += 1
                        Gz[t[k]][sp].append(gz)
                        SP[t[k]][sp].AddCut(PI0, PI1, PI2, gz)
                        print(gz, end=' ')
                        y_d, v_d, bt_d, ut_d, lt_d = SP[t[k]][sp].SolveForBenders(T1_x[t[k]])
                        cglp_class.UpdateModel(PI0, PI1, PI2, y_d)
                        for t2 in T2:
                            if sum([int(T2[t2].Y[y].LB <= y_d[y] <= T2[t2].Y[y].UB) for y in Y_int_keys]) == \
                                    len(Y_int_keys):
                                gen_cut = False
                                break
            if do_cglp:
                SP_y[sp], SP_v[sp], SP_bt[sp], SP_ut[sp], SP_lt[sp] = y_d, v_d, bt_d, ut_d, lt_d
            else:
                w_key = len(SP[t[k]][sp].sub.getVars())+1
                T1[t[k]].X[w_key] = T1[t[k]].master.addVar(ub=1, name=f'w[{w_key}]')
                T1[t[k]].X[w_key].UB, T1[t[k]].X[w_key].LB = 1, 0
                SP[t[k]][sp].wUpdate(w_key, nie)
                for omega in SP[t[k]]:
                    if omega != sp:
                        SP[t[k]][omega].wAdd(w_key)
                # Exit loop convexification
                break
            print('')

        if do_cglp:
            '''Update Upper Bound V'''
            if sum([len(YInt({y: y_test[y] for y in Y_int_keys})) for y_test in SP_y.values()]) == 0:
                print('All y(s) are Int')
                do_update_V = T1_v[t[k]] - T1[t[k]].eta.x + sum([Probs[s] * SP_v[s] for s in Probs])
                if do_update_V < V:
                    V = do_update_V
                    X_star = T1_x[t[k]]
                    Y_star = SP_y
                    nodes_to_remove = [key for key in T1 if T1_v[key] > V]
                    for key in nodes_to_remove:
                        del T1[key]
                        del T1_v[key]
                        del T1_x[key]
            print('Benders cut added.')
            T1[t[k]].AddBendersCut(SP_bt, SP_ut, SP_lt, SP[t[k]])
        B_output = T1[t[k]].Solve()
        T1_x[t[k]], T1_v[t[k]] = B_output[0], B_output[1]
        # print(T1_x[t[k]])
        print(f'V={V:0.2f}     v={v:0.2f}    Gap={100*(V-v)/abs(V):0.2f}%')
        k += 1
    print(f'Optimal Solution is: {X_star}')
    print(f'Optimal Y Solution is: {SP_y}')
    plt.plot(v_list)
    # plt.show()




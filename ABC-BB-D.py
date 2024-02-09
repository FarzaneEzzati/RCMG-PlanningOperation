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


def OpenMasterFile():
    # Load the indices
    with open(f'Models/Master.pkl', 'rb') as omf:
        A, b, upper_bounds, lower_bounds = pickle.load(omf)
    omf.close()

    # Read the saved master problem
    master_model = gp.read('Models/Master.mps', env=env)
    return master_model, A, b, upper_bounds, lower_bounds


def OpenSubFile(scen):
    # Load the indices
    with open(f'Models/Sub{scen}.pkl', 'rb') as osb:
        T, W, r, upper_bounds, lower_bounds = pickle.load(osb)
    osb.close()

    # Read the saved sub-problem
    sub_model = gp.read(f'Models/Sub{scen}.mps', env=env)
    return sub_model, T, W, r, upper_bounds, lower_bounds


with open(f'Models/Indices.pkl', 'rb') as f:
    X_keys, Y_keys, Y_int_keys = pickle.load(f)
f.close()


if __name__ == '__main__':
    D, max_cut = 4, 10
    m_model, m_A, m_b, m_ub, m_lb = OpenMasterFile()
    root_model = MasterProb(m_model, m_A, m_b, m_ub, m_lb, X_keys)
    root_output = root_model.Solve()
    X, ObjVal = root_output[0], root_output[1]
    X_star = None
    v = -float('inf')
    V = float('inf')
    v_list = []
    T1, T1_v, T1_x = {0: root_model}, {0: ObjVal}, {0: X}

    SP = {0: {}}
    for sp in Probs.keys():
        s_model, s_T, s_W, s_r, s_ub, s_lb = OpenSubFile(sp)
        SP[0][sp] = SubProb(s_model, s_T, s_W, s_r, s_ub, s_lb, X_keys)

    k, t, t_last = 1, {}, 0  # Iteration number

    Gz = {0: {sp: [] for sp in Probs.keys()}}
    gz = 0
    epsilon = 0.1   # Stopping tolerance
    print(f'V: {V:0.2f} ====== v: {v:0.2f}')
    while (k < 20, V-v > epsilon) == (True, True):
        print(f'\n{10*">"} Iteration {k}')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Branching Till Integer Found <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        while True:
            t_bar = DictMin(T1_v)
            if T1_v[t_bar] > v:
                v = copy.copy(T1_v[t_bar])
                v_list.append(v)
            non_int = XInt(T1_x[t_bar])
            if not non_int:
                t[k] = t_bar
                break
            # Node splitting
            split_key = random.choice(non_int)
            split_value = T1_x[t_bar][split_key]
            node_1, A_1, b_1, ub_1, lb_1, x_keys_1 = T1[t_bar].ReturnModel()
            for sense in ('u', 'l'):
                t_last += 1
                temp_node = MasterProb(node_1.copy(), copy.copy(A_1), copy.copy(b_1), copy.copy(ub_1), copy.copy(lb_1), copy.copy(x_keys_1))
                temp_node.AddSplit(split_key, split_value, sense)
                temp_output = temp_node.Solve()

                if temp_output != 'InfUnb':
                    T1[t_last], T1_x[t_last], T1_v[t_last] = temp_node, temp_output[0], temp_output[1]

                    SP[t_last] = {}
                    Gz[t_last] = copy.copy(Gz[t_bar])
                    for sp in Probs.keys():
                        s_model, s_T, s_W, s_r, s_ub, s_lb, s_x = SP[t_bar][sp].ReturnModel()
                        SP[t_last][sp] = SubProb(s_model.copy(), copy.copy(s_T), copy.copy(s_W), copy.copy(s_r), copy.copy(s_ub), copy.copy(s_lb), s_x)
            del T1[t_bar]
            del T1_v[t_bar]
            del T1_x[t_bar]
            del SP[t_bar]
        print(f'BB Tree has {len(T1.keys())} node(s). Node {t[k]} is selected')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Benders' Cut Generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        SP_y, SP_v, SP_bt, SP_ut, SP_lt = {}, {}, {}, {}, {}
        SP_T, SP_r, SP_ub, SP_lb = {}, {}, {}, {}

        do_cglp = None
        for sp in SP[t[k]].keys():
            print(f'Convexifying SP {sp}:', end=' ')
            gen_cut, do_cglp, max_cuts = True, True, 0
            y_zero_cglp, joint = None, None

            y_d, v_d, bt_d, ut_d, lt_d = SP[t[k]][sp].SolveForBenders(T1_x[t[k]])
            if not YInt({y_key: y_d[y_key] for y_key in SP[t[k]][sp].YIK}):
                print('All Y Int', end=' ')
                gen_cut = False
            else:
                BBT = SubBB(SP[t[k]][sp], y_d, v_d, T1_x[t[k]], D)
                cglp_class = CGLP(T1[t[k]], SP[t[k]][sp], BBT.T2, T1_x[t[k]], y_d)

                while (do_cglp, gen_cut, max_cuts <= 15) == (True, True, True):
                    max_cuts += 1
                    cglp_class.cglp.optimize()
                    if cglp_class.cglp.ObjVal <= 0:
                        print('Faced CGLP <= 0 ')
                        nie = [key for key in SP[t[k]][sp].YIK if y_d[key] != int(y_d[key])]
                        y_zero_cglp = random.choice(nie)
                        cglp_class.cglp.dispose()
                        do_cglp = False
                    else:
                        PI1 = {x_key: cglp_class.PI1[x_key].x for x_key in SP[t[k]][sp].X.keys()}
                        PI2 = {y_key: cglp_class.PI2[y_key].x for y_key in SP[t[k]][sp].Y.keys()}
                        PI0 = cglp_class.PI0.x
                        gz += 1
                        Gz[t[k]][sp].append(gz)
                        SP[t[k]][sp].AddCut(PI0, PI1, PI2, gz)
                        y_d, v_d, bt_d, ut_d, lt_d = SP[t[k]][sp].SolveForBenders(T1_x[t[k]])
                        cglp_class.UpdateModel(PI0, PI1, PI2, y_d)
                        for t2 in BBT.T2.keys():
                            if sum([int(BBT.T2[t2].lb[i] <= y_d[i] <= BBT.T2[t2].ub[i]) for i in Y_int_keys]) == \
                                    len(Y_int_keys):
                                gen_cut = False
                                break
            if do_cglp:
                print(f'Done. Last cut {gz}')
                SP_y[sp], SP_v[sp], SP_bt[sp], SP_ut[sp], SP_lt[sp] = y_d, v_d, bt_d, ut_d, lt_d
                SP_T[sp], SP_r[sp], SP_ub[sp], SP_lb[sp] = SP[t[k]][sp].T, SP[t[k]][sp].r, SP[t[k]][sp].ub, SP[t[k]][sp].lb
            else:
                w_key = len(SP[t[k]][sp].sub.getVars())+1
                T1[t[k]].X[w_key] = T1[t[k]].master.addVar(ub=1, name=f'w[{w_key}]')
                T1[t[k]].X[w_key].UB, T1[t[k]].X[w_key].LB = 1, 0
                T1[t[k]].ub[w_key], T1[t[k]].lb[w_key] = 1, 0
                SP[t[k]][sp].wUpdate(w_key, y_zero_cglp)
                for omega in SP[t[k]].keys():
                    if omega != sp:
                        SP[t[k]][omega].wAdd(w_key)
                break
        # print(SP[t[k]][2].bt)



        if do_cglp:
            print('Benders cut added.')
            T1[t[k]].AddBendersCut(SP_bt, SP_ut, SP_lt, SP_T, SP_r, SP_ub, SP_lb)
            B_output = T1[t[k]].Solve()
            T1_x[t[k]], T1_v[t[k]] = B_output[0], B_output[1]
            '''Update Upper Bound V'''
            int_flag = True
            if sum([len(YInt({y_key: y_test[y_key] for y_key in Y_int_keys})) for y_test in SP_y.values()]) == 0:
                print('All y(s) are Int')
                do_update_V = T1_v[t[k]] - T1[t[k]].eta.x + sum([Probs[s] * SP_v[s] for s in Probs.keys()])
                if do_update_V < V:
                    V = do_update_V
                    X_star = T1_x[t[k]]
                    nodes_to_remove = [key for key in T1.keys() if T1_v[key] > V]
                    for key in nodes_to_remove:
                        del T1[key]
                        del T1_v[key]
                        del T1_x[key]
        else:
            B_output = T1[t[k]].Solve()
            T1_x[t[k]], T1_v[t[k]] = B_output[0], B_output[1]

        print(f'V={V:0.2f}     v={v:0.2f}    Gap={100*(V-v)/abs(V):0.2f}%')
        k += 1
    print(f'Optimal Solution is: {X_star}')
    plt.plot(v_list)
    # plt.show()

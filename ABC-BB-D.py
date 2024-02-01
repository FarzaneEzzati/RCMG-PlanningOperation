import pickle
from MasterSub import MasterProb, SubProb, Probs
from Convexify import Convexify
from ConvCGSEP import ConvCGSEP
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
    gz = 1
    epsilon = 0.1   # Stopping tolerance
    print(f'V: {V:0.2f} ====== v: {v:0.2f}')
    while (k < 20, len(T1) > 0) == (True, True):
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
            for sense in ('u', 'l'):
                t_last += 1
                node_1, A_1, b_1, ub_1, lb_1, x_1 = T1[t_bar].ReturnModel()
                temp_node = MasterProb(node_1.copy(), copy.copy(A_1), copy.copy(b_1), copy.copy(ub_1), copy.copy(lb_1), copy.copy(x_1))
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
        do_cglp = True
        for sp in SP[t[k]].keys():
            print(f'Convexifying SP {sp}:', end=' ')
            zero_cglp, data = Convexify(T1[t[k]], SP[t[k]][sp], T1_x[t[k]], Gz[t_bar][sp], gz)
            if not zero_cglp[0]:
                w_key, y_d, ub_tau, lb_tau = len(SP[t[k]][sp].sub.getVars()) + 1, zero_cglp[1], zero_cglp[2], zero_cglp[3]
                T1[t[k]].X[w_key] = T1[t[k]].master.addVar(ub=1, name=f'w[{w_key}]')
                T1[t[k]].X[w_key].UB, T1[t[k]].X[w_key].LB = 1, 0
                T1[t[k]].ub[w_key], T1[t[k]].lb[w_key] = 1, 0
                SP[t[k]][sp].wUpdate(y_d, w_key, ub_tau, lb_tau)
                for omega in SP[t[k]].keys():
                    if omega != sp:
                        SP[t[k]][omega].wAdd(w_key)
                break
            else:
                SP_y[sp], SP_v[sp], SP_bt[sp], SP_ut[sp], SP_lt[sp], Gz[t_bar][sp], gz = data[0], data[1], data[2], data[3], data[4], data[5], data[6]
                SP_T[sp], SP_r[sp], SP_ub[sp], SP_lb[sp] = SP[t[k]][sp].T, SP[t[k]][sp].r, SP[t[k]][sp].ub, SP[t[k]][sp].lb

        if zero_cglp[0]:
            print('Benders cut added.')
            T1[t[k]].AddBendersCut(SP_bt, SP_ut, SP_lt, SP_T, SP_r, SP_ub, SP_lb)
            B_output = T1[t[k]].Solve()
            T1_x[t[k]], T1_v[t[k]] = B_output[0], B_output[1]
            if T1_v[t[k]] > v:
                v = T1_v[t[k]]
            '''Update Upper Bound V'''
            int_flag = True
            for y in SP_y.values():
                if YInt({y_key: y[y_key] for y_key in Y_int_keys}):
                    int_flag = False
                    break
            if int_flag:
                print('All y(s) are Int')
                if T1_v[t[k]] < V:
                    V = copy.copy(T1_v[t[k]])
                    X_star = copy.copy(T1_x[t[k]])
                    nodes_to_remove = []
                    for key in T1.keys():
                        if T1_v[key] > V:
                            nodes_to_remove.append(key)
                    for key in nodes_to_remove:
                        del T1[key]
                        del T1_v[key]
                        del T1_x[key]
        else:
            T1_xv = T1[t[k]].Solve()
            T1_x[t[k]], T1_v[t[k]] = T1_xv[0], T1_xv[1]

        print(f'V: {V:0.2f} ====== v: {v:0.2f}')
        k += 1
    print(f'Optimal Solution is: {X_star}')
    plt.plot(v_list)
    # plt.show()

import pickle
from static_functions import *
import gurobipy as gp
from gurobipy import quicksum, GRB
env = gp.Env()
env.setParam('OutputFlag', 0)

'''Define keys to be global parameters'''
with open(f'Models/Indices.pkl', 'rb') as f:
    X_keys, Y_keys, Y_int_keys = pickle.load(f)
f.close()

'''Open all ranges'''
with open('Data/Ranges.pkl', 'rb') as handle:
    RNGLoc, RNGDvc, RNGTime, RNGMonth, RNGHouse, RNGScen, RNGSta, Y_itg, Y_ihtg, Y_ittg = pickle.load(handle)
handle.close()
with open('Data/OutageScenarios.pkl', 'rb') as handle:
    scens, probs = pickle.load(handle)
handle.close()

Probs = {1: 0.5, 2: 0.5}
ScenCount = len(Probs)
D = 10


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


class MasterProb:
    def __init__(self, model, A, b, ubm, lbm):
        self.master, self.A, self.b, self.ub, self.lb = model, A, b, ubm, lbm
        all_vars = self.master.getVars()
        self.X = {key: all_vars[key - 1] for key in X_keys}
        self.eta = self.master.getVarByName('eta')
        self.master.update()
    def Solve(self):
        self.master.optimize()
        if self.master.status == 2:  # Model feasible
            return [{x_key: self.X[x_key].x for x_key in X_keys}, self.master.ObjVal]
        else:
            return 'InfUnb'
    def AddSplit(self, x_key, bound, split_sense):
        # Updating bounds for the corresponding variable
        if split_sense == 'u':
            self.ub[x_key] = int(bound)
            self.X[x_key].UB = int(bound)
        else:
            self.lb[x_key] = int(bound) + 1
            self.X[x_key].LB = int(bound) + 1
        self.master.update()
    def AddBendersCut(self, bt, ut, lt, Tm, rm, ubm, lbm):
        pir = {s: Probs[s] * (sum(bt[s][row] * rm[s][row] for row in rm[s].keys()) -
                              sum(ut[s][y_key] * ubm[s][y_key] for y_key in Y_keys) +
                              sum(lt[s][y_key] * lbm[s][y_key] for y_key in Y_keys))
               for s in Probs.keys()}
        e = sum(pir.values())
        E = {x_key: 0 for x_key in X_keys}
        for s in Probs.keys():
            for key in Tm[s].keys():
                E[key[1]] += Probs[s] * (bt[s][key[0]] * Tm[s][key])
        self.master.addConstr(self.eta + quicksum(self.X[x_key] * E[x_key] for x_key in X_keys) >= e, name='Benders')
        self.master.update()
    def ReturnModel(self):
        return self.master.copy(), copy.copy(self.A), copy.copy(self.b), copy.copy(self.ub), copy.copy(self.lb)
    def ReturnA_b_ub_lb(self):
        return copy.copy(self.A), copy.copy(self.b), copy.copy(self.ub), copy.copy(self.lb)


class SubProb:
    def __init__(self, model, T, W, r, ub, lb):
        # Save the model in the class
        self.sub, self.T, self.W, self.r, self.ub, self.lb = model, T, W, r, ub, lb
        all_vars = self.sub.getVars()
        self.X = {x_key: all_vars[x_key - 1] for x_key in X_keys}
        self.Y = {y_key: all_vars[y_key - 1] for y_key in Y_keys}
    def FixXSolve(self, x):
        y_values, v_value = None, None
        for x_key in self.X.keys():
            self.sub.addConstr(self.X[x_key] == x[x_key], name=f'FixX[{x_key}]')
        self.sub.update()
        self.sub.optimize()
        if self.sub.status == 2:
            # Obtain Y values for all and save in a key-format dictionary
            y_values = {y_key: self.Y[y_key].x for y_key in Y_keys}
            # Save ObjVal
            v_value = self.sub.ObjVal
        for x_key in self.X.keys():
            self.sub.remove(self.sub.getConstrByName(f'FixX[{x_key}]'))
        return y_values, v_value
    def SolveForBenders(self, x):
        y_values, v_value = None, None
        bt, lt, ut = None, None, None
        for x_key in self.X.keys():
            self.sub.addConstr(self.X[x_key] == x[x_key], name=f'FixX[{x_key}]')
        for y_key in Y_keys:
            self.sub.addConstr(self.Y[y_key] <= self.ub[y_key], name=f'Upper[{y_key}]')
            self.sub.addConstr(self.Y[y_key] >= self.lb[y_key], name=f'Lower[{y_key}]')
        self.sub.update()
        self.sub.optimize()

        if self.sub.status == 2:
            y_values = {y_key: self.Y[y_key].x for y_key in Y_keys}
            v_value = self.sub.ObjVal

            all_cons = self.sub.getConstrs()
            bt_cons, ut_cons, lt_cons = [], [], []
            for con in all_cons:
                ConName = con.ConstrName
                if 'FixX' in ConName:
                    pass
                elif 'Upper' in ConName:
                    ut_cons.append(con)
                elif 'Lower' in ConName:
                    lt_cons.append(con)
                else:
                    bt_cons.append(con)

            bt = {row_key + 1: bt_cons[row_key].Pi for row_key in range(len(bt_cons))}
            ut = {y_key: ut_cons[Y_keys.index(y_key)].Pi for y_key in Y_keys}
            lt = {y_key: lt_cons[Y_keys.index(y_key)].Pi for y_key in Y_keys}

        for x_key in self.X.keys():
            self.sub.remove(self.sub.getConstrByName(f'FixX[{x_key}]'))
        for y_key in Y_keys:
            self.sub.remove(self.sub.getConstrByName(f'Upper[{y_key}]'))
            self.sub.remove(self.sub.getConstrByName(f'Lower[{y_key}]'))
        self.sub.update()

        return y_values, v_value, bt, ut, lt
    def AddSplit(self, y_key, split_sense):
        # Update the bound as well in the bound dictionary
        if split_sense == 'u':
            self.Y[y_key].UB = 0
            self.ub[y_key] = 0
        else:
            self.Y[y_key].LB = 1
            self.lb[y_key] = 1
        self.sub.update()
    def ReturnModel(self):
        return self.sub.copy(), copy.copy(self.T), copy.copy(self.W), copy.copy(self.r), copy.copy(self.ub), copy.copy(self.lb)
    def ReturnT_W_r(self):
        return copy.copy(self.T), copy.copy(self.W), copy.copy(self.r)
    def AddCut(self, pi0, pi1, pi2):
        pi1x = quicksum(pi1[x_key] * self.X[x_key] for x_key in X_keys)
        pi2y = quicksum(pi2[y_key] * self.Y[y_key] for y_key in Y_keys)
        self.sub.addConstr(pi2y + pi1x >= pi0, name='CGLP')
        self.sub.update()

        W_new_row = SparseRowCount(self.W) + 1
        T_new_row = SparseRowCount(self.T) + 1
        for t_index in pi1.keys():
            self.T[(T_new_row, t_index)] = pi1[t_index]
        for w_index in pi2.keys():
            self.W[(W_new_row, w_index)] = pi2[w_index]
        self.r[W_new_row] = pi0



def Convexify(master, sub, x_k):
    A_1, b_1, ub_1, lb_1 = master.ReturnA_b_ub_lb()
    y_d, v_d, bt_d, ut_d, lt_d = sub.SolveForBenders(x_k)

    if not YInt({y_key: y_d[y_key] for y_key in Y_int_keys}):
        return y_d, v_d, bt_d, ut_d, lt_d
    else:
        '''BB'''
        temp_sub, T, W, r, ub, lb = sub.ReturnModel()
        T2 = {0: SubProb(temp_sub, T, W, r, ub, lb)}  # Dictionary of active nodes in the seconds stage
        T2_v, T2_y = {0: v_d}, {0: y_d}
        TI2_v = {0: v_d}  # Dictionary saving nodes that are candidates for splitting
        tt_last, d = 0, 1
        while (len(T2) <= D, len(TI2_v) != 0) == (True, True):
            # select the minimum valued node
            tt_bar = DictMin(TI2_v)
            # Save the non-int indices
            y_to_check = {y_key: T2_y[tt_bar][y_key] for y_key in Y_int_keys}
            y_nonint = YInt(y_to_check)
            y_split_key = SelectSplitKey(y_to_check, y_nonint)
            if y_split_key is None:
                del TI2_v[tt_bar]
            else:
                for split_sense in ('u', 'l'):
                    tt_last += 1
                    node_2, T_2, W_2, r_2, ub_2, lb_2 = T2[tt_bar].ReturnModel()
                    temp_split = SubProb(node_2, T_2, W_2, r_2, ub_2, lb_2)
                    temp_split.AddSplit(y_split_key, split_sense)  # Add the split to the new node
                    y2, v2 = temp_split.FixXSolve(x_k)
                    if y2 is not None:
                        T2[tt_last] = temp_split
                        T2_y[tt_last], T2_v[tt_last] = y2, v2
                        TI2_v[tt_last] = T2_v[tt_last]

                del T2[tt_bar]
                del T2_v[tt_bar]
                del T2_y[tt_bar]
                del TI2_v[tt_bar]
        '''CGLP'''
        while True:
            for tt in T2.keys():
                '''Need A and b from master problem. Need t, W, r from the sub-problem. Need upper and lower bounds for both.'''
                if True:
                    T_k, W_k, r_k = sub.ReturnT_W_r()
                    ub_2, lb_2 = T2[tt].ub, T2[tt].lb

                    cglp = gp.Model(env=env)
                    m1 = len(master.master.getConstrs())
                    m2 = len(sub.sub.getConstrs())


                    L1 = cglp.addVars(range(1, m1 + 1), lb=0, name='l1')
                    A_cons = master.master.getConstrs()
                    for a_cons in A_cons:
                        if a_cons.Sense == '=':
                            L1[A_cons.index(a_cons)+1].LB = -GRB.INFINITY

                    Mu1 = cglp.addVars(X_keys, lb=0, name='mu1')
                    Nu1 = cglp.addVars(X_keys, lb=-GRB.INFINITY, ub=0, name='nu1')

                    L2 = cglp.addVars(range(1, m2 + 1), name='l2')
                    W_cons = sub.sub.getConstrs()
                    for w_cons in W_cons:
                        if w_cons.Sense == '=':
                            L2[W_cons.index(w_cons) + 1].LB = -GRB.INFINITY
                    Mu2 = cglp.addVars(Y_keys, lb=0, name='mu2')
                    Nu2 = cglp.addVars(Y_keys, lb=-GRB.INFINITY, ub=0, name='nu2')

                    PI0 = cglp.addVar(lb=-1, ub=1, name='p0')
                    PI1 = cglp.addVars(X_keys, lb=-1, ub=1, name='p1')
                    PI2 = cglp.addVars(Y_keys, lb=-1, ub=1, name='p2')
                    cglp.update()

                    '''PI0'''
                    bL1 = 0
                    for key in b_1.keys():
                        bL1 += b_1[key] * L1[key]
                    rL2 = 0
                    for key in r_k.keys():
                        rL2 += r_k[key] * L2[key]
                    MuL2, NuL2 = 0, 0
                    for key in Y_keys:
                        MuL2 += lb_2[key] * Mu2[key]
                        NuL2 += ub_2[key] * Nu2[key]
                    MuL1, NuL1 = 0, 0
                    for key in X_keys:
                        MuL1 += lb_1[key] * Mu1[key]
                        NuL1 += ub_1[key] * Nu1[key]
                    cglp.addConstr(PI0 <= bL1 + rL2 + MuL2 + NuL2 + MuL1 + NuL1, name='')

                    '''PI1'''
                    TL2 = {x_key: 0 for x_key in X_keys}
                    for key in T_k.keys():
                        TL2[key[1]] += T_k[key] * L2[key[0]]

                    AL1 = {x_key: 0 for x_key in X_keys}
                    for key in A_1.keys():
                        AL1[key[1]] += A_1[key] * L1[key[0]]

                    for x_key in X_keys:
                        cglp.addConstr(PI1[x_key] == AL1[x_key] + TL2[x_key] + Mu1[x_key] + Nu1[x_key], name='')

                    '''PI2'''
                    WL2 = {y_key: 0 for y_key in Y_keys}
                    for key in W_k.keys():
                        WL2[key[1]] += W_k[key] * L2[key[0]]

                    for y_key in Y_keys:
                        cglp.addConstr(PI2[y_key] == WL2[y_key] + Mu2[y_key] + Nu2[y_key])

                    cglp.setObjective(PI0 -
                                      quicksum(PI1[x_key] * x_k[x_key] for x_key in X_keys) -
                                      quicksum(PI2[y_key] * y_d[y_key] for y_key in Y_keys), sense=GRB.MAXIMIZE)
                    cglp.update()
                    cglp.optimize()
                    PI1_values = {x_key: PI1[x_key].x for x_key in X_keys}
                    PI2_values = {y_key: PI2[y_key].x for y_key in Y_keys}
                    PI0_value = PI0.x

                '''Add Cut to the sub-problem sub, solve it again, get u_e and u_g to save as star'''
                sub.AddCut(PI0_value, PI1_values, PI2_values)
                y_d, v_d, bt_d, ut_d, lt_d = sub.SolveForBenders(x_k)
                y_in_union = 0
                for sp2 in T2.values():
                    for int_key in Y_int_keys:
                        if (y_d[int_key] >= sp2.lb[int_key], y_d[int_key] <= sp2.ub[int_key]) != (True, True):
                            y_in_union += 1
                            break
                if y_in_union < len(T2):
                    return y_d, v_d, bt_d, ut_d, lt_d
            d += 1


if __name__ == '__main__':
    m_model, m_A, m_b, m_ub, m_lb = OpenMasterFile()
    root_model = MasterProb(m_model, m_A, m_b, m_ub, m_lb)
    root_output = root_model.Solve()
    X, ObjVal = root_output[0], root_output[1]
    X_star = None
    v = -float('inf')
    V = float('inf')
    T1, T1_v, T1_x = {0: root_model}, {0: ObjVal}, {0: X}

    SP = {0: {}}
    for sp in Probs.keys():
        s_model, s_T, s_W, s_r, s_ub, s_lb = OpenSubFile(sp)
        SP[0][sp] = SubProb(s_model, s_T, s_W, s_r, s_ub, s_lb)

    k, t, t_last = 1, {}, 0  # Iteration number

    epsilon = 0.1   # Stopping tolerance
    while (abs(V-v) > epsilon, len(T1) > 0) == (True, True):
        print(f'\n{10*">"} Iteration {k}')
        print(f'V: {V:0.2f} ====== v: {v:0.2f}')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Branching Till Integer Found <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        while True:
            t_bar = DictMin(T1_v)
            print(T1_x[t_bar])
            if T1_v[t_bar] > v:
                v = copy.copy(T1_v[t_bar])
            non_int = XInt(T1_x[t_bar])
            if not non_int:
                t[k] = t_bar
                break
            # Node splitting
            split_key = random.choice(non_int)
            split_value = T1_x[t_bar][split_key]
            for sense in ('u', 'l'):
                t_last += 1
                temp_model, temp_A, temp_b, temp_ub, temp_lb = T1[t_bar].ReturnModel()
                temp_node = MasterProb(temp_model, temp_A, temp_b, temp_ub, temp_lb)
                temp_node.AddSplit(split_key, split_value, sense)
                temp_output = temp_node.Solve()

                if temp_output != 'InfUnb':
                    T1[t_last], T1_x[t_last], T1_v[t_last] = temp_node, temp_output[0], temp_output[1]

                    SP[t_last] = {}
                    for sp in Probs.keys():
                        s_model, s_T, s_W, s_r, s_ub, s_lb = SP[t_bar][sp].ReturnModel()
                        SP[t_last][sp] = SubProb(s_model, s_T, s_W, s_r, s_ub, s_lb)

            del T1[t_bar]
            del T1_v[t_bar]
            del T1_x[t_bar]
            del SP[t_bar]
        print(f'{T1.keys()} . Node selected {t[k]}')
        print(f'X before benders {T1_x[t[k]]}')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Benders' Cut Generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        '''Add Benders'''

        SP_y, SP_v, SP_bt, SP_ut, SP_lt = {}, {}, {}, {}, {}
        SP_T, SP_r, SP_ub, SP_lb = {}, {}, {}, {}
        for sp in SP[t[k]].keys():
            SP_y[sp], SP_v[sp], SP_bt[sp], SP_ut[sp], SP_lt[sp] = Convexify(T1[t[k]], SP[t[k]][sp], T1_x[t[k]])
            SP_T[sp], SP_r[sp], SP_ub[sp], SP_lb[sp] = SP[t[k]][sp].T, SP[t[k]][sp].r, SP[t[k]][sp].ub, SP[t[k]][sp].lb

        T1[t[k]].AddBendersCut(SP_bt, SP_ut, SP_lt, SP_T, SP_r, SP_ub, SP_lb)

        B_output = T1[t[k]].Solve()
        T1_x[t[k]], T1_v[t[k]] = B_output[0], B_output[1]
        print(f'X after benders {T1_x[t[k]]}')

        '''Update Upper Bound V'''
        int_flag = True
        for y in SP_y.values():
            if YInt({y_key: y[y_key] for y_key in Y_int_keys}):
                int_flag = False
                break
        if int_flag:
            print('Int sol for y found')
            temp_V = T1_v[t[k]]
            if temp_V < V:
                V = temp_V
                X_star = T1_x[t[k]]
                nodes_to_remove = []
                for key in T1.keys():
                    if T1_v[key] > V:
                        nodes_to_remove.append(key)
                for key in nodes_to_remove:
                    del T1[key]
                    del T1_v[key]
                    del T1_x[key]
        print(f'V: {V:0.2f} ====== v: {v:0.2f}')

        k += 1
    print(f'Optimal Solution is: {X_star}')


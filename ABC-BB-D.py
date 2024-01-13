import pickle
import time
import random
import copy
import numpy as np
from static_functions import *
import gurobipy as gp
from gurobipy import quicksum, GRB
env = gp.Env()
env.setParam('OutputFlag', 0)


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


'''Define keys to be global parameters'''
with open(f'Models/Indices.pkl', 'rb') as f:
    X_keys, X_indcies, Y_keys, Y_int_keys = pickle.load(f)
f.close()

'''Open all ranges'''
with open('Data/Ranges.pkl', 'rb') as handle:
    RNGLoc, RNGDvc, RNGTime, RNGMonth, RNGHouse, RNGScen, RNGSta, Y_itg, Y_ihtg, Y_ittg = pickle.load(handle)
handle.close()
with open('Data/OutageScenarios.pkl', 'rb') as handle:
    scens, probs = pickle.load(handle)
handle.close()
Probs = probs[:2]


class MasterProb:
    def __init__(self, model, A, b, ub, lb):
        self.master, self.A, self.b, self.ub, self.lb = model, A, b, ub, lb
        all_vars = self.master.getVars()
        self.X = {key: all_vars[key - 1] for key in X_keys}
        self.eta = all_vars[-1]
        self.master.update()
    def Solve(self):
        self.master.optimize()
        if self.master.status in (1, 2):  # Model feasible
            X_value_keys = {x_key: self.X[x_key].x for x_key in X_keys}
            return [X_value_keys, self.master.ObjVal]
        else:
            return 'InfUnb'
    def AddSplit(self, ild, bound, split_sense):
        # Updating bounds for the corresponding variable
        if split_sense == 'u':
            self.ub[self.X[ild].index+1] = int(bound)
            self.X[ild].UB = int(bound)
        else:
            self.lb[self.X[ild].index+1] = int(bound) + 1
            self.X[ild].LB = int(bound) + 1
        self.master.update()
    def AddBendersCut(self, bt, ut, lt, T, r, ub, lb, x):
        pir = {s: sum(r[s][row] * bt[s][row] for row in r[s].keys()) +
                  sum(ut[s][y_key] * ub[s][y_key] for y_key in Y_keys) +
                  sum(lt[s][y_key] * lb[s][y_key] for y_key in Y_keys)
               for s in Probs.keys()}
        e = sum(Probs[s] * pir[s] for s in Probs)
        E = {x_key: 0 for x_key in X_keys}
        for s in Probs.keys():
            for key in T[s].keys():
                E[key[1]] += Probs[s] * (bt[s][key[0]] * T[s][key])
        # print('rhs for eta', e)
        # print(f'E is {sum(x[x_key] * E[x_key] for x_key in X_keys)}')
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
        self.big_tetha = 0
        self.lower_tetha = 0
        self.upper_tetha = 0
    def FixXSolve(self, x, for_benders=False):
        for x_key in self.X.keys():
            self.sub.addConstr(self.X[x_key] == x[x_key], name=f'FixX[{x_key}]')
        if for_benders:
            for y_key in Y_keys:
                self.sub.addConstr(self.Y[y_key] <= self.ub[y_key], name=f'Upper[{y_key}]')
                self.sub.addConstr(self.Y[y_key] >= self.lb[y_key], name=f'Lower[{y_key}]')
        self.sub.update()
        self.sub.optimize()
        if for_benders:
            all_cons = self.sub.getConstrs()
            big_tetha_cons, upper_cons, lower_cons = [], [], []
            for con in all_cons:
                ConName = con.ConstrName
                if 'FixX' in ConName:
                    pass
                elif 'Upper' in ConName:
                    upper_cons.append(con)
                elif 'Lower' in ConName:
                    lower_cons.append(con)
                else:
                    big_tetha_cons.append(con)
            self.big_tetha = {row_key + 1: big_tetha_cons[row_key].Pi for row_key in range(len(big_tetha_cons))}
            self.lower_tetha = {y_key: lower_cons[Y_keys.index(y_key)].Pi for y_key in Y_keys}
            self.upper_tetha = {y_key: upper_cons[Y_keys.index(y_key)].Pi for y_key in Y_keys}

        # Obtain Y values for all and save in a key-format dictionary
        Y_values = {y_key: self.Y[y_key].x for y_key in Y_keys}
        # Save ObjVal
        ov = self.sub.ObjVal
        for x_key in self.X.keys():
            self.sub.remove(self.sub.getConstrByName(f'FixX[{x_key}]'))
        if for_benders:
            for y_key in Y_keys:
                self.sub.remove(self.sub.getConstrByName(f'Upper[{y_key}]'))
                self.sub.remove(self.sub.getConstrByName(f'Lower[{y_key}]'))
        self.sub.update()
        return Y_values, ov
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
    def ReturnT_W_r_ub_lb(self):
        return copy.copy(self.T), copy.copy(self.W), copy.copy(self.r), copy.copy(self.ub), copy.copy(self.lb)
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

def Convexify(master, sub, x_master, BB_limit):
    A_1, b_1, ub_1, lb_1 = master.ReturnA_b_ub_lb()
    y_c, ov_c = sub.FixXSolve(x_master, for_benders=False)
    ue_c_star, ug_c_star = 0, 0
    cut_counter = 0
    while YInt({y_key: y_c[y_key] for y_key in Y_int_keys})[0]:
        cut_counter += 1
        '''Branch and Bound'''
        temp_sub, T, W, r, ub, lb = sub.ReturnModel()
        T2 = {0: SubProb(temp_sub, T, W, r, ub, lb)}  # Dictionary of active nodes in the seconds stage
        TI2_v = {0: ov_c}  # Dictionary saving nodes that are candidates for splitting
        T2_v, T2_y = {0: ov_c}, {0: y_c}
        tt_last, BB_counter = 0, 0
        '''BB'''
        while (BB_counter <= BB_limit, TI2_v != {}) == (True, True):
            BB_counter += 1
            # select the minimum valued node
            tt_bar = DictMin(TI2_v)

            # Save the non-int indices
            y_nonint, split_key = YInt({y_key: T2_y[tt_bar][y_key] for y_key in Y_int_keys})
            if not y_nonint:
                del TI2_v[tt_bar]
            else:
                for split_sense in ('u', 'l'):
                    tt_last += 1
                    node_2, T_2, W_2, r_2, ub_2, lb_2 = T2[tt_bar].ReturnModel()
                    T2[tt_last] = SubProb(node_2, T_2, W_2, r_2, ub_2, lb_2)
                    T2[tt_last].AddSplit(split_key, split_sense)  # Add the split to the new node
                    T2_y[tt_last], T2_v[tt_last] = T2[tt_last].FixXSolve(x_master, False)
                    TI2_v[tt_last] = T2_v[tt_last]
                del T2[tt_bar]
                del T2_v[tt_bar]
                del T2_y[tt_bar]
                del TI2_v[tt_bar]
            # End of (while d <= D)
        '''CGLP'''
        for tt in T2.keys():
            '''Need A and b from master problem. Need t, W, r from the sub-problem. Need upper and lower bounds for both.'''
            T_2, W_2, r_2, ub_2, lb_2 = T2[tt].ReturnT_W_r_ub_lb()

            cglp = gp.Model(env=env)
            m1 = SparseRowCount(A_1)
            m2 = SparseRowCount(W_2)

            L1 = cglp.addVars(range(1, m1 + 1), name='l1')
            Mu1 = cglp.addVars(X_keys, name='mu1')
            Nu1 = cglp.addVars(X_keys, name='nu1')

            L2 = cglp.addVars(range(1, m2 + 1), name='l2')
            Mu2 = cglp.addVars(Y_keys, name='mu2')
            Nu2 = cglp.addVars(Y_keys, name='nu2')

            PI0 = cglp.addVar(lb=-1, ub=1, name='p0')
            PI1 = cglp.addVars(X_keys, lb=-1, ub=1, name='p1')
            PI2 = cglp.addVars(Y_keys, lb=-1, ub=1, name='p2')
            '''PI0'''
            bL1 = 0
            for key in b_1.keys():
                bL1 += b_1[key] * L1[key]
            rL2 = 0
            for key in r_2.keys():
                rL2 += r_2[key] * L2[key]
            MuL2, NuL2 = 0, 0
            for key in Y_keys:
                MuL2 += lb_2[key] * Mu2[key]
                NuL2 += ub_2[key] * Nu2[key]
            MuL1, NuL1 = 0, 0
            for key in X_keys:
                MuL1 += lb_1[key] * Mu1[key]
                NuL1 += ub_1[key] * Nu1[key]
            cglp.addConstr(PI0 <= -bL1 + rL2 + MuL2 - NuL2 + MuL1 - NuL1, name='')

            '''PI1'''
            TL2 = {x_key: 0 for x_key in X_keys}
            for key in T_2.keys():
                TL2[key[1]] += T_2[key] * L2[key[0]]

            AL1 = {x_key: 0 for x_key in X_keys}
            for key in A_1.keys():
                AL1[key[1]] += A_1[key] * L1[key[0]]

            for x_key in X_keys:
                cglp.addConstr(PI1[x_key] == -AL1[x_key] + TL2[x_key] + Mu1[x_key] - Nu1[x_key], name='')
            '''PI2'''
            WL2 = {y_key: 0 for y_key in Y_keys}
            for key in W_2.keys():
                WL2[key[1]] += W_2[key] * L2[key[0]]

            for y_key in Y_keys:
                cglp.addConstr(PI2[y_key] == WL2[y_key] + Mu2[y_key] - Nu2[y_key])

            cglp.setObjective(PI0 -
                              quicksum(PI1[x_key] * x_master[x_key] for x_key in X_keys) -
                              quicksum(PI2[y_key] * T2_y[tt][y_key] for y_key in Y_keys), sense=GRB.MAXIMIZE)
            cglp.update()
            cglp.optimize()
            PI1_values = {x_key: PI1[x_key].x for x_key in X_keys}
            PI2_values = {y_key: PI2[y_key].x for y_key in Y_keys}
            PI0_value = PI0.x
            '''Add Cut to the sub-problem sub, solve it again, get u_e and u_g to save as star'''
            sub.AddCut(PI0_value, PI1_values, PI2_values)
        '''Resolve'''
        y_c, ov_c = sub.FixXSolve(x_master, for_benders=False)
    return sub.FixXSolve(x_master, for_benders=True)


if __name__ == '__main__':
    m_model, m_A, m_b, m_ub, m_lb = OpenMasterFile()
    root_class = MasterProb(m_model, m_A, m_b, m_ub, m_lb)
    X, ObjVal = root_class.Solve()
    X_star, V_star = 0, -float('inf')
    V = float('inf')  # Upper bound
    v1 = -float('inf')  # Lower bound
    v2 = 0
    T1, T1_v, T1_x = {0: root_class}, {0: ObjVal}, {0: X}

    SP = {0: {}}
    for sp in Probs.keys():
        s_model, s_T, s_W, s_r, s_ub, s_lb = OpenSubFile(sp)
        sp_instance = SubProb(s_model, s_T, s_W, s_r, s_ub, s_lb)
        SP[0][sp] = sp_instance

    k, t, t_last = 1, {}, 0  # Iteration number
    D = 5
    b_cut = 0  # Benders' cut counter
    Gx = {0: []}
    # Gz = [{0: []} for _ in SP]
    epsilon = 0.1   # Stopping tolerance
    while T1 != {}:
        print(f'\n{V_star}')
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Branching Till Integer Found <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        while True:  # while number 1
            t_bar = DictMin(T1_v)

            non_int = XInt(T1_x[t_bar])
            if len(non_int) == 0:
                t[k] = t_bar
                break
            # Node splitting
            temp_ild = random.choice(non_int)  # it is the form of [l, (ii, d)]
            split_value = T1_x[t_bar][temp_ild]
            for sense in ('u', 'l'):
                t_last += 1
                temp_model, temp_A, temp_b, temp_ub, temp_lb = T1[t_bar].ReturnModel()
                temp_node = MasterProb(temp_model, temp_A, temp_b, temp_ub, temp_lb)
                temp_node.AddSplit(temp_ild, split_value, sense)
                temp_output = temp_node.Solve()

                if temp_output != 'InfUnb':
                    T1[t_last], T1_x[t_last], T1_v[t_last] = temp_node, temp_output[0], temp_output[1]
                    Gx[t_last] = copy.copy(Gx[t_bar])

                    SP[t_last] = {}
                    for sp in SP[t_bar].keys():
                        temp_model, temp_T, temp_W, temp_r, temp_ub, temp_lb = SP[t_bar][sp].ReturnModel()
                        SP[t_last][sp] = SubProb(temp_model, temp_T, temp_W, temp_r, temp_ub, temp_lb)
            del T1[t_bar]
            del T1_v[t_bar]
            del T1_x[t_bar]
            del SP[t_bar]
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Benders' Cut Generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        '''Add Benders'''
        print(f'X before: {T1_x[t[k]]}')
        SP_y, SP_v = {}, {}
        SP_bt, SP_ut, SP_lt, SP_T, SP_r, SP_ub, SP_lb = {}, {}, {}, {}, {}, {}, {}
        for sp in SP[t[k]].keys():
            # SP_y[sp], SP_v[sp] = SP[t[k]][sp].FixXSolve(T1_x[t[k]], True)
            SP_y[sp], SP_v[sp] = Convexify(T1[t[k]], SP[t[k]][sp], T1_x[t[k]], D)
            SP_bt[sp], SP_ut[sp], SP_lt[sp], SP_T[sp], SP_r[sp], SP_ub[sp], SP_lb[sp] = SP[t[k]][sp].big_tetha, SP[t[k]][sp].upper_tetha, SP[t[k]][sp].lower_tetha, SP[t[k]][sp].T, SP[t[k]][sp].r, SP[t[k]][sp].ub, SP[t[k]][sp].lb
        T1[t[k]].AddBendersCut(SP_bt, SP_ut, SP_lt, SP_T, SP_r, SP_ub, SP_lb, T1_x[t[k]])
        B_output = T1[t[k]].Solve()
        v_before = copy.copy(T1_v[t[k]])
        T1_x[t[k]] = B_output[0]
        T1_v[t[k]] = B_output[1]
        print(f'Node {t[k]}. v before {v_before:0.2f} and v after {T1_v[t[k]]}')
        print(f'X after: {T1_x[t[k]]}')
        break


        if not XInt(T1_x[t[k]]):
            print(f'Node {t[k]} fathomed')
            X_star = T1_x[t[k]]
            V_star = T1_v[t[k]]
            del T1[t[k]]
            del T1_v[t[k]]
            del T1_x[t[k]]
        else:
            if abs(v_before - T1_v[t[k]]) < epsilon:
                print(f'Node {t[k]} fathomed')
                X_star = T1_x[t[k]]
                V_star = T1_v[t[k]]
                del T1[t[k]]
                del T1_v[t[k]]
                del T1_x[t[k]]
        print(T1)
        k += 1
    print(X_star)



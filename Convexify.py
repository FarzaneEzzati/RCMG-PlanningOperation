import pickle
from static_functions import *
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import quicksum, GRB
from Classes import *
env = gp.Env()
env.setParam('OutputFlag', 0)

D = 2


def Convexify(master, sub, x_k, Gz, gz):
    A_1, b_1, ub_1, lb_1 = master.ReturnA_b_ub_lb()
    y_d, v_d, bt_d, ut_d, lt_d = sub.SolveForBenders(x_k)  # bt = BigTheta, ut = UpperTheta, lt = LowerTheta
    if not YInt({y_key: y_d[y_key] for y_key in Y_int_keys}):
        return y_d, v_d, bt_d, ut_d, lt_d, Gz, gz
    else:
        '''BB'''
        temp_sub, T, W, r, ub, lb = sub.ReturnModel()
        T2 = {0: SubProb(temp_sub, T, W, r, ub, lb)}  # Dictionary of active nodes in the seconds stage
        T2_v, T2_y = {0: v_d}, {0: y_d}
        TI2_v = {0: v_d}  # Dictionary saving nodes that are candidates for splitting
        tt_last, d = 0, 1
        while (len(T2) < D, len(TI2_v) != 0) == (True, True):
            tt_bar = min(TI2_v, key=TI2_v.get)
            ytc = {y_key: T2_y[tt_bar][y_key] for y_key in Y_int_keys}
            ynin = YInt(ytc)
            ysk = SelectSplitKey(ytc, ynin)
            if ysk is None:
                del TI2_v[tt_bar]
            else:
                for split_sense in ('u', 'l'):
                    tt_last += 1
                    node_2, T_2, W_2, r_2, ub_2, lb_2 = T2[tt_bar].ReturnModel()
                    temp_split = SubProb(node_2, T_2, W_2, r_2, ub_2, lb_2)
                    temp_split.AddSplit(ysk, split_sense)  # Add the split to the new node
                    y2, v2 = temp_split.FixXSolve(x_k)
                    if y2 is not None:
                        T2[tt_last] = temp_split
                        T2_y[tt_last], T2_v[tt_last] = y2, v2
                        TI2_v[tt_last] = T2_v[tt_last]

                if (tt_last in T2.keys(), tt_last-1 in T2.keys()) == (False, False):
                    del TI2_v[tt_bar]
                else:
                    del T2[tt_bar]
                    del T2_v[tt_bar]
                    del T2_y[tt_bar]
                    del TI2_v[tt_bar]
        '''CGLP'''
        ub_2 = {t2: T2[t2].ub for t2 in T2.keys()}
        lb_2 = {t2: T2[t2].lb for t2 in T2.keys()}
        cglp = gp.Model('CGLP', env=env)
        A_cons = master.master.getConstrs()
        L1 = cglp.addVars(range(1, len(A_cons) + 1), lb=0, name='l1')
        for c in A_cons:
            if c.Sense == '=':
                L1[A_cons.index(c) + 1].LB = -GRB.INFINITY

        Mu1 = cglp.addVars(X_keys, lb=0, name='mu1')
        Nu1 = cglp.addVars(X_keys, lb=0, name='nu1')
        Mu2 = {t2: cglp.addVars(Y_keys, lb=0, name='mu2') for t2 in T2.keys()}
        Nu2 = {t2: cglp.addVars(Y_keys, lb=0, name='nu2') for t2 in T2.keys()}
        PI0 = cglp.addVar(lb=-1, ub=1, name='p0')
        PI1 = cglp.addVars(X_keys, lb=-1, ub=1, name='p1')
        PI2 = cglp.addVars(Y_keys, lb=-1, ub=1, name='p2')
        cglp.update()

        while True:
            T_k, W_k, r_k = sub.ReturnT_W_r()
            W_cons = sub.sub.getConstrs()
            L2 = {t2: cglp.addVars(range(1, len(W_cons) + 1), name='l2') for t2 in T2.keys()}
            for c in W_cons:
                if c.Sense == '=':
                    for t2 in T2.keys():
                        L2[t2][W_cons.index(c) + 1].LB = -GRB.INFINITY
            '''PI0'''
            rL2 = {t2: quicksum(r_k[key] * L2[t2][key] for key in r_k.keys()) for t2 in T2.keys()}
            MuLo = {t2: quicksum(lb_2[t2][key] * Mu2[t2][key] for key in Y_keys) for t2 in T2.keys()}
            NuUp = {t2: quicksum(ub_2[t2][key] * Nu2[t2][key] for key in Y_keys) for t2 in T2.keys()}


            bL1 = quicksum(b_1[key] * L1[key] for key in b_1.keys())
            MuLo1 = quicksum(lb_1[key] * Mu1[key] for key in X_keys)
            NuUp1 = quicksum(ub_1[key] * Nu1[key] for key in X_keys)
            '''PI1'''
            TL2 = {t2: {x_key: 0 for x_key in X_keys} for t2 in T2.keys()}
            for t2 in T2.keys():
                for key in T_k.keys():
                    TL2[t2][key[1]] += T_k[key] * L2[t2][key[0]]

            AL1 = {x_key: 0 for x_key in X_keys}
            for key in A_1.keys():
                AL1[key[1]] += A_1[key] * L1[key[0]]

            '''PI2'''
            WL2 = {t2: {y_key: 0 for y_key in Y_keys} for t2 in T2.keys()}
            for t2 in T2.keys():
                for key in W_k.keys():
                    WL2[t2][key[1]] += W_k[key] * L2[t2][key[0]]

            cglp.addConstrs(PI0 <= bL1 + rL2[t2] + MuLo[t2] - NuUp[t2] + MuLo1 - NuUp1
                            for t2 in T2.keys())
            cglp.addConstrs(PI1[x_key] == AL1[x_key] + TL2[t2][x_key] + Mu1[x_key] - Nu1[x_key]
                            for t2 in T2.keys() for x_key in X_keys)
            cglp.addConstrs(PI2[y_key] == WL2[t2][y_key] + Mu2[t2][y_key] - Nu2[t2][y_key]
                            for t2 in T2.keys() for y_key in Y_keys)
            cglp.setObjective(PI0 -
                              quicksum(PI1[x_key] * x_k[x_key] for x_key in X_keys) -
                              quicksum(PI2[y_key] * y_d[y_key] for y_key in Y_keys), sense=GRB.MAXIMIZE)
            cglp.update()
            cglp.optimize()
            PI1_values = {x_key: PI1[x_key].x for x_key in X_keys}
            PI2_values = {y_key: PI2[y_key].x for y_key in Y_keys}
            PI0_value = PI0.x

            sub.AddCut(PI0_value, PI1_values, PI2_values)
            # print(sum(PI1_values[x_key] * x_k[x_key] for x_key in X_keys)+sum(PI2_values[y_key] * y_d[y_key] for y_key in Y_keys) >= PI0_value)
            y_d, v_d, bt_d, ut_d, lt_d = sub.SolveForBenders(x_k)
            gz += 1
            Gz.append(gz)
            # print(v_d, end=' ')
            for t2 in T2.keys():
                y_in_union = True
                for int_key in Y_int_keys:
                    if not lb_2[t2][int_key] <= y_d[int_key] <= ub_2[t2][int_key]:
                        y_in_union = False
                        break
                if y_in_union:
                    print('Done')
                    cglp.dispose()
                    return y_d, v_d, bt_d, ut_d, lt_d, Gz, gz
            cglp.remove(L2)
            cglp.remove(cglp.getConstrs())
            cglp.update()
            break
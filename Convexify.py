import pickle
import math
from CGLP import CGLP
from static_functions import *
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import quicksum, GRB
from SubBB import SubBB
from MasterSub import SubProb, MasterProb
import time
env = gp.Env()
env.setParam('OutputFlag', 0)


with open(f'Models/Indices.pkl', 'rb') as f:
    X_keys, Y_keys, Y_int_keys = pickle.load(f)
f.close()
D = 3


def Convexify(master, sub, x_k, Gz, gz):
    w_key = None
    gen_cut, do_cglp, max_cuts = True, True, 0
    bt_d, ut_d, lt_d = None, None, None
    ub_tau, lb_tau = None, None

    while (gen_cut, do_cglp) == (True, True):
        y_d, v_d = sub.SolveForBenders(x_k)
        if not YInt({y_key: y_d[y_key] for y_key in sub.YIK}):
            print('All Y Int.')
            gen_cut = False
        else:
            BBT = SubBB(sub, y_d, v_d, x_k, D)
            cglp_class = CGLP(master, sub, BBT.T2, x_k, y_d)
            do_cglp = True

            while (do_cglp, max_cuts <= 15) == (True, True):
                max_cuts += 1
                cglp_class.cglp.optimize()
                # print(cglp_class.cglp.ObjVal)
                if cglp_class.cglp.ObjVal <= 0:
                    print('Faced CGLP <= 0')
                    violated = {}
                    for t2 in BBT.T2.keys():
                        violated[t2] = sum([int(BBT.T2[t2].lb[i] <= y_d[i] <= BBT.T2[t2].ub[i]) for i in Y_int_keys])
                    attention_leaf = min(violated, key=violated.get)
                    ub_tau = BBT.T2[attention_leaf].ub
                    lb_tau = BBT.T2[attention_leaf].lb
                    cglp_class.cglp.dispose()
                    do_cglp = False
                else:
                    PI1 = {x_key: cglp_class.PI1[x_key].x for x_key in sub.X.keys()}
                    PI2 = {y_key: cglp_class.PI2[y_key].x for y_key in sub.Y.keys()}
                    PI0 = cglp_class.PI0.x
                    gz += 1
                    Gz.append(gz)
                    sub.AddCut(PI0, PI1, PI2, gz)
                    cglp_class.Update(PI0, PI1, PI2)
                    y_d, v_d = sub.SolveForBenders(x_k)
                    '''print(f'{gz}', end=' ')'''
                    for t2 in BBT.T2.keys():
                        if sum([int(BBT.T2[t2].lb[i] <= y_d[i] <= BBT.T2[t2].ub[i]) for i in Y_int_keys]) == len(Y_int_keys):
                            gen_cut = False
                            break
        if do_cglp:
            bt_d, ut_d, lt_d = sub.bt, sub.ut, sub.lt
        print(f'Done. Last cut {gz}')
        return [do_cglp, y_d, ub_tau, lb_tau], [y_d, v_d, bt_d, ut_d, lt_d, Gz, gz]

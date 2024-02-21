import pickle
from static_functions import *
import gurobipy as gp
from gurobipy import quicksum, GRB
import math

with open('Data/OutageScenarios.pkl', 'rb') as handle:
    scens, probs = pickle.load(handle)
handle.close()

Probs = {1: 0.5, 2: 0.5}



class MasterProb:
    def __init__(self, model, X_keys):
        self.master = model
        all_vars = self.master.getVars()
        self.eta = self.master.getVarByName('eta')
        for i in range(len(all_vars)):
            if all_vars[i].VarName == 'eta':
                break
        all_Xs = [all_vars[index] for index in range(len(all_vars)) if index != i]
        self.X = {X_keys[index]: all_Xs[index] for index in range(len(X_keys))}
        self.master.update()
    def Solve(self):
        self.master.optimize()
        if self.master.status == 2:  # Model feasible
            return [{x: self.X[x].x for x in self.X}, self.master.ObjVal]
        else:
            return 'None'
    def AddSplit(self, x, bound, split_sense):
        # Updating bounds for the corresponding variable
        if split_sense == 'u':
            self.X[x].UB = int(bound)
        else:
            self.X[x].LB = int(bound) + 1
        self.master.update()
    def AddBendersCut(self, bt, ut, lt, SP):
        Tm, rm, ubm, lbm = {}, {}, {}, {}
        for scen in SP:
            Am = IndexUp(SP[scen].sub.getA().todok())
            Constrs_1 = SP[scen].sub.getConstrs()
            Tm[scen], rm[scen], ubm[scen], lbm[scen] = {}, {}, {}, {}
            for key in Am:
                if key[1] in SP[scen].X:
                    Tm[scen][key] = Am[key]
            for c in Constrs_1:
                rm[scen][Constrs_1.index(c) + 1] = c.RHS
            for y in SP[scen].Y:
                ubm[scen][y] = SP[scen].Y[y].UB
                lbm[scen][y] = SP[scen].Y[y].LB

        pir = {s: Probs[s] * (sum(bt[s][row] * rm[s][row] for row in bt[s]) +
                              sum(ut[s][y] * ubm[s][y] for y in ut[s]) +
                              sum(lt[s][y] * lbm[s][y] for y in lt[s]))
               for s in Probs}
        e = sum(pir.values())
        Es = {s: {x: 0 for x in self.X} for s in Probs}
        for s in Probs:
            for key in Tm[s]:
                Es[s][key[1]] += bt[s][key[0]] * Tm[s][key]
        E = {x: sum(Es[s][x] * Probs[s] for s in Probs) for x in self.X}
        self.master.addConstr(self.eta + quicksum(self.X[x] * E[x] for x in self.X) >= e, name='Benders')
        # print(self.eta + quicksum(self.X[x] * E[x] for x in self.X) >= e)
        self.master.update()
    def ReturnModel(self):
        return self.master.copy(), list(self.X)
    def AddNeww(self, xkey):
        self.X[xkey] = self.master.addVar(ub=1, name='w')
        self.master.update()


class SubProb:
    def __init__(self, model, X_keys, Y_keys, Y_int_keys):
        # Save the model in the class
        self.sub = model
        self.Y_int_keys = Y_int_keys
        all_vars = self.sub.getVars()
        self.X = {X_keys[index]: all_vars[index] for index in range(len(X_keys))}
        self.Y = {y: all_vars[y - 1] for y in Y_keys}

    def FixXSolve(self, x_k):
        y_values, v_value = None, None
        for x in self.X:
            self.X[x].UB = x_k[x]
            self.X[x].LB = x_k[x]
        self.sub.update()
        self.sub.optimize()
        if self.sub.status == 2:
            y_values = {y: self.Y[y].x for y in self.Y}
            v_value = self.sub.ObjVal
        return y_values, v_value
    def SolveForBenders(self, x_k):
        y_values, v_value = None, None
        bt, ut, lt = None, None, None
        for x in self.X:
            self.X[x].UB = x_k[x]
            self.X[x].LB = x_k[x]
        self.sub.update()
        self.sub.optimize()

        if self.sub.status == 2:
            y_values = {y: self.Y[y].x for y in self.Y}
            v_value = self.sub.ObjVal
            bt_cons, ut_cons, lt_cons = [], [], []
            bt_rows = []
            constrs_2 = self.sub.getConstrs()
            for con in constrs_2:
                ConName = con.ConstrName
                if 'Upper' in ConName:
                    ut_cons.append(con)
                elif 'Lower' in ConName:
                    lt_cons.append(con)
                else:
                    bt_cons.append(con)
                    bt_rows.append(constrs_2.index(con) + 1)

            bt = {bt_rows[j]: bt_cons[j].Pi for j in range(len(bt_cons))}
            ut = {list(self.Y)[i]: ut_cons[i].Pi for i in range(len(ut_cons))}
            lt = {list(self.Y)[i]: lt_cons[i].Pi for i in range(len(lt_cons))}
        else:
            self.sub.computeIIS()
            for c in self.sub.getConstrs():
                if c.IISConstr:
                    print(c.ConstrName)
            raise ValueError('Scenario infeasible')

        return y_values, v_value, bt, ut, lt
    def AddSplit(self, y, split_sense):
        # Update the bound as well in the bound dictionary
        if split_sense == 'u':
            self.Y[y].UB = 0
        else:
            self.Y[y].LB = 1
        self.sub.update()
    def ReturnModel(self):
        return self.sub.copy(), list(self.X), list(self.Y), self.Y_int_keys
    def AddCut(self, pi0, pi1, pi2, gz):
        pi1x = quicksum(pi1[x] * self.X[x] for x in self.X)
        pi2y = quicksum(pi2[y] * self.Y[y] for y in self.Y)
        self.sub.addConstr(pi2y + pi1x >= pi0, name=f'CGLP{gz}')
        self.sub.update()
        # print(pi2y + pi1x >= pi0)

    def wUpdate(self, w_key, y):
        self.X[w_key] = self.sub.addVar(ub=1, name=f'w[{w_key}-{y}]')
        self.sub.addConstr(self.Y[y] >= self.X[w_key], name='w_lb')
        self.sub.addConstr(-self.Y[y] >= -self.X[w_key], name='w_ub')
        self.sub.update()

    def wAdd(self, w_key):
        self.X[w_key] = self.sub.addVar(ub=1, name=f'w[{w_key}]')
        self.sub.update()

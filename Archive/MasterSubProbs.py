import pickle
import gurobipy as gp
from gurobipy import quicksum, GRB
import math


class MasterProb:
    def __init__(self, model, Xkeys):
        self.master = model
        all_vars = self.master.getVars()
        self.eta = self.master.getVarByName('eta')
        self.X = {x: all_vars[x] for x in range((len(all_vars))) if 'eta' not in all_vars[x].VarName}
        self.master.update()
    def Solve(self):
        self.master.optimize()
        if self.master.status == 2:  # Model feasible
            return [{x: self.X[x].x for x in self.X}, self.master.ObjVal]
        else:
            '''self.master.computeIIS()
            for c in self.master.getConstrs():
                if c.IISConstr:
                    print(c.ConstrName)'''
            return 'None'
    def AddSplit(self, x, bound, sense):
        # Updating bounds for the corresponding variable
        if sense == 'u':
            self.X[x].UB = int(bound)
        else:
            self.X[x].LB = int(bound) + 1
        self.master.update()

    def AddBendersCut(self, PIs, SP, TMatrices, rVectors, Probs):
        pir = {sp: Probs[sp] * sum(PIs[sp][row] * rVectors[sp][row] for row in rVectors[sp].keys())
               for sp in Probs.keys()}
        e = sum(pir.values())
        E = {sp: {x: 0 for x in self.X.keys()} for sp in Probs.keys()}
        for sp in Probs.keys():
            for key in TMatrices[sp].keys():
                #print(key)
                E[sp][key[1]] += PIs[sp][key[0]] * TMatrices[sp][key]
        E = {x: sum(E[sp][x] * Probs[sp] for sp in Probs.keys()) for x in self.X.keys()}
        self.master.addConstr(self.eta + quicksum(self.X[x] * E[x] for x in self.X.keys()) >= e, name='Benders')
        # print(self.eta + quicksum(self.X[x] * E[x] for x in self.X) >= e)
        self.master.update()


class SubProb:
    def __init__(self, model, Xkeys):
        # Save the model in the class
        self.sub = model
        all_vars = self.sub.getVars()
        self.X = {i: all_vars[i] for i in Xkeys}
        self.sub.update()

    def SolveForBenders(self, xx):
        PI = None
        for x in self.X:
            self.X[x].UB = xx[x]
            self.X[x].LB = xx[x]
        self.sub.update()
        self.sub.optimize()
 
        if self.sub.status == 2:
            constrs = self.sub.getConstrs()
            PI = {c: constrs[c].Pi for c in range(len(constrs))}
        else:

            self.sub.computeIIS()
            for c in self.sub.getConstrs():
                if c.IISConstr:
                    print(c.ConstrName)
            raise ValueError('Scenario infeasible')
        return PI


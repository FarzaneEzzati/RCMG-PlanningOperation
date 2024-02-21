import pickle
from static_functions import *
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
    def AddBendersCut(self, PIs, SP, Probs):
        TMatrix = {sp: {} for sp in SP}
        rVector = {sp: {} for sp in SP}
        for sp in SP:
            AMatrix = IndexUp(SP[sp].sub.getA().todok())
            Constrs = SP[sp].sub.getConstrs()
            for key in AMatrix:
                if key[1] in SP[sp].X:
                    TMatrix[sp][key] = AMatrix[key]
            for c in Constrs:
                rVector[sp][Constrs.index(c) + 1] = c.RHS

        pir = {sp: Probs[sp] * sum(PIs[sp][row] * rVector[sp][row] for row in rVector[sp])
               for sp in Probs}
        e = sum(pir.values())
        E = {sp: {x: 0 for x in self.X} for sp in Probs}
        for sp in Probs:
            for key in TMatrix[sp]:
                E[sp][key[1]] += PIs[sp][key[0]] * TMatrix[sp][key]
        E = {x: sum(E[sp][x] * Probs[sp] for sp in Probs) for x in self.X}
        self.master.addConstr(self.eta + quicksum(self.X[x] * E[x] for x in self.X) >= e, name='Benders')
        # print(self.eta + quicksum(self.X[x] * E[x] for x in self.X) >= e)
        self.master.update()


class SubProb:
    def __init__(self, model):
        # Save the model in the class
        self.sub = model
        all_vars = self.sub.getVars()
        self.X = {i: all_vars[i] for i in range(len(all_vars)) if 'X' in all_vars[i].VarName}
        self.sub.update()

    def SolveForBenders(self, x_k):
        PI = None
        for x in self.X:
            self.X[x].UB = x_k[x]
            self.X[x].LB = x_k[x]
        self.sub.update()
        self.sub.optimize()

        if self.sub.status == 2:
            constrs = self.sub.getConstrs()
            PI = {constrs.index(c)+1: c.Pi for c in constrs}
        else:
            self.sub.computeIIS()
            for c in self.sub.getConstrs():
                if c.IISConstr:
                    print(c.ConstrName)
            raise ValueError('Scenario infeasible')
        return PI, self.sub.ObjVal


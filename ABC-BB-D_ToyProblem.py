import pickle
import time
import random
import copy
import numpy as np
from static_functions import DictMin, IndexUp, XInt
import gurobipy
from gurobipy import quicksum, GRB
env = gurobipy.Env()
env.setParam('OutputFlag', 0)


class EquiDeter:
    def __init__(self):
        model = gurobipy.Model('EquiDeter', env=env)
        self.indices = [1, 2]
        X = model.addVars(self.indices, lb=0, ub=1, name='X')
        Y1 = model.addVars([1, 2, 3, 4], lb=0, ub=5, name='Y1')
        Y2 = model.addVars([1, 2, 3, 4], lb=0, ub=5, name='Y2')
        R = model.addVars([1, 2])
        model.addConstr(2 * Y1[1] - 3 * Y1[2] + 4 * Y1[3] + 5 * Y1[4] - R[1] + X[1] <= 10, name='C1-1')
        model.addConstr(6 * Y1[1] + Y1[2] + 3 * Y1[3] + 2 * Y1[4] - R[1] + X[2] <= 4, name='C2-1')
        model.addConstr(2 * Y2[1] - 3 * Y2[2] + 4 * Y2[3] + 5 * Y2[4] - R[2] + X[1] <= 6, name='C1-2')
        model.addConstr(6 * Y2[1] + Y2[2] + 3 * Y2[3] + 2 * Y2[4] - R[2] + X[2] <= 2, name='C2-2')
        model.setObjective(- 1.5 * X[1] - 4 * X[2] +
                           0.5 * (-16 * Y1[1] - 19 * Y1[2] - 23 * Y1[3] - 28 * Y1[4] + 100 * R[1]) +
                           0.5 * (-16 * Y2[1] - 19 * Y2[2] - 23 * Y2[3] - 28 * Y2[4] + 100 * R[2]), GRB.MINIMIZE)
        model.optimize()
        print(model.ObjVal)
        print(X)


class MasterProb:
    def __init__(self):
        # Load the indices
        with open(f'Models/MasterToy-indices.pkl', 'rb') as handle:
            self.indices = pickle.load(handle)
        handle.close()

        # Read the saved master problem
        self.master = gurobipy.read('Models/MasterToy.mps', env=env)

        # Indicate the X vars
        self.X = {}
        for i in self.indices:
            self.X[i] = self.master.getVarByName(f'X[{i}]')
        self.eta = self.master.getVarByName('eta')

    def Solve(self):
        self.master.optimize()
        if self.master.status in (1, 2):  # Model feasible
            return [{i: self.X[i].x for i in self.indices}, self.master.ObjVal]
        else:
            return 'InfUnb'

    def AddSplit(self, element, bound, sense):
        if sense == 'upper':
            self.master.addConstr(self.X[element] <= int(bound), name='UpperBound')
        else:
            self.master.addConstr(self.X[element] >= int(bound) + 1, name='LowerBound')
        self.master.update()
    def AddBendersCut(self, pi_r, pi_T, pr):
        e = np.sum([pr[i] * pi_r[i-1] for i in pr.keys()])
        E = 0
        for scen in pr.keys():
            E += pr[scen] * quicksum(pi_T[scen-1][key] * self.X[key] for key in pi_T[scen-1].keys())
        self.master.addConstr(self.eta + E >= e, name='Benders')
        self.master.update()


class SubProb:
    def __init__(self, name):
        # Load the indices
        with open(name, 'rb') as handle:
            self.X_indices, self.Y_indices, self.XX, self.YY = pickle.load(handle)
        handle.close()

        # Read the saved sub-problem
        self.sub = gurobipy.read(name, env=env)

        # Indicate the X vars
        self.X = {}
        for i in self.X_indices:
            self.X[i] = self.sub.getVarByName(f'X[{i}]')

        # Indicate the Y vars
        self.Y = {}
        for i in [1, 2, 3, 4]:
            self.Y[i] = self.sub.getVarByName(f'Y{i}')

        self.R = self.sub.getVarByName('R')


    '''This function fixes variables x in sub-problems,
     and then resolve the sub-problem.'''
    def FixXSolve(self, x):
        for i in self.X_indices:
            self.sub.addConstr(self.X[i] == x[i], name=f'FixX{i}')
        self.sub.update()
        self.sub.optimize()
        pi_list = {}
        counter = 1
        for constraint in self.sub.getConstrs():
            if 'FixX' not in constraint.ConstrName:
                pi_list[counter] = constraint.Pi
                counter += 1
        for i in self.X_indices:
            self.sub.remove(self.sub.getConstrByName(f'FixX{i}'))
        self.sub.update()

        return pi_list

    def Convexify(self):
        pass

    def AddBound(self, YBound):
        # YBound is a list with two elements.
        # First element defines the element index to be bounded.
        # Second element determines the bound value.
        pass


    '''This function returns T, W, and r functions in sparse format.
    Note that the indexes start from 0.
    Using YY and XX the indices for X and Y are distinguished.'''
    def GetTWr(self):
        Coefs = IndexUp(self.sub.getA().todok())  # dictionary that indexed up
        # Increase indexes by one to avoid 0 index
        m = np.max([key[0] for key in Coefs.keys()])  # m only covers the count of constraints that we need
        T_sparse = {}
        W_sparse = {}
        r_sparse = {}
        ConstrList = self.sub.getConstrs()

        for row in range(1, m+1):
            for yindex in self.YY:
                if (row, yindex) in Coefs:
                    W_sparse[(row, yindex)] = Coefs[(row, yindex)]
            for xindex in self.XX:
                if (row, xindex) in Coefs:
                    T_sparse[(row, xindex)] = Coefs[(row, xindex)]
            r_sparse[row] = ConstrList[row-1].rhs

        # Note that T, W, r are spares matrices in a dictionary format.
        return T_sparse, W_sparse, r_sparse



if __name__ == '__main__':
    Probs = {1: 0.5, 2: 0.5}

    T1 = {0: MasterProb()}
    X, v = T1[0].Solve()
    T1_v = {0: v}
    T1_x = {0: X}
    # T1 is a list of master problems at node t (the start is with root node 0).
    # For a newly created node, the class instance is coppied.

    SP = {0: [SubProb('Models/SubToy1.mps'), SubProb('Models/SubToy2.mps')]}
    # SP is the list of all sub-problem instances at node t (the key of dictionary).
    # Not that for each node t we have different sub-problem instances.


    '''From now on, we write the algorithm in a general form. 
    Avoid figures and numbers directly obtained from the models.'''
    k = 1  # Iteration number
    t = {}  # Dictionary saving the iteration number of each node. It denotes the subset of solutions as well.
    t_last = 0  # A variable saving the last node created
    V = float('inf')  # Upper bound

    b_cut = 0  # Benders' cut counter

    Gx = {0: []}
    # Gx is a dictionary that determines the cuts added to the master problem.
    # The key is the node number and the key item is list of cuts.

    Gz = [{0: []} for _ in SP]
    # Gz A list of dictionaries that each dictionary corresponds to a scenario.
    # In each dictionary, the ket is node number, and
    # key item is a list of cuts added the that sub-problem.

    epsilon = 0.0001   # Stopping tolerance
    terminate = 0
    while terminate < 2:
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Branching Till Integer Found <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        while True:  # while number 1

            # Update the global lower bound + get the corresponding node t-bar
            t_bar, v = DictMin(T1_v)

            # Check if x(t_bar) is integer. If non-integer, non-int could be [(0, 2)] that means X1 element third
            non_int = XInt(T1_x[t_bar])

            # Save the iteration k of node t_bar
            if len(non_int) == 0:
                t[k] = t_bar
                break

            # Node splitting
            split_element = random.choice(non_int)
            for sense in ('upper', 'lower'):
                t_last += 1
                temp_node = copy.copy(T1[t_bar])
                temp_node.AddSplit(split_element, T1_x[t_bar][split_element], sense)
                temp_output = temp_node.Solve()

                if temp_output != 'InfUnb':
                    # Model feasible, so added to the active nodes
                    T1[t_last] = temp_node
                    T1_x[t_last], T1_v[t_last] = temp_output[0], temp_output[1]
                    # Create Gx and Gz
                    Gx[t_last] = copy.copy(Gx[t_bar])
                    for Gz_sp in Gz:
                        Gz_sp[t_last] = copy.copy(Gz_sp[t_bar])

            # Remove splitted node from the active nodes
            del T1[t_bar]
            del T1_v[t_bar]
            del T1_x[t_bar]
            # end of while number 1


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Benders' Cut Generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # We now have t_bar that determines which x need to be used for the next step.
        # Here we need to generate Benders' cuts
        PIRs = []  # a list
        PITs = []  # a list
        for sp in SP[t[k]]:
            # Get pi as dictionary. keys start from 1.
            pi = sp.FixXSolve(T1_x[t[k]])
            # Get T, r
            T, W, r = sp.GetTWr()
            # Calculate pi*r and pi*T and save in PIR and PIT
            # pi*r is supposed to be scaler.
            PIR = np.sum([pi[i] * r[i] for i in pi.keys()])
            PIRs.append(PIR)
            # pi*T is supposed to be 1*n1.
            PIT = {}
            for j in T1_x[t[k]].keys():
                temp_j_pit = 0
                for i in pi.keys():
                    if (i, j) in T.keys():
                        temp_j_pit += T[(i, j)] * pi[i]
                PIT[j] = temp_j_pit
            PITs.append(PIT)
        # Generate cut and add it to the master problem in node t[k]
        T1[t[k]].AddBendersCut(PIRs, PITs, Probs)

        # Solve the model corresponding to the node with added benders cut
        T1_x[t[k]], T1_v[t[k]] = T1[t[k]].Solve()

        # Add new Benders cut to Gx
        b_cut += 1
        Gx[t[k]].append(b_cut)

        # Increase iteration
        k += 1
        terminate += 1
    print(f'B&B: {v}')
    EquiDeter()




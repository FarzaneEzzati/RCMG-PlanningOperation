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
    with open(f'Models/Master.pkl', 'rb') as handle:
        A, b, upper_bounds, lower_bounds = pickle.load(handle)
    handle.close()

    # Read the saved master problem
    master_model = gp.read('Models/Master.mps', env=env)
    return master_model, A, b, upper_bounds, lower_bounds


def OpenSubFile(scen):
    # Load the indices
    with open(f'Models/Sub{scen}.pkl', 'rb') as handle:
        T, W, r, upper_bounds, lower_bounds = pickle.load(handle)
    handle.close()

    # Read the saved sub-problem
    sub_model = gp.read(f'Models/Sub{scen}.mps', env=env)
    return sub_model, T, W, r, upper_bounds, lower_bounds


'''Define keys to be global parameters'''
with open(f'Models/Indices.pkl', 'rb') as f:
    X_keys, X_indcies, XY_keys = pickle.load(f)
f.close()

'''Open all ranges'''
with open('Data/Ranges.pkl', 'rb') as handle:
    RNGLoc, RNGDvc, RNGTime, RNGMonth, RNGHouse, RNGScen, RNGSta, Y_tg, Y_htg, Y_ttg= pickle.load(handle)
handle.close()


class MasterProb:
    def __init__(self, model, A, b, ub, lb):
        # Save the data in the class
        self.master, self.A, self.b, self.ub, self.lb = model, A, b, ub, lb
        self.X_indices = X_indcies
        self.X_keys = X_keys

        # Indicate the X vars
        self.X = {}  # X can be treated as the original var in Master Prob
        for l in self.X_indices.keys():
            self.X[l] = {}
            for i in self.X_indices[l]:
                self.X[l][(i[0], i[1])] = self.master.getVarByName(f'X[{l}][({i[0]},{i[1]})]')
        self.eta = self.master.getVarByName('eta')
        self.master.update()

    def Solve(self):
        self.master.optimize()
        if self.master.status in (1, 2):  # Model feasible
            X_star = {l: {ii_d: self.X[l][ii_d].x for ii_d in self.X_indices[l]} for l in self.X_indices.keys()}
            return [X_star, self.master.ObjVal]
        else:
            return 'InfUnb'
    def AddSplit(self, indices, bound, sense):
        # Updating bounds for the corresponding variable
        l, ii_d = indices[0], indices[1]  # indices must be the form [l, (ii,d)]
        if sense == 'upper':
            self.ub[l][ii_d] = int(bound)
            self.X[l][ii_d].UB = int(bound)
        else:
            self.lb[l][ii_d] = int(bound) + 1
            self.X[l][ii_d].LB = int(bound) + 1
        self.master.update()
    def AddBendersCut(self, pi_r, pi_T, pr):
        e = np.sum([pr[i] * pi_r[i-1] for i in pr.keys()])
        E = 0
        for scen in pr.keys():
            E += pr[scen] * quicksum(pi_T[scen-1][key] * self.X[key] for key in pi_T[scen-1].keys())
        self.master.addConstr(self.eta + E >= e, name='Benders')
        self.master.update()

    def ReturnModel(self):
        return self.master.copy(), copy.copy(self.A), copy.copy(self.b), copy.copy(self.ub), copy.copy(self.lb)


class SubProb:
    def __init__(self, model, T, W, r, ub, lb):
        # Save the model in the class
        self.sub, self.T, self.W, self.r, self.ub, self.lb = model, T, W, r, ub, lb
        self.XY_keys = XY_keys
        self.X_keys = X_keys
        self.X_indices = X_indcies


        # Indicate the X vars
        self.X = {}
        for l in self.X_indices.keys():
            self.X[l] = {}
            for i in self.X_indices[l]:
                self.X[l][i] = self.sub.getVarByName(f'X[{l}][({i[0]},{i[1]})]')

        # Indicate U_E and U_G vars
        self.U_E = {i: {} for i in RNGSta}  # Charge/discharge binary
        self.U_G = {i: {} for i in RNGSta}  # Import/export binary
        for ii in RNGSta:
            for tg in Y_tg:
                self.U_E[ii][tg] = self.sub.getVarByName(f'U_E[{ii}][({tg[0]},{tg[1]})]')
                self.U_G[ii][tg] = self.sub.getVarByName(f'U_G[{ii}][({tg[0]},{tg[1]})]')


    def FixXSolve(self, x):
        for l in self.X_indices.keys():
            for i in self.X_indices[l]:
                self.sub.addConstr(self.X[l][i] == x[l][i], name=f'FixX[{l}][({i[0]},{i[1]})]')
        self.sub.update()
        self.sub.optimize()

        # Get constraint multipliers
        pi_list = {}
        counter = 1
        for constraint in self.sub.getConstrs():
            if 'FixX' not in constraint.ConstrName:
                pi_list[counter] = constraint.Pi
                counter += 1

        # Obtain Y values
        U_E = {i: {} for i in RNGSta}  # Charge/discharge binary
        U_G = {i: {} for i in RNGSta}  # Import/export binary
        for ii in RNGSta:
            for tg in Y_tg:
                U_E[ii][tg] = self.U_E[ii][tg].x
                U_G[ii][tg] = self.U_G[ii][tg].x

        # Save ObjVal
        ov = self.sub.ObjVal
        for l in self.X_indices.keys():
            for i in self.X_keys:
                self.sub.remove(self.sub.getConstrByName(f'FixX[{l}][({i[0]},{i[1]})]'))
        self.sub.update()
        return pi_list, U_E, U_G, ov

    def AddSplit(self, name, ii_tg, sense):
        # Update the bound as well in the bound dictionary
        if name == 'U_E':
            if sense == 'upper':
                self.U_E[ii_tg[0]][ii_tg[1]].UB = 0
                self.ub['U_E'][ii_tg[0]][ii_tg[1]] = 0
            else:
                self.U_E[ii_tg[0]][ii_tg[1]].LB = 1
                self.lb['U_E'][ii_tg[0]][ii_tg[1]] = 1
        else:
            if sense == 'upper':
                self.U_G[ii_tg[0]][ii_tg[1]].UB = 0
                self.ub['U_G'][ii_tg[0]][ii_tg[1]] = 0
            else:
                self.U_G[ii_tg[0]][ii_tg[1]].LB = 1
                self.lb['U_E'][ii_tg[0]][ii_tg[1]] = 1
        self.sub.update()


    def AddBound(self, YBound):
        # YBound is a list with two elements.
        # First element defines the element index to be bounded.
        # Second element determines the bound value.
        pass
    def UpdateTWr(self):
        pass

    def ReturnModel(self):
        return self.sub.copy(), copy.copy(self.T), copy.copy(self.W), copy.copy(self.r), copy.copy(self.ub), copy.copy(self.lb)

def Convexify(sub, x, D):
    # First a Branch and Bound with D nodes is applied here. Then using the nodes, cuts are generated fo the problem
    # The cuts are added, problem solved again, and if still not integer, the problem is solved again.
    pi_c, y_c, ov_c = sub.FixXSolve(x)
    y_c_star = 0
    while True:

        # Get solution y and see if is integer
        all_int = [y_c[i] == int(y_c[i]) for i in sub.Y_indices]
        if np.sum(all_int) == 0:
            # if integer, break. y_c_star the the solution passed as the integer solution obtained after convexification
            y_c_star = y_c
            break

        else:
            # Do branch and bound
            T2 = {0: sub}  # Dictionary of active nodes in the seconds stage
            T2_v = {0: ov_c}
            T2_y = {0: y_c}
            tt_last = 0  # Node number for the second stage
            d = 1  # Counter for B&B
            while d <= D:
                # select the minimum valued node
                tt_bar = DictMin(T2_v)

                # Save the non-int indices
                non_int = YInt(T2_y[tt_bar])

                # Select one randomly for now
                elem_to_split = random.choice(non_int)

                # New splits are added if are feasible or bounded
                all_inf_unb = 0
                for split_sense in ('upper', 'lower'):
                    tt_last += 1
                    temp_node = copy.copy(T2[tt_bar])  # Copy the node selected for splitting. It is a version of sub-problem
                    temp_node.AddSplit(elem_to_split, T2_y[tt_bar][elem_to_split], split_sense)  # Add the split to the new node
                    temp_output = temp_node.Solve()  # Solve the new node to check its feasibility or boundedness

                    if temp_output != 'InfUnb':
                        # Model feasible, so added to the active nodes
                        T2[tt_last], T2_y[tt_last], T2_v[tt_last] = temp_node, temp_output[0], temp_output[1]
                    else:
                        all_inf_unb += 1

                # If both splits are infeasible or unbounded, do not remove the parent node
                if all_inf_unb != 2:
                    del T2[tt_bar]
                    del T2_v[tt_bar]
                    del T2_y[tt_bar]

                # next iteration
                d += 1
            # End of (while d <= D)

            # Now T2 has at most 2^(D-1) nodes that we need to generate the cuts from them.
            # Generate cuts, add to the model.
            for tt in T2.keys():
                # Define CGLP model, variables, and constraints
                cglp = gp.Model()

                # T, W, and r have been updated with added lower and upper bounds.
                # After one CGLP, new cuts are added that do not belong to upper-lower bounds. So, be careful.

                # [Lambda 2, mu2, nu2] for T2[tt] node. Let's call it lmn2 standing for Lambda, Mu, and Nu
                # Then calculate W_lmn2
                W_row_count = SparseRowCount(T2[tt].W)
                row_keys = range(1, W_row_count + 1)
                l2 = cglp.addVars(row_keys)
                # We need all y indices. no matter if they are integer or continuous.
                PI2 = {}
                for j_y in T2[tt].Y_keys:
                    # Y_indices must be a vector starting from 1. That means all (t, g) must turn into one vector.
                    # Example: (1-168, 1) and (1-168, 2) ==> [1, 2, ..., 168, 168+1, 168+2, ..., 168+168]. Crazy, I know
                    # i_r and j_y means i relates to rows, and j relates to y indices.
                    PI2[j_y] = 0
                    for i_r in row_keys:
                        if (i_r, j_y) in T2[tt].W.keys():
                            PI2[j_y] += T2[tt].W[(i_r, j_y)] * l2[i_r]

                PI0 = 0
                for i_r in row_keys:
                    # i_r and j_y means i relates to rows, and j relates to y indices.
                    if i_r in T2[tt].r.keys():
                        PI0 += T2[tt].r[i_r] * l2[i_r]  # still continuous


                PI1 = {}
                for j_x in T2[tt].X_keys:
                    PI1[j_x] = 0
                    for i_r in row_keys:
                        if (i_r, j_x) in T2[tt].T.keys():
                            PI1[j_x] += T2[tt].T[(i_r, j_x)] * l2[i_r]


            # solve again and get solution y, continue. Note that the sub-problem has various vars but only y is required.

            # End of (while True)
            print('Hi')


if __name__ == '__main__':
    Probs = {1: 0.5, 2: 0.5}

    # Open master problem and save it in the root node.
    # There are various dictionaries saving the master tree data.
    m_model, m_A, m_b, m_ub, m_lb = OpenMasterFile()
    root_class = MasterProb(m_model, m_A, m_b, m_ub, m_lb)
    X, v = root_class.Solve()
    T1, T1_v, T1_x = {0: root_class}, {0: v}, {0: X}
    T1_A, T1_b, T1_ub, T1_lb = {0: m_A}, {0: m_b}, {0: m_ub}, {0: m_lb}

    # SP is the list of all sub-problem instances at node t (the key of dictionary).
    # Not that for each node t we have different sub-problem instances.
    SP = {0: {}}
    for sp in Probs.keys():
        s_model, s_T, s_W, s_r, s_ub, s_lb = OpenSubFile(sp)
        sp_instance = SubProb(s_model, s_T, s_W, s_r, s_ub, s_lb)
        SP[0][sp] = sp_instance

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

    #Gz = [{0: []} for _ in SP]
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
            liid = random.choice(non_int)  # it is the form of [l, (ii, d)]
            split_value = T1_x[t_bar][liid[0]][liid[1]]
            for sense in ('upper', 'lower'):
                # First get the model from the parent class. Second, create a new class with the model. Third, solve it.
                t_last += 1
                temp_model, temp_A, temp_b, temp_ub, temp_lb = T1[t_bar].ReturnModel()
                temp_node = MasterProb(temp_model, temp_A, temp_b, temp_ub, temp_lb)
                temp_node.AddSplit(liid, split_value, sense)
                temp_output = temp_node.Solve()

                if temp_output != 'InfUnb':
                    # Model feasible, so added to the active nodes.
                    # Copy Gx and Gz for the new nodes
                    T1[t_last], T1_x[t_last], T1_v[t_last] = temp_node, temp_output[0], temp_output[1]
                    Gx[t_last] = copy.copy(Gx[t_bar])

                    # Generate new sub-problems for the new generated nodes
                    SP[t_last] = {}
                    for sp in SP[t_bar].keys():
                        temp_model, temp_T, temp_W, temp_r, temp_ub, temp_lb = SP[t_bar][sp].ReturnModel()
                        SP[t_last][sp] = SubProb(temp_model, temp_T, temp_W, temp_r, temp_ub, temp_lb)

            # Remove splitted node from the active nodes
            # Remove subproblems from the list of active nodes
            del T1[t_bar]
            del T1_v[t_bar]
            del T1_x[t_bar]
            del SP[t_bar]

            # end of while number 1
            print(T1_x)
            print(SP)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Benders' Cut Generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # We now have t[k] that determines which x need to be used for the next step.
        # The model at node t[k] is used and then subproblems are convexified.
        # Here we need to generate Benders' cuts


        # Increase iteration
        k += 1
        terminate += 1





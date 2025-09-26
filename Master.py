import pickle
import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import quicksum, GRB
import warnings
from tqdm import tqdm
import gc  # garbage collecter
import psutil  # to monitor memory
from time import time
from Methods import *
from Data import ModelsData

class Master:
    def __init__(self, data: ModelsData, master_env, scale):
        ############################## First Stage Model
        l_index = data.l_index
        d_index = data.d_index
        master = gp.Model('MasterProb', env=master_env)
        self.X_I = master.addMVar((data.L, data.D), vtype=GRB.INTEGER, name='X')
        self.U_I = master.addMVar(data.L, vtype=GRB.BINARY, name='U')
        self.eta = master.addVar(lb=-1e10, vtype=GRB.CONTINUOUS, name='eta')

        ##### Investment costs and constraint
        self.capital_cost = (data.F * self.U_I).sum() + sum(self.X_I[:, d].sum() * data.C[d] for d in d_index)
        self.operation_cost = sum(self.X_I[:, d].sum() * data.O[d] for d in d_index)

        master.addConstr(self.capital_cost <= data.Budget_I, name='budget')
        for l in l_index:
            for d in d_index:
                master.addConstr((self.X_I[l, d] <= scale * data.device_ub[l, d] * self.U_I[l]), name=f'ub[{l},{d}]')
        master.addConstr(self.X_I[:, 2].sum() <= scale * data.device_ub[0, 2], name='dg_limit')

        master.addConstr(self.X_I[:, 1].sum() <= 4 * self.X_I[:, 0].sum(), name='pv_if_es')

        master.addConstr(self.X_I[:, 2].sum() <= self.X_I[:, 0].sum(), name='dg_if_es')

        master.addConstr(self.X_I[:, 2].sum() <= self.X_I[:, 1].sum(), name='dg_if_pv')

        #### Objective
        self.total_cost = self.capital_cost + data.N * data.to_year * self.operation_cost + \
                     self.eta
        master.setObjective(self.total_cost, sense=GRB.MINIMIZE)
        ##### Save master in mps + save data in pickle
        master.update()
        master.write('Models/Master.mps')
        self.model = master


def save_solutions(master:Master):
    costs_to_save = {
        'capital': master.capital_cost,
        'operation': master.operation_cost,
        'eta': master.eta
    }
    # Save to a pickle file
    with open(f'Results/Master-mg {mg_id}.pkl', 'wb') as file:
        pickle.dump(vars_to_save, costs_to_save, file)
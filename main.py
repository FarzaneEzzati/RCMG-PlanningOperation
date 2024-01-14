# Dec 14, 2023. Repository was created to keep track of codings for DOE project.

import pickle
import time
import random
import copy
import numpy as np
from static_functions import *
import gurobipy as gp
from gurobipy import quicksum, GRB
env = gp.Env()
env.setParam('OutputFlag', 1)

if __name__ == '__main__':
    
    real_model = gp.read('Models/real25.mps', env=env)
    all_vars = real_model.getVars()
    real_model.optimize()
    all_vars_data = []
    for var in all_vars:
        all_vars_data.append(var.x)
    with open('Data/Sol.pkl', 'wb') as handle:
        pickle.dump(all_vars_data, handle)
    handle.close()




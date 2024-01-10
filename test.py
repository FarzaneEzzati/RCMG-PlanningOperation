# Dec 14, 2023. Repository was created to keep track of codings for DOE project.

import copy
import pickle
import random
import math
import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import GRB


with open('Data/OutageScenarios.pkl', 'rb') as handle:
        scens, probs = pickle.load(handle)
handle.close()
print(probs, scens)

model = gp.Model()
x = model.addVars((1, 2, 3), lb=0, ub=4)
model.addConstr(x[1] + 2*x[2] - x[3] <= 6)
model.setObjective(x[1]+x[2]+2*x[3], sense=GRB.MAXIMIZE)
model.update()
print(model.getA().todok())
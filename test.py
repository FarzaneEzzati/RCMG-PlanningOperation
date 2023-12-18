import pickle
import random
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import quicksum, GRB


m = gp.Model()
x = [m.addVars([(1, 2), (3, 4)]), m.addVars([(3, 6), (8, 9)])]
print(x[1][(3,6)])

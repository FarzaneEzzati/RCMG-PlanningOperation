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


with open('Data/Sol.pkl', 'rb') as handle:
        v = pickle.load(handle)
handle.close()
print(v[:17])

# Dec 14, 2023. Repository was created to keep track of codings for DOE project.

import copy
import pickle
import random
import math
import numpy as np
import pandas as pd
import time


if __name__ == '__main__':
    from BuildModels import SingleScenario, RealScale
    model = RealScale()
    tt = time.time()
    print(model.Solve())
    print(f'Run Time: {time.time()-tt}')

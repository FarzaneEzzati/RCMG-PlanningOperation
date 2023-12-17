import pickle
import numpy as np

scens = [0, 4, 10, 15, 26, 37, 47, 55, 71, 90]
probs = [scens[s]/np.sum(scens) for s in range(len(scens))][::-1]

scens_dic = {i + 1: scens[i] for i in range(len(scens))}
probs_dic = {i + 1: probs[i] for i in range(len(probs))}

with open('Data/OutageScenarios.pkl', 'wb') as handle:
     pickle.dump([scens_dic, probs_dic], handle)
handle.close()
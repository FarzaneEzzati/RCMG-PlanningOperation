import pickle
import gurobipy as gp
from gurobipy import Model
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from typing import Dict

env_master = gp.Env()
env_sub = gp.Env()
env_sub.setParam('OutputFlag', 0)
env_sub.setParam('DualReductions', 0)
env_sub.setParam('InfUnbdInfo', 1)


'''class BendersCut:
    def __init__(self, models, probs, TMatrices, rVectors, x_keys):
        self.models = models
        self.probs = probs
        self.TMatrices = TMatrices
        self.rVectors = rVectors
        self.x_keys = x_keys

    def __call__(self, model, where):
        if where == gp.GRB.Callback.MIPSOL:
            X = model.cbGetSolution(model._vars)
            # Get solutions in subproblems, calculate e and E
            E, e = GetPIs(optimal_x=X, models=self.models, probs=self.probs,
                          TMatrices=self.TMatrices, rVectors=self.rVectors,
                          x_keys=self.x_keys)
            # Add a cut based on the solution # For example, adding a simple cut:
            model.cbLazy(model._vars[-1] + sum(model._vars[x] * E_x for E_x in E) >= e)
'''

def LoadSubProblems():
    # Load scenario probabilities
    with open('Data/Probabilities.pkl', 'rb') as handle:
        probs = pickle.load(handle)

    # Open subproblems
    models, TMatrices, rVectors = [], [], []
    print('Loading Subproblems')
    with tqdm(range(len(probs)), desc='Loading') as tbar:
        for s in range(len(probs)):
            models.append(gp.read(f'Models/Sub{s}.mps', env=env_sub))
            with open(f'Models/Sub{s}-Tr.pkl', 'rb') as handle:
                TMatrix, rVector, x_keys = pickle.load(handle)
            TMatrices.append(TMatrix)
            rVectors.append(rVector)
            tbar.update(1)
    TMatrices = np.array(TMatrices)
    rVectors = np.array(rVectors)

    return models, probs, TMatrices, rVectors, x_keys



def GetPIs(optimal_x, models, probs, TMatrices, rVectors, x_keys):
    duals = []  # Dictionary of dual multipliers
    s_range = range(len(probs))
    n_xs = len(x_keys)
    #### Optimize first
    for s, f in enumerate(models):
        vars = f.getVars()
        for x in x_keys:
            vars[x].UB = optimal_x[x]
            vars[x].LB = optimal_x[x]
        f.update()
        f.optimize()
        if f.status == 2:
            duals.append(np.array([c.Pi for c in f.getConstrs()]))
    duals = np.array(duals)

    # Calculate two values e and E for Benders
    duals_r = np.array([np.dot(duals[s], rVectors[s]) for s in s_range])   # [S]
    e = np.average(duals_r, weights=probs)
    E = []
    for x_key in x_keys:
        Temp1 = np.array([np.dot(TMatrices[s][:, x_key], duals[s]) for s in s_range]) # [S, C]
        E.append(np.average(Temp1, weights=probs))
    return E, e



def BendersCut(model, where, models, probs, TMatrices, rVectors, x_keys):
    if where == gp.GRB.Callback.MIPSOL:
        X = model.cbGetSolution(model._vars)
        # Get solutions in subproblems, calculate e and E
        E, e = GetPIs(optimal_x=X, models=models, probs=probs,
                          TMatrices=TMatrices, rVectors=rVectors,
                          x_keys=x_keys)
        # Add a cut based on the solution # For example, adding a simple cut:
        model.cbLazy(model._vars[-1] + sum(model._vars[x] * E_x for E_x in E) >= e)


def BD_BandB(models, probs, TMatrices, rVectors, x_keys):
    ##### Solve master by callback
    master = gp.read('Models/Master.mps', env=env_master)
    master._vars = master.getVars()
    master.Params.LazyConstraints = 1
    master.Params.LogFile = "Models/master_log.log"
    master.Params.DegenMoves = 0
    master.optimize(lambda model, where: BendersCut(model, where, models, probs, TMatrices, rVectors, x_keys))

    ##### Get optimal x
    X_values = [x.x for x in master.getVars()] # Save optimal solution of master problem
    return X_values


if __name__ == '__main__':


    # Reporting
    X_values = [x.x for x in master.getVars()] # Save optimal solution of master problem
    total_cost = master.ObjVal # the objective value of master problem
    X1, counter = {}, 0  # start changing solution format from a list to dictionary to find ild
    for ld in X_ld:
        X1[(ld[0], ld[1])] = X_values[counter]
        counter += 1

    #  Solve subproblems for optimal x found
    X2 = {ld: 0 for ld in X_ld}
    for scen in SP.keys():
        vars_optimal = SP[scen].getVars()
        for x in Xkeys:
            vars_optimal[x].UB = X_values[x]
            vars_optimal[x].LB = X_values[x]
        SP[scen].update()
        SP[scen].optimize()
        for ld in X_ld:
            X2[ld] += Probs[scen] * SP[scen].getVarByName(f'X2[{ld[0]},{ld[1]}]').x
    pd.DataFrame(X1, index=[0]).to_csv('X1.csv') # master problem solution save
    pd.DataFrame(X2, index=[0]).to_csv('X2.csv') # subproblem solution save


    Save_Ys = True
    if Save_Ys:
        l, m, h = None, None, None
        for os in Outage_scens.keys():
            if Outage_scens[os] == 10:
                if l is None:
                    l = os
            elif Outage_scens[os] == 42:
                if m is None:
                    m = os
            elif Outage_scens[os] == 106:
                if h is None:
                    h = os

        # Save for scenario low
        l_ESL = {itg: SP[l].getVarByName(f'Y_ESL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_PVES = {itg: SP[l].getVarByName(f'Y_PVES[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_DGES = {itg: SP[l].getVarByName(f'Y_DGES[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_DGL = {itg: SP[l].getVarByName(f'Y_DGL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_PVL = {itg: SP[l].getVarByName(f'Y_PVL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_PVCur = {itg: SP[l].getVarByName(f'Y_PVCur[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_PVGrid = {itg: SP[l].getVarByName(f'Y_PVGrid[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_LL = {itg: SP[l].getVarByName(f'Y_LL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_LT = {ittg: SP[l].getVarByName(f'Y_LT[{ittg[0]},{ittg[1]},{ittg[2]},{ittg[3]}]').x for ittg in Y_ittg}
        l_E = {itg: SP[l].getVarByName(f'Y_E[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_GL = {itg: SP[l].getVarByName(f'Y_GridL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        l_GES = {itg: SP[l].getVarByName(f'Y_GridES[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        with open(f'Visualizations/{com_folder[com]} Low_Outage.pkl', 'wb') as handle:
            pickle.dump([l_ESL, l_PVL, l_PVES, l_DGES, l_DGL, l_LT, l_LL, l_E, l_GL, l_GES, l_PVCur, l_PVGrid], handle)
        handle.close()

        m_ESL = {itg: SP[m].getVarByName(f'Y_ESL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_PVES = {itg: SP[m].getVarByName(f'Y_PVES[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_DGES = {itg: SP[m].getVarByName(f'Y_DGES[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_DGL = {itg: SP[m].getVarByName(f'Y_DGL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_PVL = {itg: SP[m].getVarByName(f'Y_PVL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_PVCur = {itg: SP[m].getVarByName(f'Y_PVCur[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_PVGrid = {itg: SP[m].getVarByName(f'Y_PVGrid[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_LL = {itg: SP[m].getVarByName(f'Y_LL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_LT = {ittg: SP[m].getVarByName(f'Y_LT[{ittg[0]},{ittg[1]},{ittg[2]},{ittg[3]}]').x for ittg in Y_ittg}
        m_E = {itg: SP[m].getVarByName(f'Y_E[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_GL = {itg: SP[m].getVarByName(f'Y_GridL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        m_GES = {itg: SP[m].getVarByName(f'Y_GridES[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        with open(f'Visualizations/{com_folder[com]} Medium_Outage.pkl', 'wb') as handle:
            pickle.dump([m_ESL, m_PVL, m_PVES, m_DGES, m_DGL, m_LT, m_LL, m_E, m_GL, m_GES, m_PVCur, m_PVGrid], handle)
        handle.close()

        h_ESL = {itg: SP[h].getVarByName(f'Y_ESL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_PVES = {itg: SP[h].getVarByName(f'Y_PVES[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_DGES = {itg: SP[h].getVarByName(f'Y_DGES[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_DGL = {itg: SP[h].getVarByName(f'Y_DGL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_PVL = {itg: SP[h].getVarByName(f'Y_PVL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_PVCur = {itg: SP[h].getVarByName(f'Y_PVCur[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_PVGrid = {itg: SP[h].getVarByName(f'Y_PVGrid[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_LL = {itg: SP[h].getVarByName(f'Y_LL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_E = {itg: SP[h].getVarByName(f'Y_E[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_LT = {ittg: SP[h].getVarByName(f'Y_LT[{ittg[0]},{ittg[1]},{ittg[2]},{ittg[3]}]').x for ittg in Y_ittg}
        h_GL = {itg: SP[h].getVarByName(f'Y_GridL[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        h_GES = {itg: SP[h].getVarByName(f'Y_GridES[{itg[0]},{itg[1]},{itg[2]}]').x for itg in Y_itg}
        with open(f'Visualizations/{com_folder[com]} High_Outage.pkl', 'wb') as handle:
            pickle.dump([h_ESL, h_PVL, h_PVES, h_DGES, h_DGL, h_LT, h_LL, h_E, h_GL, h_GES, h_PVCur, h_PVGrid], handle)
        handle.close()


    Report = True
    if Report:
        print('Reporting started')
        #  Resilience Metrics
        EndurList = []
        SusList = [] # times that load was completely lost
        LSOList, LOList = [], []  # Load Served in Outage List, Load in Outage List
        LSnTList, LnTList = [], []  # Load Served when no Transfer List
        LSnOList, LnOList = [], []  # Load Served when no Outage List
        LLOList = []  # Load Lost when Outage
        LTList = []  # Load Transfered
        ImportList = []
        for scen in SP.keys():
            if Outage_scens[scen] >= 168 - 16:
                outage_hours = range(16, 169)
            else:
                outage_hours = range(16, 16 + Outage_scens[scen] + 1)

            AllLoadFailsRate, SustainRate = [], []
            AllLoadTrans = 0
            for i in RNGSta:
                for g in RNGMonth:
                    Fail = copy.copy(Outage_scens[scen])
                    for oh in outage_hours[:len(outage_hours)-2]:
                        yll1 = SP[scen].getVarByName(f'Y_LL[{i},{oh},{g}]').x >=\
                               0.75 * ((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][oh-1]
                        yll2 = SP[scen].getVarByName(f'Y_LL[{i},{oh+1},{g}]').x >= \
                              0.75 * ((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][oh]

                        if (yll1, yll2) == (True, True):
                            Fail = oh - 16
                            break
                    AllLoadFailsRate.append(Fail/Outage_scens[scen])

                    for t in RNGTime[:-1]:
                        AllLoadTrans += sum(SP[scen].getVarByName(f'Y_LT[{i},{t},{tt},{g}]').x
                                            for tt in range(t, 169))
                    sustain = 0
                    for oh in outage_hours:
                        if SP[scen].getVarByName(f'Y_LL[{i},{oh},{g}]').x != 0:
                            sustain += 1
                    SustainRate.append(sustain/Outage_scens[scen])
            EndurList.append(np.mean(AllLoadFailsRate))
            SusList.append(np.mean(SustainRate))
            LTList.append(AllLoadTrans)


            LOList.append(sum(((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][t - 1]
                                      for i in RNGSta for t in outage_hours for g in RNGMonth))

            LLOList.append(sum(SP[scen].getVarByName(f'Y_LL[{i},{t},{g}]').x
                                      for i in RNGSta for t in outage_hours for g in RNGMonth))

            LSOList.append(sum(SP[scen].getVarByName(f'Y_ESL[{i},{t},{g}]').x +
                              SP[scen].getVarByName(f'Y_DGL[{i},{t},{g}]').x +
                              SP[scen].getVarByName(f'Y_PVL[{i},{t},{g}]').x
                                        for i in RNGSta for t in outage_hours for g in RNGMonth))

            LSnTList.append(sum(SP[scen].getVarByName(f'Y_ESL[{i},{t},{g}]').x +
                                         SP[scen].getVarByName(f'Y_DGL[{i},{t},{g}]').x +
                                         SP[scen].getVarByName(f'Y_PVL[{i},{t},{g}]').x
                                         for i in RNGSta for g in (1, 2, 3, 6, 7, 8, 9) for t in outage_hours if
                                         t in DontTrans[g]))

            LnTList.append(sum(((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][t - 1]
                                        for i in RNGSta for g in (1, 2, 3, 6, 7, 8, 9) for t in outage_hours if
                                        t in DontTrans[g]))

            LSnOList.append(sum(SP[scen].getVarByName(f'Y_ESL[{i},{t},{g}]').x +
                                          SP[scen].getVarByName(f'Y_DGL[{i},{t},{g}]').x +
                                          SP[scen].getVarByName(f'Y_PVL[{i},{t},{g}]').x
                                          for i in RNGSta for t in RNGTime if t not in outage_hours for g in RNGMonth))
            LnOList.append(sum(((1 + (i - 1) * AG_scens[scen]) ** ReInvsYear) * Load_scens[scen][g][t - 1]
                                         for i in RNGSta for t in RNGTime if t not in outage_hours for g in RNGMonth))

        Endurance = sum([Probs[s] * EndurList[s] for s in SP.keys()])
        Sustain = sum([Probs[s] * SusList[s] for s in SP.keys()])
        LoadAssur = eta_i * sum([Probs[s] * LSOList[s] / LOList[s] for s in SP.keys()])
        PeakAssur = eta_i * sum([Probs[s] * LSnTList[s] / LnTList[s] for s in SP.keys()])

        Bill1 = sum([Probs[s] * LnOList[s] * GridPlus for s in SP.keys()])
        Bill2 = sum([Probs[s] * (LSnOList[s] * LoadPrice + (LnOList[s] - LSnOList[s]) * GridPlus)
                     for s in SP.keys()])
        GridExport = sum(Probs[scen] * sum(SP[scen].getVarByName(f'Y_ESGrid[{itg[0]},{itg[1]},{itg[2]}]').x +
                                        SP[scen].getVarByName(f'Y_PVGrid[{itg[0]},{itg[1]},{itg[2]}]').x +
                                        SP[scen].getVarByName(f'Y_DGGrid[{itg[0]},{itg[1]},{itg[2]}]').x
                                        for itg in Y_itg) for scen in SP.keys())
        GridImport = sum([Probs[scen] * (LnOList[scen] - LSnOList[scen]) for scen in SP.keys()])
        GridImportPerc = sum(Probs[scen] * (LnOList[scen] - LSnOList[scen])/LnOList[scen] for scen in SP.keys())
        #  Save reports
        report = {'Investment': sum(C[ld[1]] * X1[ld] for ld in X_ld),
                  'Reinvestment': sum(C[ld[1]] * X2[ld] for ld in X_ld),
                  'Avg Recourse': sum(Probs[scen] * SP[scen].ObjVal for scen in SP.keys()),
                  'Load Lost%': sum(Probs[scen] * LLOList[scen]/LOList[scen] for scen in SP.keys()),
                  'Load Served%': sum(Probs[scen] * LSOList[scen]/LOList[scen] for scen in SP.keys()),
                  'Load Transferred%': sum(Probs[scen] * LTList[scen] for scen in SP.keys()),
                  'Grid Load%': GridImportPerc,
                  'Grid Exported': GridExport,
                  'Grid Imported': GridImport,
                  'Bill Before': Bill1,
                  'Bill After': Bill2,
                  'Impact Endurance': Endurance,
                  'Sustained Access': Sustain,
                  'Load Assurance': LoadAssur,
                  'Peak Assurance': PeakAssur,
                  'ES1': sum(X1[ld] for ld in X_ld if ld[1] == 1),
                  'PV1': sum(X1[ld] for ld in X_ld if ld[1] == 2),
                  'DG1': sum(X1[ld] for ld in X_ld if ld[1] == 3),
                  'ES2': sum(X2[ld] for ld in X_ld if ld[1] == 1),
                  'PV2': sum(X2[ld] for ld in X_ld if ld[1] == 2),
                  'DG2': sum(X2[ld] for ld in X_ld if ld[1] == 3)
                  }
        pd.DataFrame(report, index=[0]).to_csv('Report.csv')


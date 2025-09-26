import pickle
import gurobipy as gp
from gurobipy import Model
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from typing import Dict
from itertools import product


def GetPIs(optimal_x, seps, probs, TMatrices, rVectors):
    duals_e = []
    duals_T = []
    #### Optimize first

    for f in seps:
        vars = f.model.getVars()[:len(optimal_x)]
        f.model.setAttr('LB', vars, optimal_x)
        f.model.setAttr('UB', vars, optimal_x)
        f.model.optimize()

        if f.model.status == 2:
            dual_e = np.array([c.Pi for c in f.model.getConstrs()])
            dual_T = np.array([c.Pi for c in f.linking_constrs])
            duals_e.append(dual_e)
            duals_T.append(dual_T)
        else:
            f.model.computeIIS()
            for c in f.model.getConstrs():
                if c.IISConstr:
                    print(c.ConstrName)
            raise ValueError(f'{f.model.ModelName} status {f.model.status}')

    # duals in rhs
    e = np.fromiter((np.dot(d, r) for r, d in zip(rVectors, duals_e)), dtype=float)
    # E (vector of size x)
    E = [T.T.dot(d) for T, d in zip(TMatrices, duals_T)]
    return E, e



def Cuts(model, where, seps, probs, TMatrices, rVectors, data):
    # Benders cut
    vars = None
    if where == gp.GRB.Callback.MIPNODE and model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL:
        vars = model.cbGetNodeRel(model._vars)
    if where == gp.GRB.Callback.MIPSOL and vars is None:
        vars = model.cbGetSolution(model._vars)
    if vars is not None:
        split_point = len(data.x_keys)
        optimal_x, etas = np.array(vars[:split_point]), vars[split_point:]
        # Get solutions in subproblems, calculate e and E
        E, e = GetPIs(
            optimal_x=optimal_x,
            seps=seps,
            probs=probs,
            TMatrices=TMatrices,
            rVectors=rVectors)
        # Avg E and e
        e = np.average(e, weights=probs)
        E = probs[:, None] * E
        E = np.sum(E, axis=0)
        # Add a cut
        cut_expr = gp.LinExpr()
        for x, coeff in enumerate(E):  # E = π^T T, so coeffs on x_j
            cut_expr += coeff * model._vars[x]
        model.cbLazy(model._vars[-1] >= e - cut_expr  )  # e = π^T r
        model.update()

        """# Generate heuristic solution
        lambda_rnd = np.random.uniform(0, 0.99)
        optimal_x = lambda_rnd * optimal_x + (1-lambda_rnd) * max_x
        optimal_x[-data.L:] = np.ceil(optimal_x[-data.L:])
        # Get solutions in subproblems, calculate e and E
        E, e = GetPIs(
            optimal_x=optimal_x,
            seps=seps,
            probs=probs,
            TMatrices=TMatrices,
            rVectors=rVectors)
        # Avg E and e
        e = np.average(e, weights=probs)
        E = probs[:, None] * E
        E = np.sum(E, axis=0)
        # Add a cut
        cut_expr = gp.LinExpr()
        for x, coeff in enumerate(E):  # E = π^T T, so coeffs on x_j
            cut_expr += coeff * model._vars[x]
        model.cbLazy(model._vars[-1] >= e - cut_expr )  # e = π^T r
        model.update()"""


def solve_with_BD_BandB(master, seps, TMatrices, rVectors, probs, data):
    master.model._vars = master.model.getVars()
    master.model.Params.LazyConstraints = 1
    master.model.Params.NumericFocus = 3
    master.model.Params.LogFile = "Models/master_log.log"
    master.model.optimize(lambda model, where: Cuts(model, where, seps, probs, TMatrices, rVectors, data))

    ##### Get optimal x
    optimal_x = np.array([x.X for x in master.model._vars[:-1]]) # Save optimal solution of master problem
    optimal_obj = master.model.ObjVal
    return optimal_x, optimal_obj


def get_T_r(sep, data):
    A = sep.model.getA()
    x_len = len(data.x_keys)
    linking_constrs = sep.linking_constrs

    # Directly get row indices of linking constraints
    linking_rows = [con.index for con in linking_constrs]

    T = A[linking_rows, :x_len]
    r = np.array([con.RHS for con in sep.model.getConstrs()])
    return T, r


def get_bill_saving(sep, scenario, data, e_load, e_drp):
    outage_time = range(data.outage_start, data.outage_start + int(data.OT[0][scenario[0]])+1)
    not_outage_time = [t for t in range(168) if t not in outage_time]
    load_from_grid = np.sum([sep.Load.x[:, :, t].sum() for t in not_outage_time])
    grid_bill = data.e_grid_import * load_from_grid

    load_from_MG = np.sum(sep.Y_ESL.x[:, :, t].sum() +
                          sep.Y_PVL.x[:, :, t].sum() +
                          sep.Y_DGL.x[:, :, t].sum()
                          for t in not_outage_time )
    load_from_grid = sep.Y_GridL.x.sum()
    incentives = sep.Y_LT.x.sum()
    MG_bill = e_load * load_from_MG + \
              load_from_grid * data.e_grid_import - \
              e_drp * incentives
    bill_saving = (grid_bill - MG_bill)/grid_bill
    return bill_saving


def get_resilience(sep, scenario, data):
    outage_duration = int(data.OT[0][scenario[0]])
    start = data.outage_start
    end = data.outage_start + outage_duration

    # Phi
    did_shed = sep.Y_LSh.x[:, :, start:end] >= 0.5 * sep.Load.x[:, :, start:end]
    time_to_fail = np.argmax(did_shed, axis=2)
    all_true_mask = np.all(~did_shed, axis=2)
    time_to_fail[all_true_mask] = did_shed.shape[2]

    Phi = time_to_fail / outage_duration
    Phi = np.average(Phi)

    # Lambda
    shed_amount = sep.Y_LSh.x[:, :, start:end].sum(axis=2)
    load = sep.Load.x[:, :, start:end].sum(axis=2)
    Lambda = 1 - shed_amount / load
    Lambda = np.average(Lambda)

    # E
    shedding_hours = did_shed.sum(axis=2)
    E = 1 - np.average(shedding_hours)/outage_duration
    return Phi, Lambda, E


def get_peak_shift(sep, scenario, data):
    peak_shift = 0
    for g in range(data.G):
        DRP =sep.Y_LT.x[:, g, :].sum()
        Load = sum([sep.Load.x[:, g, t].sum() for t in data.no_trans[g]])
        peak_shift += DRP/Load
    peak_shift /= data.G
    return peak_shift
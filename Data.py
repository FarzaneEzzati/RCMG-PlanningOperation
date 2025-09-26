import pickle
import numpy as np
import pandas as pd
import time
import gurobipy as gp
from itertools import product


def read_data(mg_id):
    ############################## Import Data
    location_df = pd.read_csv(f'Data/location_{mg_id}.csv')
    general_df = pd.read_csv(f'Data/data_{mg_id}.csv')
    general_dict = {c: general_df[c].iloc[0] for c in general_df.columns}
    location_dict = {l: dict(zip(location_df.columns[1:], location_df.iloc[l].values[1:])) for l in
                     location_df['location']}

    ############################## Power Outage Scenarios
    OT = pd.read_csv('Scenarios/Outage/Outage_Scenarios_reduced.csv', usecols=['Gamma_Scenario', 'Normalized'])
    OT_prob = OT['Normalized']
    OT_scen = np.tile(OT['Gamma_Scenario'], (12, 1))  # [12, OT_count]
    OT_count = OT_scen.shape[1]
    ############################## PV
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    PV_scen = []
    for m in months:
        df = pd.read_csv(f'Scenarios/PV/{mg_id}/PVscenario-{m}.csv')
        week_scen = np.tile(df[df.columns[2:]].iloc[0], (1, 7))/4
        PV_scen.append(week_scen)
    PV_scen = np.array(PV_scen).squeeze(1)
    ############################## Load scenarios
    joint_months = ['JanFeb', 'MarApr', 'MayJun', 'JulAug', 'SepOct', 'NovDec']
    LD_scen = []
    for m in joint_months:
        df_w = pd.read_csv(f'Scenarios/Load Demand/{mg_id}/LoadScenarios-{m}-w.csv')
        df_e = pd.read_csv(f'Scenarios/Load Demand/{mg_id}/LoadScenarios-{m}-e.csv')
        weekday_scen = np.tile(df_w[df_w.columns[1:]], (1, 5))
        weekend_scen = np.tile(df_e[df_e.columns[1:]], (1, 2))
        weeklong_scen = np.concatenate((weekday_scen, weekend_scen), axis=1)
        num_scens = weeklong_scen.shape[0]
        weeklong_scen = weeklong_scen.reshape((num_scens, 168, 4)).sum(axis=2)
        LD_scen.append(weeklong_scen)
        LD_scen.append(weeklong_scen)
    LD_prob = np.array(df_w['probs'])
    LD_scen = np.transpose(np.array(LD_scen), (0, 2, 1))  # [12, 168, LD_count]
    LD_count = LD_scen.shape[-1]
    ############################## Load Growth Scenarios
    LG = pd.read_csv('Scenarios/Load Demand/annual growth scenarios.csv')
    LG_scen = LG['Lognorm Scenario'] / 100
    LG_prob = LG['Probability']  # [LG_count]
    LG_count = LG_scen.shape[0]
    ############################## Combination Scenarios
    combination_scen = np.array(list(product(range(OT_count), range(LD_count), range(LG_count))))
    combination_prob = np.array([OT_prob[s[0]] * LD_prob[s[1]] * LG_prob[s[2]]
                                 for s in combination_scen])
    ############################## Max of Outage
    max_outage_index = np.argmax(OT_scen)
    max_outage_scenario_index = next(i for i, s in enumerate(combination_scen) if s[0] == max_outage_index)
    with open('Data/Probabilities.pkl', 'wb') as handle:
        pickle.dump([combination_scen, combination_prob, max_outage_scenario_index], handle)

    return general_dict, location_dict, \
        OT_scen, LD_scen, LG_scen, PV_scen, \
        combination_scen, combination_prob


class ModelsData:
    def __init__(self, mg_id):
        ############################## Read Data
        self.general_dict, self.location_dict, \
            self.OT, self.LD, self.LG, self.PV, \
            self.Sc, self.Pr = read_data(mg_id)
        ############################## Set Parameters
        self.I = 2
        self.T = 168  # count of hours
        self.L = len(self.location_dict.keys())  # count of locations
        self.D = 3  # count of devices
        self.G = 4  # count of months
        self.S = len(self.Pr)
        self.x_keys = range(self.L * self.D + self.L)

        # Peak Hours
        summer_peak = [23 * i + 12 + j for i in range(6) for j in range(8)]  # 13-19PM
        winter_peak = np.concatenate(([23 * i + 5 + j for i in range(6) for j in range(5)],
                                      [23 * i + 17 + j for i in range(6) for j in range(5)]))  # 6-10AM, 6-10PM
        fall_spring_peak = [23 * i + 16 + j for i in range(6) for j in range(4)]
        self.no_trans = {0: winter_peak, 1: fall_spring_peak, 2: summer_peak, 3: fall_spring_peak}
        self.trans_period = 6
        ###### General Data
        self.Budget_I = self.general_dict['Budget_I']
        self.Budget_E = self.general_dict['Budget_E']
        self.C = np.array([self.general_dict[key] for key in ['C_es', 'C_pv', 'C_dg']])
        self.e_grid_import = self.general_dict['e_grid_import']
        self.e_grid_export = self.general_dict['e_grid_export']
        self.wtp = self.general_dict['wtp']
        self.sv = self.general_dict['sv']
        self.subsidy_rate = self.general_dict['subsidy_rate']
        self.sv_subsidy_rate = ((0.5 + self.sv) / 1.5) * self.subsidy_rate
        ###### Device Efficiencies
        self.es_soc_ub, self.es_soc_lb = 0.9, 0.1
        self.es_eta = 0.9
        self.es_effi = 0.90
        self.dg_effi = 0.95
        self.dg_fuel_usage = self.general_dict['dg_efficiency(gal/kw)']
        self.fuel_price = self.general_dict['fuel($/gal)']
        self.dg_fuel_cost = self.dg_fuel_usage * self.fuel_price  # Fuel cost of DG: $/kWh
        self.degrad_rate = 0.02
        ###### General Parameter
        self.outage_start = 15
        self.to_year = (365 / 7) / self.G
        self.N = 20  # Life Time
        self.n = 10  # Expansion year
        self.labor_rate = 0.1
        self.operation_rate = np.array([0.05, 0.04, 0.1])
        self.C = self.C * (1 + self.labor_rate)  # unit cost with labor
        self.C = self.C * (1 - self.sv_subsidy_rate) # subsidied
        self.C[1] = self.C[1] * (1 - 0.2)  # 20% Federal Solar Investment Tax Credit (ITC)
        self.O = self.operation_rate * self.C
        self.F = np.array([value['cost']
                           for _, value in self.location_dict.items()])  # location cost
        self.device_ub = np.array([[value['es_max'], value['pv_max'], value['dg_max']]
                                   for _, value in self.location_dict.items()])  # [n_locations, n_devices]
        ###### Electricity Prices
        self.e_load_rate = 0.9
        self.e_load = self.e_load_rate * self.e_grid_import
        shedding_priority = np.zeros(self.T)
        shedding_priority[self.outage_start:] = np.linspace(1, 0.25, self.T - self.outage_start)
        self.e_sheding =  5 * (2 - self.wtp) * self.e_grid_import * shedding_priority
        self.e_cur_pv = self.e_grid_export # + self.O[1]/(52*24)
        self.e_cur_dg = self.e_grid_export + self.dg_fuel_cost
        ###### Demand Response
        self.drp = 0.2
        self.e_drp = (1 + self.drp) * self.e_grid_import
        ############################## Indices
        self.g_index = range(self.G)
        self.l_index = range(self.L)
        self.i_index = range(self.I)
        self.t_index = range(self.T)
        self.d_index = range(self.D)



import pandas as pd

selected_scenarios = 2


def getCommunityData(community):
    D1 = pd.read_csv('Data/data-all.csv')
    D1 = D1[D1['community'] == community]
    mg_dict = {}
    mg_dict['sv'] = D1['sv'].iloc[0]
    mg_dict['wtp'] = D1['wtp'].iloc[0]
    mg_dict['grid buy'] = D1['grid buy'].iloc[0]
    mg_dict['grid buy back'] = D1['grid buy back'].iloc[0]
    mg_dict['B0'] = D1['budget 0'].iloc[0]
    mg_dict['B1'] = D1['budget 1'].iloc[0]

    D2 = pd.read_csv(f'Data/data-{community}.csv')
    mg_dict['d max'] = {0: D2['es max'].values, 1: D2['pv max'].values, 2: D2['dg max'].values}
    mg_dict['F'] = D2['location price'].values

    # PV scenarios
    mg_dict['pv_s'] = {}
    mg_dict['pv_pr'] = dict.fromkeys(range(selected_scenarios), 1)
    months = dict(enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            start=0))
    for m in months.keys():
        dpv = pd.read_csv(f'Scenarios/PV/{community}/PVscenario-{months[m]}.csv')
        cols = dpv.columns[2:27]
        mg_dict['pv_s'][m] = {}
        for s in range(selected_scenarios):
            mg_dict['pv_pr'][s] *= dpv['probs'].iloc[s]
            week_long = pd.concat([dpv[cols].iloc[s] for _ in range(7)])/4
            mg_dict['pv_s'][m][s] = week_long
    prob_sum = sum(mg_dict['pv_pr'].values())
    mg_dict['pv_pr'] = {s: mg_dict['pv_pr'][s]/prob_sum for s in mg_dict['pv_pr'].keys()}

    # demand scenarios
    mg_dict['load_s'] = {}
    mg_dict['load_pr'] = dict.fromkeys(range(selected_scenarios), 1)
    for m in months.keys():
        if m % 2 == 0:
            joint_months = months[m]+months[m+1]
            weekdays = pd.read_csv(f'Scenarios/Load Demand/{community}/LoadScenarios-{joint_months}-w.csv')
            weekends = pd.read_csv(f'Scenarios/Load Demand/{community}/LoadScenarios-{joint_months}-e.csv')
            cols = weekdays.columns[2:99]
            mg_dict['load_s'][m], mg_dict['load_s'][m + 1] = {}, {}
            for s in range(selected_scenarios):
                mg_dict['load_pr'][s] *= weekdays['probs'].iloc[s] * weekends['probs'].iloc[s]
                dw_single = weekdays[cols].iloc[s]
                de_single = weekends[cols].iloc[s]
                week_long = pd.concat([dw_single] * 5 + [de_single] * 2)
                week_long_hourly = []
                j = 0
                while j <= 668:
                    week_long_hourly.append(sum(week_long[j + k] for k in range(4)))
                    j += 4
                mg_dict['load_s'][m][s] = week_long_hourly
                mg_dict['load_s'][m + 1][s] = week_long_hourly
    prob_sum = sum(mg_dict['load_pr'].values())
    mg_dict['load_pr'] = {s: mg_dict['load_pr'][s]/prob_sum for s in mg_dict['load_pr'].keys()}
    return mg_dict


def getOutageDemandData():
    ldo_dict = {}
    AG = pd.read_csv('Scenarios/Load Demand/annual growth scenarios.csv')
    ldo_dict['ldg_pr'] = {s: AG['Probability'].iloc[s] for s in range(selected_scenarios)}
    ldo_dict['ldg_s'] = {s: AG['Lognorm Scenario'].iloc[s] / 100 for s in range(selected_scenarios)}

    OT = pd.read_csv('Scenarios/Outage/Outage Scenarios - reduced.csv')
    ldo_dict['po_pr'] = {s: OT['Probability'].iloc[s] for s in range(selected_scenarios)}
    ldo_dict['po_s'] = {s: int(OT['Gamma Scenario'].iloc[s]) for s in range(selected_scenarios)}
    return ldo_dict
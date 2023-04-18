# Libraries
from pulp import *
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys
import os
import numpy as np


#////////////////////////////////////////////////////////////////////
#---------------- Battery Operation Model Definition-----------------
#////////////////////////////////////////////////////////////////////

class OptimalBatteryOperation(object):

    # Initialise the object
    def __init__(self):
        # Path for input data
        self.data_path = "Data\\"

    # Read input parameters
    def read_parameters(self):
        sh = "Data"
        try:
            df_para = pd.read_excel(self.data_path + "battery_parameters.xlsx", sheet_name=sh)
        except:
            print('Error: Read battery parameters - Check input file or close it')
            sys.exit()
        return df_para

    # Read market data
    def read_market_data(self):
        sh1 = "Half-hourly data"
        sh2 = "Daily data"
        try:
            df_half_hourly = pd.read_excel(self.data_path + "market_data.xlsx", sheet_name=sh1)
            df_daily = pd.read_excel(self.data_path + "market_data.xlsx", sheet_name=sh2)
        except:
            print('Error: Read market data - Check input file or close it')
            sys.exit()
        return df_half_hourly, df_daily

    # Process daily data
    def process_daily_data(self, df_daily):
        # Create a copy of the dataframe to avoid modifying original data
        df_daily_t = df_daily.copy()
        # Create a temporal dataframe to produce half hour granularity
        half_period_day = 48
        df_day_period = pd.DataFrame(index=np.arange(half_period_day), columns=np.arange(1))
        df_day_period.columns = ["Market 3 Price [£/MWh]"]
        # Create an empty dataframe to concatenate df_day_period
        df_half_hourly_m3 = pd.DataFrame()
        # Extend original daily data to a half hour granularity
        # All prices are the same during the day
        for t in df_daily_t.index:
            day_price = df_daily_t.loc[t]["Market 3 Price [£/MWh]"]
            df_day_period["Market 3 Price [£/MWh]"] = day_price
            df_half_hourly_m3 = pd.concat([df_half_hourly_m3, df_day_period], axis=0)
        # Reset index
        df_half_hourly_m3.reset_index(inplace=True, drop=True)

        return df_half_hourly_m3

    # Merge a single market data dataframe
    def merge_full_market_data(self, df_half_hourly, df_half_hourly_m3, start_year, end_year):
        # Create a copy of the dataframe to avoid modifying original data
        df_half_hourly_t = df_half_hourly.copy()
        df_half_hourly_m3_t = df_half_hourly_m3.copy()
        # Create full dataset
        df_full_market_data = pd.concat([df_half_hourly_t, df_half_hourly_m3_t], axis=1)
        # Rename date
        df_full_market_data.rename({df_full_market_data.columns[0]:"Date"}, inplace=True, axis=1)

        # Filter selected years
        start_date = f"{start_year}-01-01 00:00:00"
        end_date = f"{end_year}-12-31 23:30:00"
        start_date_format = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_date_format = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        t1 = df_full_market_data[df_full_market_data["Date"] == start_date_format]["Date"].index[0]
        t2 = df_full_market_data[df_full_market_data["Date"] == end_date_format]["Date"].index[0]
        # Filtered dataframe
        df_full_market_data = df_full_market_data.loc[t1:t2].copy()

        return df_full_market_data

    # Calculate the objective function for selected period
    def calculate_obj_function(self, v_objective, df_battery_operation, df_para):
        # Create a copy to avoid modifying data
        df_battery_operation_t = df_battery_operation.copy()
        df_para_t = df_para.copy()

        # Fixed operation costs per period simulated
        fixed_operation = df_para_t[df_para_t[df_para_t.columns[0]] == "Fixed Operational Costs"]["Values"].values[0]

        # CAPEX per period simulated
        capex = df_para_t[df_para_t[df_para_t.columns[0]] == "Capex"]["Values"].values[0] # per lifetime
        life_time = df_para_t[df_para_t[df_para_t.columns[0]] == "Lifetime (1)"]["Values"].values[0] # in years
        capex_per_period = capex/(life_time)
        # Total revenue
        obj_function_per_period = v_objective - fixed_operation - capex_per_period

        # Create dataframe
        data = {'Total_Revenue': [obj_function_per_period], 'From': [df_battery_operation_t.index[0]], "To": [df_battery_operation_t.index[-1]]}
        df_objetive_function = pd.DataFrame(data=data)

        return df_objetive_function


    # Optimisation model
    def model(self, df_para, df_full_market_data, df_initial_conditions, df_initial_conditions_m3):

        #////////////////////////////////////////////////////////////////////
        #------------------------ Data definition --------------------------
        #////////////////////////////////////////////////////////////////////

        # Make a copy of the market data dataframe
        df_full_market_data_t = df_full_market_data.copy()

        # Define time
        time1 = df_full_market_data_t.shape[0] # half hour granularity
        time2 = int((time1+1)/48) # daily granularity

        # Define prices
        # Assuming that the curtailed electricity is traded in an intraday market (max of the Market 1 and Market 2)
        s_price_m1 = df_full_market_data_t["Market 1 Price [£/MWh]"]
        price_m1 = list(s_price_m1)
        s_price_m2 = df_full_market_data_t["Market 2 Price [£/MWh]"]
        price_m2 = list(s_price_m2)
        s_price_m3 = df_full_market_data_t["Market 3 Price [£/MWh]"]
        price_m3 = list(s_price_m3)
        s_price_curt = np.maximum(s_price_m1, s_price_m2)
        price_curt = list(s_price_curt)

        # Curtailment calculation
        total_demand = df_full_market_data_t["Transmission System Electricity Demand [MW]"]
        gen_w = df_full_market_data_t["Wind Generation [MW]"]
        gen_s = df_full_market_data_t["Solar Generation [MW]"]
        gen_c = df_full_market_data_t["Coal Generation [MW]"]
        gen_g = df_full_market_data_t["Gas Generation [MW]"]
        total_gen  = gen_w + gen_s + gen_c + gen_g
        curtailment = total_demand - total_gen
        curtailment[curtailment>=0] = 0
        curtailment = abs(curtailment)
        curtailment = list(curtailment)

        #////////////////////////////////////////////////////////////////////
        #-------------------- Model Definition and Solver-------------------
        #////////////////////////////////////////////////////////////////////
        cwd = os.getcwd()
        solverdir = 'Solver\\bin\\cbc.exe'
        solverdir = os.path.join(cwd, solverdir)
        solver = COIN_CMD(path=solverdir)
        opt_model = LpProblem(name="Optimal_Battery_Operation")

        #////////////////////////////////////////////////////////////////////
        #----------------------Decision Variables----------------------------
        #////////////////////////////////////////////////////////////////////

        # Battery State of Charge - Energy
        max_storage = df_para[df_para[df_para.columns[0]] == "Max storage volume"]["Values"].values[0]
        SOE = [LpVariable(cat='Continuous', lowBound=0, upBound = max_storage, name=f"SOE_{t}") for t in range(time1)]
        # Battery Charging - Energy
        max_capacity = df_para[df_para[df_para.columns[0]] == "Max charging rate"]["Values"].values[0]
        Charge_m1 = [LpVariable(cat='Continuous', lowBound=0, upBound=max_capacity/2, name=f"Charge_m1_{t}") for t in range(time1)]
        Charge_m2 = [LpVariable(cat='Continuous', lowBound=0, upBound=max_capacity/2, name=f"Charge_m2_{t}") for t in range(time1)]
        Charge_m3 = [LpVariable(cat='Continuous', lowBound=0, upBound=max_capacity/2, name=f"Charge_m3_{t}") for t in range(time1)]
        Charge_curt = [LpVariable(cat='Continuous', lowBound=0, upBound=max_capacity/2, name=f"Charge_curt_{t}") for t in range(time1)]

        # Battery discharging - Energy
        min_capacity = df_para[df_para[df_para.columns[0]] == "Max discharging rate"]["Values"].values[0]
        Discharge_m1 = [LpVariable(cat='Continuous', lowBound=-min_capacity/2, upBound=0, name=f"Discharge_m1_{t}") for t in range(time1)]
        Discharge_m2 = [LpVariable(cat='Continuous', lowBound=-min_capacity/2, upBound=0, name=f"Discharge_m2_{t}") for t in range(time1)]
        Discharge_m3 = [LpVariable(cat='Continuous', lowBound=-min_capacity/2, upBound=0, name=f"Discharge_m3_{t}") for t in range(time1)]
        Discharge_curt = [LpVariable(cat='Continuous', lowBound=-min_capacity/2, upBound=0, name=f"Discharge_curt_{t}") for t in range(time1)]


        #////////////////////////////////////////////////////////////////////
        #----------------------Auxiliary Variables----------------------------
        #////////////////////////////////////////////////////////////////////

        # Net charge from all markets
        charging_eff = df_para[df_para[df_para.columns[0]] == "Battery charging efficiency"]["Values"].values[0]
        Net_Charge = [LpVariable(cat='Continuous', lowBound=0, upBound=max_capacity * (1 - charging_eff) / 2,name=f"Net_Charge{t}") for t in range(time1)]

        # Net discharge form all markets
        discharging_eff = df_para[df_para[df_para.columns[0]] == "Battery discharging efficiency"]["Values"].values[0]
        Net_Discharge = [LpVariable(cat='Continuous', lowBound=-min_capacity / [2 * (1 - discharging_eff)], upBound=0, name=f"Net_Discharge{t}") for t in range(time1)]

        # Binary variables defining the battery operation
        Bin_charge = [LpVariable(cat='Binary', name=f"Bin_charge_{t}") for t in range(time1)]
        Bin_discharge = [LpVariable(cat='Binary', name=f"Bin_discharge_{t}") for t in range(time1)]

        # Binary variables defining the trading on market 3
        Bin_charge_m3 = [LpVariable(cat='Binary', name=f"Bin_charge_m3_{t}") for t in range(time1)]
        Bin_discharge_m3 = [LpVariable(cat='Binary', name=f"Bin_discharge_m3_{t}") for t in range(time1)]

        # Binary variables to buy or sell on the Market 3
        Bin_m3_buy = [LpVariable(cat='Binary', name=f"Bin_m3_buy_{t2}") for t2 in range(time2)]
        Bin_m3_sell = [LpVariable(cat='Binary', name=f"Bin_m3_sell_{t2}") for t2 in range(time2)]

        #////////////////////////////////////////////////////////////////////
        #--------------------------Setting initial conditions----------------
        #////////////////////////////////////////////////////////////////////

        opt_model += SOE[0] == df_initial_conditions['SOE']

        opt_model += Charge_m1[0] == df_initial_conditions['Charge_m1']
        opt_model += Charge_m2[0] == df_initial_conditions['Charge_m2']
        opt_model += Charge_m3[0] == df_initial_conditions['Charge_m3']
        opt_model += Charge_curt[0] == df_initial_conditions['Charge_curt']

        opt_model += Discharge_m1[0] == df_initial_conditions['Discharge_m1']
        opt_model += Discharge_m2[0] == df_initial_conditions['Discharge_m2']
        opt_model += Discharge_m3[0] == df_initial_conditions['Discharge_m3']
        opt_model += Discharge_curt[0] == df_initial_conditions['Discharge_curt']

        opt_model += Net_Charge[0] == df_initial_conditions['Net_Charge']
        opt_model += Net_Discharge[0] == df_initial_conditions['Net_Discharge']


        opt_model += Bin_charge[0] == df_initial_conditions['Bin_Charge']
        opt_model += Bin_discharge[0] == df_initial_conditions['Bin_Discharge']

        opt_model += Bin_charge_m3[0] == df_initial_conditions['Bin_Charge_m3']
        opt_model += Bin_discharge_m3[0] == df_initial_conditions['Bin_Discharge_m3']

        opt_model += Bin_m3_buy[0] == df_initial_conditions_m3['Bin_m3_buy']
        opt_model += Bin_m3_sell[0] == df_initial_conditions_m3['Bin_m3_sell']

        #////////////////////////////////////////////////////////////////////
        #--------------------------Constraints---------------------------
        #////////////////////////////////////////////////////////////////////

        charging_eff = df_para[df_para[df_para.columns[0]] == "Battery charging efficiency"]["Values"].values[0]
        for t in range(1, time1):
            opt_model += Net_Charge[t] == (Charge_m1[t] + Charge_m2[t] + Charge_m3[t] + Charge_curt[t])*(1-charging_eff)
            opt_model += Net_Charge[t] <= max_storage - SOE[t - 1]

        for t in range(1, time1):
            opt_model += Charge_m1[t] <= (max_capacity/2)
            opt_model += Charge_m2[t] <= (max_capacity/2)
            opt_model += Charge_m3[t] <= (max_capacity/2)
            opt_model += Charge_curt[t] <= (max_capacity/2)

        for t in range(1, time1):
            opt_model += Net_Charge[t] <= (max_capacity*(1-charging_eff)/2) * Bin_charge[t]
            opt_model += -Net_Discharge[t] <= (min_capacity/(2*(1-discharging_eff))) * Bin_discharge[t]
            opt_model += Bin_charge[t] + Bin_discharge[t] <= 1

        for t in range(1, time1):
            opt_model += Charge_m3[t] == (max_capacity/2) * Bin_charge_m3[t]
            opt_model += -Discharge_m3[t] == (min_capacity / 2) * Bin_discharge_m3[t]

        discharging_eff = df_para[df_para[df_para.columns[0]] == "Battery discharging efficiency"]["Values"].values[0]
        for t in range(1, time1):
            opt_model += Net_Discharge[t]*(1-discharging_eff) == (Discharge_m1[t] + Discharge_m2[t] + Discharge_m3[t] + Discharge_curt[t])
            opt_model += -Net_Discharge[t] <= SOE[t - 1]

        for t in range(1, time1):
            opt_model += -Discharge_m1[t] <= (min_capacity / 2)
            opt_model += -Discharge_m2[t] <= (min_capacity / 2)
            opt_model += -Discharge_m3[t] <= (min_capacity / 2)
            opt_model += -Discharge_curt[t] <= (min_capacity / 2)

        for t in range(1, time1):
            opt_model += Charge_curt[t] >= 0
            opt_model += Discharge_curt[t] <= 0
            opt_model += Charge_curt[t] <= curtailment[t]
            opt_model += Discharge_curt[t] >= -curtailment[t]

        for t in range(1, time1):
            opt_model += SOE[t] == SOE[t-1] + Net_Charge[t] + Net_Discharge[t]
            opt_model += SOE[t] <= max_storage

        for t2 in range(1, time2):
            p = t2*48
            opt_model += -lpSum(Discharge_m3[p + t] for t in range(0, 48)) <= SOE[t - 1]
            opt_model += lpSum(Charge_m3[p + t] for t in range(0, 48)) <= max_storage - SOE[t - 1]
            opt_model += lpSum(Bin_discharge_m3[p + t] for t in range(0,48)) == 48 * Bin_m3_sell[t2]
            opt_model += lpSum(Bin_charge_m3[p + t] for t in range(0,48)) == 48 * Bin_m3_buy[t2]

        #////////////////////////////////////////////////////////////////////
        #------------------------Objective function-------------------------
        #////////////////////////////////////////////////////////////////////

        # The model optimises the battery's operation
        # Capex and fixed operation costs are included out of the model

        # Revenue
        revenue_m1 = -lpSum(price_m1[t] * Discharge_m1[t] * 0.5 for t in range(time1))
        revenue_m2 = -lpSum(price_m2[t] * Discharge_m2[t] * 0.5 for t in range(time1))
        revenue_m3 = -lpSum(price_m3[t] * Discharge_m3[t] * 0.5 for t in range(time1))
        revenue_curt = -lpSum(price_curt[t] * Discharge_curt[t] * 0.5 for t in range(time1))
        total_revenue = revenue_m1 + revenue_m2 + revenue_m3 + revenue_curt

        # Cost
        price_curtailment = 0
        cost_m1 = lpSum(price_m1[t] * Charge_m1[t] * 0.5 for t in range(time1))
        cost_m2 = lpSum(price_m2[t] * Charge_m2[t] * 0.5 for t in range(time1))
        cost_m3 = lpSum(price_m3[t] * Charge_m3[t] * 0.5 for t in range(time1))
        cost_curt = lpSum(price_curtailment * Charge_curt[t] * 0.5 for t in range(time1))
        total_cost = cost_m1 + cost_m2 + cost_m3 + cost_curt

        # Total objective
        objective = total_revenue -  total_cost

        # Solve the optimisation problem
        try:
            opt_model.sense = LpMaximize
            opt_model.setObjective(objective)
            opt_model.solve(solver)
        except:
            print('Error: Model optimization - Check model')
            sys.exit()

        # Processing results
        res_half_hourly = []
        for t in range(time1):
            record = {'period': t,
                      'Charge_m1': Charge_m1[t].varValue,
                      'Charge_m2': Charge_m2[t].varValue,
                      'Charge_m3': Charge_m3[t].varValue,
                      'Charge_curt': Charge_curt[t].varValue,
                      'Net_Charge': Net_Charge[t].varValue,
                      'Discharge_m1': Discharge_m1[t].varValue,
                      'Discharge_m2': Discharge_m2[t].varValue,
                      'Discharge_m3': Discharge_m3[t].varValue,
                      'Discharge_curt': Discharge_curt[t].varValue,
                      'Net_Discharge': Net_Discharge[t].varValue,
                      'Bin_Charge': Bin_charge[t].varValue,
                      'Bin_Discharge': Bin_discharge[t].varValue,
                      'Bin_Charge_m3': Bin_charge_m3[t].varValue,
                      'Bin_Discharge_m3': Bin_discharge_m3[t].varValue,
                      'SOE': SOE[t].varValue}
            res_half_hourly.append(record)

        df_results_half_hourly = pd.DataFrame.from_records(res_half_hourly)
        df_results_half_hourly.set_index('period', inplace=True)

        res_daily = []
        for t2 in range(time2):
            record = {'period': t2,
                      'Bin_m3_buy': Bin_m3_buy[t2].varValue,
                      'Bin_m3_sell': Bin_m3_sell[t2].varValue}
            res_daily.append(record)

        df_results_daily = pd.DataFrame.from_records(res_daily)
        df_results_daily.set_index('period', inplace=True)
        
        objetive_function_value = opt_model.objective.value()

        return df_results_half_hourly, df_results_daily, objetive_function_value

if __name__ == "__main__":

    start_time = datetime.now()
    print(" -----------   Battery Operation Optimisation   -------------")
    print(" -----------       Nicolas Achury Beltran        ------------")
    print(" --------------    Aurora Energy Research     ---------------")
    print(f"Starting time        : {start_time}")
    print("-------------------------------------------")
    print(f"General parameters:")

    start_year = 2018
    end_year   = 2018

    print(f"Start year                   : {start_year}")
    print(f"End year                     : {end_year}")

    print("-------------------------------------------")
    print(f"Defining initial conditions...")

    initial_conditions_half_hourly = {'period': 0,
                                     'Charge_m1': 0,
                                     'Charge_m2': 0,
                                     'Charge_m3': 0,
                                     'Charge_curt': 0,
                                     'Net_Charge': 0,
                                     'Discharge_m1': 0,
                                     'Discharge_m2': -1,
                                     'Discharge_m3': 0,
                                     'Discharge_curt': 0,
                                     'Net_Discharge': -1.05,
                                     'Bin_Charge': 0,
                                     'Bin_Discharge': 0,
                                     'Bin_Charge_m3': 0,
                                     'Bin_Discharge_m3': 0,
                                     'SOE': 4}

    initial_conditions_daily = {'period':0,
                                'Bin_m3_buy':1,
                                'Bin_m3_sell':1}

    df_initial_conditions_0 = pd.DataFrame.from_records([initial_conditions_half_hourly])
    df_initial_conditions_0.set_index('period', inplace=True)
    df_initial_conditions_0_m3 = pd.DataFrame.from_records([initial_conditions_daily])
    df_initial_conditions_0_m3.set_index('period', inplace=True)

    print("-------------------------------------------")
    print(f"Defining Opt problem...")
    obj_optimal_battery = OptimalBatteryOperation()
    print("-------------------------------------------")
    print(f"Reading and processing input data...")
    df_para = obj_optimal_battery.read_parameters()
    df_half_hourly, df_daily = obj_optimal_battery.read_market_data()
    df_half_hourly_m3 = obj_optimal_battery.process_daily_data(df_daily)
    df_full_market_data = obj_optimal_battery.merge_full_market_data(df_half_hourly, df_half_hourly_m3, start_year, end_year)

    print("-------------------------------------------")
    print(f"Producing Rolling Horizons...")

    years = list(range(start_year, end_year+1, 1))
    yearly_results = dict()
    yearly_objective = dict()

    for i, year in enumerate(years):
        if i == 0:
            print("-------------------------------------------")
            print(f"Running year {year}")
            print(f"Optimising...")

            start_date = f"{year}-01-01 00:00:00"
            end_date = f"{year}-12-31 23:30:00"
            start_date_format = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            end_date_format = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            t1 = df_full_market_data[df_full_market_data["Date"] == start_date_format]["Date"].index[0]
            t2 = df_full_market_data[df_full_market_data["Date"] == end_date_format]["Date"].index[0]
            df_full_market_data_t_ini = df_full_market_data.loc[t1:t2].copy()
            # Run the model
            df_battery_operation_half_hourly_ini, df_battery_operation_daily_ini, objetive_function_value = obj_optimal_battery.model(df_para,
                                                                                                                                     df_full_market_data_t_ini,
                                                                                                                                     df_initial_conditions_0,
                                                                                                                                     df_initial_conditions_0_m3)
            # Get the latest state of decision and auxiliary variables
            initial_conditions_n_1 = df_battery_operation_half_hourly_ini.iloc[-1]
            initial_conditions_n_1_m3 = df_battery_operation_daily_ini.iloc[-1]
            # Process results
            df_battery_operation_half_hourly_ini["Date"] = df_full_market_data_t_ini["Date"]
            df_battery_operation_half_hourly_ini.set_index("Date", drop=True, inplace=True)
            # Save results
            yearly_results[year] = df_battery_operation_half_hourly_ini
            yearly_objective[year] = objetive_function_value
        else:
            print("-------------------------------------------")
            print(f"Running year {year}")
            print(f"Optimising...")

            start_date = f"{year}-01-01 00:00:00"
            end_date = f"{year}-12-31 23:30:00"
            start_date_format = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            end_date_format = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            t1 = df_full_market_data[df_full_market_data["Date"] == start_date_format]["Date"].index[0]
            t2 = df_full_market_data[df_full_market_data["Date"] == end_date_format]["Date"].index[0]
            df_full_market_data_t = df_full_market_data.loc[t1:t2].copy()
            # Run the model
            df_battery_operation_half_hourly, df_battery_operation_daily, objetive_function_value = obj_optimal_battery.model(df_para,
                                                                                                                             df_full_market_data_t,
                                                                                                                             initial_conditions_n_1,
                                                                                                                             initial_conditions_n_1_m3)
            # Get the latest state of decision and auxiliary variables
            initial_conditions_n_1 = df_battery_operation_half_hourly.iloc[-1]
            initial_conditions_n_1_m3 = df_battery_operation_daily.iloc[-1]
            # Process results
            df_full_market_data_t.reset_index(inplace=True, drop=True)
            df_battery_operation_half_hourly["Date"] = df_full_market_data_t["Date"]
            df_battery_operation_half_hourly.set_index("Date", drop=True, inplace=True)
            # Save results
            yearly_results[year] = df_battery_operation_half_hourly
            yearly_objective[year] = objetive_function_value


    print("-------------------------------------------")
    print(f"Exporting...")
    # Define names
    name_file_output = "Output_" + start_date[0:start_date.find(" ")]+"__"+end_date[0:end_date.find(" ")]+".csv"
    name_file_input = "Input_" + start_date[0:start_date.find(" ")] + "__" + end_date[0:end_date.find(" ")] + ".csv"

    current_dateTime = datetime.now()
    folder_now = current_dateTime.strftime("%d-%m-%Y__%H.%M.%S")

    path_base = os.path.dirname(os.path.realpath(__file__))
    path_results = path_base + "\\Results\\" + folder_now
    folder_results = os.path.isdir(path_results)
    # Create results folder
    if folder_results == False:
        os.makedirs(path_results)
    # Export dataframe
    df_battery_operation = yearly_results[start_year]
    df_full_market_data.set_index("Date", inplace=True)
    df_full_market_data.to_csv(path_results + f"\\{name_file_input}")
    df_battery_operation.to_csv(path_results + f"\\{name_file_output}")
    # Export objetive function
    v_objective = yearly_objective[start_year]
    df_objetive_function = obj_optimal_battery.calculate_obj_function(v_objective, df_battery_operation, df_para)
    name_file_obj = "Total_Revenue_" + start_date[0:start_date.find(" ")] + "__" + end_date[0:end_date.find(" ")] + ".csv"
    df_objetive_function.to_csv(path_results + f"\\{name_file_obj}", index=False)


    print("-------------------------------------------")
    print(f"Success...")
    print("-------------------------------------------")
    end_time = datetime.now()
    print(f"Time finished           :{end_time}")
    print("Duration                :{}".format(end_time-start_time))
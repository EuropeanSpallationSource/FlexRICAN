from toolbox.units import *
from toolbox.tools import *

def scenario(df = None, weight = 1, lp:bool = True, pv_co2 = 36):


    temp = df.copy()

    for col in temp.columns:
        if "MW" in col:
            print(f"new column: {f"{col.split("[")[0]}[kW]"}")

            temp[f"{col.split("[")[0]}[kW]"] = temp[col] * 1000

    time = TimeUnit(start = "2023-01-01 00:00" , periods = len(temp), dt = 1)

    i = 0
    var_ids = []
    elec_node = EnergyNode(time=time, name = "elec_node", energy_type=elec)
    model = OptimisationModel(name='flex_potential_1', time=time)
    # for ind, frame in wrk_df.groupby(wrk_df.block_id):

    temp["available"] = [0] * len(temp)

    for ind, frame in  temp.groupby(pd.Grouper(freq="W-TUE")):
        temp.loc[frame.index[0], "available"] = 1


    for ind, frame in  temp.groupby(pd.Grouper(freq="W-TUE")):
        # print (frame)
        
        var_ids.append(i)


        value = frame.loc[:,"Electricity_Consumption[kW]"].tolist()
        co2_out = frame.loc[:,"Taux de Co2"]
        
        
        name = f"consumption_block_{i}"

    
        if sum(value) == 0:
            value = [0.001 for v in value]
         
  
        exec(f"{name}= ShiftableConsumptionUnit(time = time, name = '{name}', power_values = value,  verbose=True, binary = False)")



        exec(f"elec_node.connect_units({name}  )")
        # exec(f"heat_node.connect_units({name}.thermal_production_unit )")

        i += 1 

 


    pv = PhotovoltaicUnit(time=time, name = "pv", profile = temp.pv.tolist(),  energy_type=elec, co2_cost_per_kw = 800000, co2_out=pv_co2)

    grid_imp = VariableProductionUnit(time=time , name = "grid_imp" , energy_type=elec, co2_out= temp["Taux de Co2"].tolist(), binary=True)
    grid_exp = VariableConsumptionUnit(time=time , name = "grid_exp" , energy_type=elec, co2_out= (0.5 * temp["Taux de Co2"]).tolist() , binary = True)


    
    elec_node.connect_units(grid_imp, grid_exp,pv)
    


    available = temp.available.tolist()
    exp = " + ".join([f"consumption_block_{j}_start_up[t]" for j in var_ids]) + f" <= {available}[t]"
    cst = DefinitionDynamicConstraint(name="defined_start_up", t_range='for t in time.I', exp_t=exp, parent=None)
    setattr(elec_node, 'defined_start_up', cst)

    # if "co2" in objectives:
    grid_imp.minimize_co2_emissions(weight  = weight, pareto = False)
    pv.minimize_co2_emissions(weight  = weight, pareto = False)
    
    
    # buffer.charge.minimize_co2_emissions(weight = weight)
    # buffer.minimize_capacity(weight=weight)
    # model.add_nodes(elec_node,heat_node)


    
    model.add_nodes(elec_node,  verbose = False)
    
    # model.addConstraint(name="singleUnit" , constraint="")
    print ("READY TO SOLVE")
    if lp:
        model.writeLP("test_copy.lp")
    


    t0 = tm.time()
    model.solve_and_update(solver = GUROBI_CMD())
    solve_time = tm.time() - t0

    stats = get_lp_stats(model, solve_time=solve_time)


    temp = pd.DataFrame()

    for j in range (i):

        cmd = f"temp['consumption_block_{j}'] = consumption_block_{j}.p.get_value()"

        exec (cmd)


    df["PV_Cap_S2_KWp"] =pv.nb_unit.get_value()

    df["grid_import_S2_kWh"] = grid_imp.p.get_value()
    df["grid_export_S2_kWh"] = grid_exp.p.get_value()
    df["OPEX_S2_€"] = df[["grid_import_S2_kWh", "electricity_price[€/kWh]"]].product(axis = 1) - df[["grid_export_S2_kWh", "elec_export_price[€/kWh]"]].product(axis = 1)

    df["estimated_CO2_S2_elec"] = grid_imp.co2_emissions.get_value() 
    df["estimated_CO2_S2_pv"] = pv.co2_emissions.get_value()
    df['estimated_CO2_district_heating'] = df[["Heat_demand[MW]", "cciag_co2"]].product(axis = 1) * 1000
    df["estimated_CO2_S2_total"] = df[["estimated_CO2_S2_elec","estimated_CO2_S2_pv","estimated_CO2_district_heating" ]].sum(axis =1).tolist()

    df["Consumption_S2"] = temp.sum(axis =1).tolist()

    df["PV_production_S2"] = df.pv.copy() * round(pv.nb_unit.get_value(),2)


    return df, stats


if __name__ == "__main__":
    d_sets = [{"country" : "France", "PV_CO2" : 36},{"country" : "Czech" , "PV_CO2" : 38.3}]

    for d_set in d_sets:
        file = f"LNCMI/flexibility_study/data/{d_set["country"]}_results.csv"
        try:
            
            df = pd.read_csv(file, sep=",", decimal=".", index_col="datetime", parse_dates=True)
        except:
            file = f"./data/{d_set["country"]}_results.csv"
            df = pd.read_csv(file, sep=",", decimal=".", index_col="datetime", parse_dates=True)
        print (f"Data loaded from {file} , Starting Optimization Scenario 2 for {d_set["country"]}")
                
        df, stats = scenario(df, weight = 1,pv_co2 = d_set["PV_CO2"])

        with open(f'data/stats/Scenario_2_{d_set["country"]}.txt', 'w') as f:
            f.write(str(stats))

        df.to_csv(file, sep=",", decimal=".", index=True, index_label="datetime")

        print (f"Optimization Scenario 2 completed, results saved to {file}")
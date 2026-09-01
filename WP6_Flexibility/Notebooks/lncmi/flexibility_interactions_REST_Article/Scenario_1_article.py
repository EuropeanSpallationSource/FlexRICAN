
from toolbox.units import *
from toolbox.tools import *

def scenario(df , weight, pv_co2):

    temp = df.copy()

    for col in temp.columns:
        if "MW" in col:
            print(f"new column: {f"{col.split("[")[0]}[kW]"}")

            temp[f"{col.split("[")[0]}[kW]"] = temp[col] * 1000

    time = TimeUnit(start = "2023-01-01 00:00" , periods = len(temp), dt = 1)

    pv = PhotovoltaicUnit(time=time, name = "pv", profile = temp.pv.tolist(),  energy_type=elec, co2_cost_per_kw = 800000, co2_out=pv_co2)
    
    conso = FixedConsumptionUnit(time = time, name = "conso", p = temp["Electricity_Consumption[kW]"].tolist(), energy_type=elec)
   
    grid_imp = VariableProductionUnit(time=time , name = "grid_imp" , energy_type=elec, co2_out= temp["Taux de Co2"].tolist()   )
    grid_exp = VariableConsumptionUnit(time=time , name = "grid_exp" , energy_type=elec, co2_out= (0.5*temp["Taux de Co2"]).tolist()  )
    
    
    pv.minimize_co2_emissions(weight = weight)
    grid_imp.minimize_co2_emissions(weight  = weight, pareto = False)


    elec_node = EnergyNode(time=time, name = "elec_node", energy_type=elec)
  
    elec_node.connect_units(pv,conso,grid_imp, grid_exp)

    model = OptimisationModel(name='flex_potential_1', time=time)

    model.add_nodes(elec_node)

    t0 = tm.time()
    model.solve_and_update(solver = GUROBI_CMD())
    solve_time = tm.time() - t0

    stats = get_lp_stats(model, solve_time=solve_time)

    df["PV_Cap_S1_KWp"] =pv.nb_unit.get_value()   

    df["grid_import_S1_kWh"] = grid_imp.p.get_value()
    df["grid_export_S1_kWh"] = grid_exp.p.get_value()
    df["OPEX_S1_€"] = df[["grid_import_S1_kWh", "electricity_price[€/kWh]"]].product(axis = 1) - df[["grid_export_S1_kWh", "elec_export_price[€/kWh]"]].product(axis = 1)


    df["estimated_CO2_S1_elec"] = grid_imp.co2_emissions.get_value()
    df["estimated_CO2_S1_pv"] = pv.co2_emissions.get_value()
    df['estimated_CO2_district_heating'] = df[["Heat_demand[MW]", "cciag_co2"]].product(axis = 1) * 1000
    df["estimated_CO2_S1_total"] = df[["estimated_CO2_S1_elec", "estimated_CO2_S1_pv","estimated_CO2_district_heating"]].sum(axis =1).tolist()

    

    df["Consumption_S1"] = conso.p.get_value()

    df["PV_production_S1"] = df.pv.copy() * round(pv.nb_unit.get_value(),2)


    return df, stats



if __name__ == "__main__":
    d_sets = [{"country" : "France", "PV_CO2" : 36},{"country" : "Czech" , "PV_CO2" : 38.3}]

    for d_set in d_sets:
        file = f"LNCMI/flexibility_study/data/{d_set["country"]}_dataset.csv"
        try:
            
            df = pd.read_csv(file, sep=",", decimal=".", index_col="datetime", parse_dates=True)
        except:
            file = f"./data/{d_set["country"]}_dataset.csv"
            df = pd.read_csv(file, sep=",", decimal=".", index_col="datetime", parse_dates=True)
        print (f"Data loaded from {file} , Starting Optimization Scenario 1 for {d_set["country"]}")

        df = df.loc["2023-01-24" : "2023-05-08 23:50"]
       

        df.fillna(0, inplace = True)



        df, stats = scenario(df, weight = 1,pv_co2 = d_set["PV_CO2"])

        with open(f'data/stats/Scenario_1_{d_set["country"]}.txt', 'w') as f:
            f.write(str(stats))
            
        
        df.to_csv(f"{file.split('dataset')[0]}results.csv", sep=",", decimal=".", index=True, index_label="datetime")

        print (f"Optimization Scenario 1 completed, results saved to {file.split('dataset')[0]}results.csv")
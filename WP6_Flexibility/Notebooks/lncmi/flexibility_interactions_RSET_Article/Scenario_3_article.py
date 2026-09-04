
from toolbox.units import *
from toolbox.tools import *


def scenario(df = None, weight = 1, lp:bool = True, cop = 3, copper_coeeff = 0.045, elec_therm_ratio = 0.85,hp_cf = 0.2):


    temp = df.copy()

    for col in temp.columns:
        if "MW" in col:
            print(f"new column: {f"{col.split("[")[0]}[kW]"}")

            temp[f"{col.split("[")[0]}[kW]"] = temp[col] * 1000

    time = TimeUnit(start = "2023-01-01 00:00" , periods = len(temp), dt = 1)
    # limit  = temp["Heat_demand[kW]"].gt(0.0001).sum()
    temp ["valorize"] = 0 

    temp ["valorize"].loc[temp['Heat_demand[kW]']> 0.0001] = 1
    i = 0
    var_ids = []
    elec_node = EnergyNode(time=time, name = "elec_node", energy_type=elec)
    heat_node_1 = EnergyNode(time=time, name = "heat_node_1", energy_type=thermal)
    heat_node_2 = EnergyNode(time=time, name = "heat_node_2", energy_type=thermal)
    model = OptimisationModel(name='flex_potential_1', time=time)

    temp["available"] = [0] * len(temp)

    for ind, frame in  temp.groupby(pd.Grouper(freq="W-TUE")):
        temp.loc[frame.index[0], "available"] = 1


    for ind, frame in  temp.groupby(pd.Grouper(freq="W-TUE")):
        # print (frame)
        
        var_ids.append(i)


        value = frame.loc[:,"Electricity_Consumption[kW]"].tolist()
        co2_out = frame.loc[:,"Taux de Co2"]
        
        
        name = f"consumption_block_{i}"

    # ElectricalToThermalConversionUnit
        # available = df["available"].tolist()
        if sum(value) == 0:
            value = [0.001 for v in value]
            

        cmd =  name + f" = ElectricalToThermalConversionUnit(time = time, name = '" + name + \
            f"',elec_to_therm_ratio = {elec_therm_ratio}, p_in_elec = value, shiftable = True,  verbose=True, binary_elec = False,binary_therm = False)"
  
        exec(cmd)



        exec(f"elec_node.connect_units({name}.elec_consumption_unit  )")
        exec(f"heat_node_1.connect_units({name}.thermal_production_unit )")

        i += 1 

 
    # total_consumption  = VariableConsumptionUnit(time = time, name = "total_consumption", energy_type=elec)
    # elec_node.connect_units(total_consumption)

    # exp = " + ".join([f"consumption_block_{j}_elec_cons_p[t]" for j in var_ids]) + " == total_consumption_p[t]" 
    # cst = DefinitionDynamicConstraint(name="total_consumption_cst", t_range='for t in time.I', exp_t=exp, parent=None)
    # setattr(total_consumption, 'total_consumption_cst', cst)

    
 

    grid_imp = VariableProductionUnit(time=time , name = "grid_imp" , energy_type=elec, co2_out= temp["Taux de Co2"].tolist(), binary=False)
    grid_exp = VariableConsumptionUnit(time=time , name = "grid_exp" , energy_type=elec, co2_out= (0.5 * temp["Taux de Co2"]).tolist(), binary=False)

    total_consumption = Quantity(name= "total_consumption_magnets", opt= True , unit = "kW" ,
                                  vlen = time.LEN, lb = 0, parent = grid_imp)
    
    setattr(grid_imp, "total_consumption_magnets", total_consumption)
    
    exp = " + ".join([f"consumption_block_{j}_elec_cons_p[t]" for j in var_ids]) + " == grid_imp_total_consumption_magnets[t]" 
    cst = DefinitionDynamicConstraint(name="total_consumption_cst", t_range='for t in time.I', exp_t=exp, parent=grid_imp)
    setattr(grid_imp, 'total_consumption_cst', cst)



    district_heat = VariableProductionUnit(time = time, name = "district_heat" ,p_min = 0, p_max=1e9, co2_out= temp["cciag_co2"].tolist(), operating_cost=100, energy_type=thermal, e_max=None, binary = False)

    buffer = StorageUnit(time = time , name = "heat_storage",  energy_type=thermal  )

    buffer.charge._add_co2_emissions(co2_out=0)
    buffer.discharge._add_co2_emissions(co2_out=0)

    heat_dissipation = VariableConsumptionUnit(time = time, name = "heat_dissipation" , p_max =1e9, energy_type = thermal,e_max=None, binary = False )

    cnrs_heat = FixedConsumptionUnit(time = time, name = "CNRS_heat" , p = temp["Heat_demand[kW]"].tolist(), energy_type = thermal )

    heat_pump = HeatPump(time = time, name="heat_pump", cop=cop)
    heat_pump.thermal_production_unit._add_co2_emissions(co2_out= 0)


    exp = f" + heat_pump_therm_prod_p[t] <= CNRS_heat_p[t]"                                            
    cst = DefinitionDynamicConstraint(name="heat_pump_valorize", t_range='for t in time.I', exp_t=exp, parent=None)
    setattr(heat_pump, 'heat_pump_valorize', cst)
    
    # exp = f" + lpSum(heat_pump_elec_cons_u[t] for t in time.I) <= {temp.valorize[t]}"                                            
    # cst = DefinitionConstraint(name="heat_pump_activation", exp=exp, parent=None)
    # setattr(heat_pump, 'heat_pump_activation', cst)

    valo = temp.valorize.tolist()
    surplus_consumption = ElectricalToThermalConversionUnit(time=time, name="surplus_consumption",elec_to_therm_ratio=elec_therm_ratio,binary_elec= False, binary_therm=False)
    exp = f" + {copper_coeeff} * grid_imp_total_consumption_magnets[t] * {valo}[t] == surplus_consumption_elec_cons_p[t]"                                            
    cst = DefinitionDynamicConstraint(name="over_consumption_cst", t_range='for t in time.I', exp_t=exp, parent=None)
    setattr(surplus_consumption, 'over_consumption_cst', cst)

    # heat_node_2.connect_units(cnrs_heat,cciag)

    
    elec_node.connect_units(grid_imp, grid_exp, surplus_consumption.elec_consumption_unit, heat_pump.elec_consumption_unit)
    heat_node_1.connect_units(heat_dissipation, surplus_consumption.thermal_production_unit,heat_pump.thermal_consumption_unit,buffer)
    heat_node_2.connect_units(cnrs_heat,district_heat,heat_pump.thermal_production_unit) 

    # exp = f" + {heat_node_2._imports}[t] == heat_pump.thermal_consumption_unit_p[t]"                                            
    # cst = DefinitionDynamicConstraint(name="heat_pump_usage", t_range='for t in time.I', exp_t=exp, parent=None)
    # setattr(heat_pump, 'heat_pump_usage', cst)

    heat_node_2.export_to_node(heat_node_1,export_max=0)
    heat_node_1.export_to_node(heat_node_2,export_max=0)
   
    available = temp.available.tolist()
    exp = " + ".join([f"consumption_block_{j}_elec_cons_start_up[t]" for j in var_ids]) + f" <= {available}[t]"
    cst = DefinitionDynamicConstraint(name="defined_start_up", t_range='for t in time.I', exp_t=exp, parent=None)
    setattr(elec_node, 'defined_start_up', cst)

    # if "co2" in objectives:
    grid_imp.minimize_co2_emissions(weight  = weight, pareto = False)

    
    district_heat.minimize_co2_emissions(weight =  weight, pareto = False)
    
    buffer.discharge.minimize_co2_emissions(weight = weight)
    
    buffer.minimize_capacity(weight=weight)
    
    
    lca_years = 30
    gwp = (2204/1350) * 1000  # gco2/kwh
    
    gwp_storage = ( gwp /lca_years) *  (len(df.resample("1D").mean())/365)  #gco2/kwh
    
    # gwp_storage = 80   # example value

    storage_gwp_obj = Objective(
        name='storage_embodied_co2',
        exp=f'{buffer.name}_capacity * {gwp_storage}',
        weight=1,
        parent=buffer
    )

    setattr(buffer, 'storage_gwp_obj', storage_gwp_obj)
    heat_pump.thermal_production_unit.minimize_co2_emissions(weight= weight)
    # model.add_nodes(elec_node,heat_node)


    
    model.add_nodes(elec_node, heat_node_1, heat_node_2 , verbose = False)
    
    


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

        cmd = f"temp['consumption_block_{j}'] = consumption_block_{j}.elec_consumption_unit.p.get_value()"

        exec (cmd)

    
    
    df["grid_import_S3_kWh"] = grid_imp.p.get_value()
    df["grid_export_S3_kWh"] = grid_exp.p.get_value()
    df["OPEX_S3_€"] = df[["grid_import_S3_kWh", "electricity_price[€/kWh]"]].product(axis = 1) - df[["grid_export_S3_kWh", "elec_export_price[€/kWh]"]].product(axis = 1)

    df["Buffer_Cap_S3_KWp"] =buffer.capacity.get_value()
    df["Buffer_Power_S3_KW"] = buffer.p.get_value()
    df["Buffer_Energy_S3_KW"] = buffer.e.get_value()

    df["heatpump_elec_Power_S3_KW"] = heat_pump.elec_consumption_unit.p.get_value()
    df["heatpump_thermal_Power_S3_KW"] = heat_pump.thermal_production_unit.p.get_value()

    
    df["heatpump_elec_binary_S3"] = heat_pump.elec_consumption_unit.u.get_value()

    df["surplus_elec_cons_Power_S3_kW"] = surplus_consumption.elec_consumption_unit.p.get_value()
    df["surplus_therm_prod_Power_S3_kW"] = surplus_consumption.thermal_production_unit.p.get_value()

    df["estimated_CO2_S3_elec"] = grid_imp.co2_emissions.get_value()
    
    df["estimated_CO2_S3_district_heat"] = district_heat.co2_emissions.get_value()
    df["estimated_CO2_S3_buffer"] = buffer.discharge.co2_emissions.get_value()
    # df["estimated_CO2_S3_heatpump"] = heat_pump.thermal_production_unit.co2_emissions.get_value()

    df["estimated_CO2_S3_heatpump"] = df["heatpump_elec_Power_S3_KW"] * df["Taux de Co2"] * hp_cf
    # df["estimated_CO2_S3_pv"] = pv.co2_emissions.get_value()
    df["estimated_CO2_S3_total"] = df[["estimated_CO2_S3_elec","estimated_CO2_S3_district_heat",
                                       "estimated_CO2_S3_buffer"]].sum(axis =1).tolist()

    df["Consumption_S3"] = temp.sum(axis =1).tolist()
    df["Consumption_S3_compare"] = total_consumption.get_value()
    # df["PV_production_S3"] = df.pv.copy() * round(pv.nb_unit.get_value(),2)
    df["heat_demand_district_heat_S3_kW"] = district_heat.p.get_value()
    df["heat_demand_district_heat_S3_MW"] = df["heat_demand_district_heat_S3_kW"] /1000

    return df, stats


if __name__ == "__main__":
    # d_sets = [{"country" : "France", "PV_CO2" : 36, "hp_cf" : 1-0.80},{"country" : "Czech" , "PV_CO2" : 38.3 , "hp_cf" : 1-0.946}]
    d_sets = [{"country" : "France", "hp_cf" : 1-0.80},{"country" : "Czech" ,  "hp_cf" : 1-0.946}]

    for d_set in d_sets: 
        file = f"LNCMI/flexibility_study/data/{d_set["country"]}_results.csv"
        try:
            
            df = pd.read_csv(file, sep=",", decimal=".", index_col="datetime", parse_dates=True)
        except:
            file = f"./data/{d_set["country"]}_results.csv"
            df = pd.read_csv(file, sep=",", decimal=".", index_col="datetime", parse_dates=True)
        print (f"Data loaded from {file} , Starting Optimization Scenario 4 for {d_set["country"]}")
        
        
        
        df, stats = scenario(df, weight = 1,hp_cf = d_set["hp_cf"])

        with open(f'data/stats/Scenario_3_{d_set["country"]}.txt', 'w') as f:
            f.write(str(stats))
        
        df.to_csv(file, sep=",", decimal=".", index=True, index_label="datetime")

        print (f"Optimization Scenario 3 test completed, results saved to {file}")
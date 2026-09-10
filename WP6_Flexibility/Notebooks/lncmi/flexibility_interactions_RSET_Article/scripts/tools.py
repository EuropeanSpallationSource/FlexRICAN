import time as tm
import pulp
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_scenario_figure( country_1_df, country_2_df, scenarios, title, y_title,nrows,scale=1, layout = {},
                           shared_yaxes=True,  height=None, column_titles=["FR", "CZ"], n_cols = 2):
    """
    Create a FR/CZ scenario subplot figure.

    Parameters
    ----------
    scenarios : list of tuples
        Each tuple is:
        (row_title, france_column, czech_column, color)

    scale : float
        Divisor applied to the data, e.g. 1000 or 1e6.
    """

    if type(layout) is int:
        layout = get_layout(layout)

    elif type(layout) is not dict:
        pass

    figure = make_subplots( rows=nrows, cols=n_cols, shared_xaxes=True, shared_yaxes = shared_yaxes , column_titles = column_titles,
                                   row_titles=[s[0] for s in scenarios], vertical_spacing=0.05,  horizontal_spacing=0.055, 
                                     y_title=y_title,  )

    x = country_2_df.index

    for row, (name, fr_col, cz_col, color) in enumerate(scenarios, start=1):
        if n_cols== 2:
            figure.add_trace( 
                        go.Scatter( x=x, y=country_1_df[fr_col] / scale, mode="lines", name=name,  line={"color": color}, ),
                        row=row,
                        col=1,
                         )
        
            figure.add_trace(
                            go.Scatter(x=x, y=country_2_df[cz_col] / scale, mode="lines", name=f"{name} [CZ]", line={"color": color}, showlegend=False, ),
                            row=row,
                            col=2,
                        )
        elif n_cols == 1  :
            if row == 1 : 
                figure.add_trace( 
                                        go.Scatter( x=x, y=country_1_df[fr_col] / scale, mode="lines", name=name,  line={"color": color}, showlegend=False,),
                                        row=row,
                                        col=1,
                                        )
            elif row == 2 : 
                figure.add_trace(
                                            go.Scatter(x=x, y=country_2_df[cz_col] / scale, mode="lines", name=f"{name} [CZ]", line={"color": color}, showlegend=False, ),
                                            row=row,
                                            col=1,
                                        )
            # break

    figure.update_layout(layout,  title={ "text": title,
                                            "x": 0.7,
                                            "y": 0.99,
                                            "xanchor": "center",
                                            "font": {"color": "black"},
                                        },
                        height=height,
                    )

    figure.layout.title["font"] = {"color": "black"}
    figure.layout.title["xanchor"] = "right"

    for annot in figure.layout["annotations"]:
        annot["font"]["size"] = 32
        annot["font"]["color"] = "black"

    return figure



def get_lp_stats(model, solve_time=None):
    variables = model.variables()

    n_true_binary = 0     
    n_binary_like = 0     
    n_integer_other = 0   
    n_continuous = 0

    for v in variables:
        if v.cat == pulp.LpBinary:
            n_true_binary += 1
        elif v.cat == pulp.LpInteger:
            if v.lowBound == 0 and v.upBound == 1:
                n_binary_like += 1
            else:
                n_integer_other += 1
        else:  
            n_continuous += 1

    stats = {
        "n_variables_total": len(variables),
        "n_binary": n_true_binary + n_binary_like,
        "n_binary_declared": n_true_binary,
        "n_binary_by_bounds": n_binary_like,
        "n_integer_general": n_integer_other,
        "n_continuous": n_continuous,
        "n_constraints": len(model.constraints),
        "status": pulp.LpStatus[model.status],
        "solve_time_s": solve_time,
    }
    return stats


def post_processing(df:pd.DataFrame = pd.DataFrame()):
    df["electric_emissions_ref"] = df[["Electricity_Consumption[MW]" , "Taux de Co2"]].product(axis= 1) * 1000

    df["thermal_emissions_ref"] = df[["Heat_demand[MW]" , "cciag_co2"]].product(axis= 1) * 1000

    df["Reference_emissions"] = df[["electric_emissions_ref", "thermal_emissions_ref"]].sum(axis = 1)

    df["Electricity_Consumption[kW]"] = df["Electricity_Consumption[MW]"] * 1000

    df["heatpump_thermal_Power_S3_MW"] = df["heatpump_thermal_Power_S3_KW"] / 1e3

    df["heatpump_thermal_Power_S4_MW"] = df["heatpump_thermal_Power_S4_KW"] / 1e3

    return  df

def get_layout(type: int = 1):
    if type == 1:
        layout = {"height" : 900, "width": 1400, "template" : "plotly_white",
                    "title" : "",
                    "font":{"size" : 28},

                'xaxis1': {
                    'zerolinewidth': 2,
                    'zerolinecolor': 'black',
                    'showticklabels': True,
                    "tickangle": -45,  
                    'showline': True, 
                    "title": "Date",
                    'linewidth': 2,  
                    'linecolor': 'black',  
                    'mirror': True  # This creates the frame by mirroring the axis line
                },
                "yaxis1": {
                    "side": 'left',
                    "range": [0, 24],
                    "dtick": 4,
                    'showline': True,
                    'linewidth': 2,
                    'linecolor': 'black',
                    'mirror': True
                },
            
                    "legend" : {"tracegroupgap":8 ,"font_size": 28, "orientation":"h", "yanchor":"bottom","xanchor":"center",
                                "y":-0.152,"x":0.5, "title" : "","itemwidth" : 60
                                },
                                
                                "showlegend": False

                    }


    else:
        layout = {"height" : 1800, "width": 2000, "template" : "plotly_white",
        "font":{"size" : 22},

        "legend" : {"tracegroupgap":8 ,"font_size": 28, "orientation":"h", "yanchor":"bottom","xanchor":"center",
                    "y":-0.12,"x":0.5, "title" : "","itemwidth" : 60
                    },
                    
                    "showlegend": True

        }

        for i in range (1,18):
            layout[f"xaxis{i}"] = { 'showline': True, "tickangle": -45,'showgrid': True,'gridcolor': 'lightgrey',
                'gridwidth': 1, 'linewidth': 2,  'linecolor': 'black',  'mirror': True}
            layout[f"yaxis{i}"] = {'zerolinewidth': 1,'zerolinecolor': 'black', 'showline': True,
                                'showgrid': True,'gridcolor': 'lightgrey','gridwidth': 1, 
                                'linewidth': 2,  'linecolor': 'black',  'mirror': True}

    return layout



def summary(country_1 :str =  "France" , country_1_df = None ,  country_2:str = "Czech Republic" , country_2_df = None):
    country_1 = f"{country_1} [t CO2_eqv]"
    country_2 = f"{country_2} [t CO2_eqv]"
    lca_years = 30
    gwp = (2204/1350) * 1000  # gco2/kwh

    gwp_storage = ( gwp /lca_years) *  (len(country_1_df.resample("1D").mean())/365) / 1e6 #convert to tonnes


    cols = [col for col in country_1_df.columns if "Ref" in col or "total" in col]

    table = pd.DataFrame({
        country_1 : country_1_df[cols].sum() / 1e6,
        country_2 : country_2_df[cols].sum() / 1e6
    })

    table = table.rename(index = {"estimated_CO2_S1_total" :"SS",
                        "estimated_CO2_S2_total" :"SSOP",
                        "estimated_CO2_S3_total" :"OPHR",
                        "estimated_CO2_S4_total" :"SSOPHR",
                        "Reference_emissions" : "Reference"
                        })

    table.loc["OPHR", country_1] += gwp_storage + (country_1_df.loc[ : , "estimated_CO2_S3_heatpump"].sum()/1e6)
    table.loc["OPHR", country_2] += gwp_storage + (country_2_df.loc[ : , "estimated_CO2_S3_heatpump"].sum()/1e6)

    table.loc["SSOPHR", country_1] += gwp_storage + (country_1_df.loc[ : , "estimated_CO2_S3_heatpump"].sum()/1e6)
    table.loc["SSOPHR", country_2] += gwp_storage + (country_2_df.loc[ : , "estimated_CO2_S3_heatpump"].sum()/1e6)

    table[f"{country_1.split(' ')[0]}_gain[%]"] = (1 - table[country_1]/ table.loc["Reference" , country_1]) * 100
    table[f"{country_2.split(' ')[0]}_gain[%]"] = (1 - table[country_2]/ table.loc["Reference" , country_2]) * 100

    table.loc["SS" , f"{country_1.split(' ')[0]}_PV_Capacity[MWp]"] = country_1_df["PV_Cap_S1_KWp" ].mean()/1e3
    table.loc["SSOP" , f"{country_1.split(' ')[0]}_PV_Capacity[MWp]"] = country_1_df["PV_Cap_S2_KWp" ].mean()/1e3
    table.loc["SSOPHR" , f"{country_1.split(' ')[0]}_PV_Capacity[MWp]"] = country_1_df["PV_Cap_S4_KWp" ].mean()/1e3

    table.loc["SS" , f"{country_2.split(' ')[0]}_PV_Capacity[MWp]"] = country_2_df["PV_Cap_S1_KWp" ].mean()/1e3
    table.loc["SSOP" , f"{country_2.split(' ')[0]}_PV_Capacity[MWp]"] = country_2_df["PV_Cap_S2_KWp" ].mean()/1e3
    table.loc["SSOPHR" , f"{country_2.split(' ')[0]}_PV_Capacity[MWp]"] = country_2_df["PV_Cap_S4_KWp" ].mean()/1e3


    
    table.loc["OPHR" , f"{country_1.split(' ')[0]}_Buffer_Capacity[MWh]"] = country_1_df["Buffer_Cap_S3_KWp" ].mean()/1e3
    table.loc["SSOPHR" , f"{country_1.split(' ')[0]}_Buffer_Capacity[MWh]"] = country_1_df["Buffer_Cap_S4_KWp" ].mean()/1e3

    
    table.loc["OPHR" , f"{country_2.split(' ')[0]}_Buffer_Capacity[MWh]"] = country_2_df["Buffer_Cap_S3_KWp" ].mean()/1e3
    table.loc["SSOPHR" , f"{country_2.split(' ')[0]}_Buffer_Capacity[MWh]"] = country_2_df["Buffer_Cap_S4_KWp" ].mean()/1e3

    return table.round(2)

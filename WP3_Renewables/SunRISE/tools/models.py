
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd


class FinancialModels:
    """Financial viability models for renewable energy projects."""

    def __init__(self, default_discount_rate: float = 0.07, currency: str = "€"):
        """
        Initialize financial model defaults.

        Parameters
        ----------
        default_discount_rate : float, optional
            Discount rate used when none is provided to methods.
        currency : str, optional
            Currency symbol used for reporting context.
        """
        self.default_discount_rate = default_discount_rate
        self.currency = currency

    @staticmethod
    def _scenario_name(column_name: str) -> str:
        """Create a scenario label from a generation column name."""
        if "[" in column_name and "]" in column_name:
            scenario = column_name.split("[", 1)[1].split("]", 1)[0].strip()
            return scenario.replace(" ", "_") if scenario else "base"

        if column_name in {"Power_ac", "Power_dc"}:
            return "base"

        return column_name.replace(" ", "_")

    @staticmethod
    def _infer_step_hours(index: pd.DatetimeIndex) -> float:
        """Infer time step in hours from a DatetimeIndex."""
        deltas = index.to_series().diff().dropna()
        if deltas.empty:
            return 1.0

        step_seconds = float(deltas.dt.total_seconds().median())
        if step_seconds <= 0:
            return 1.0

        return step_seconds / 3600.0

    @classmethod
    def prepare_annual_energy(
        cls,
        generation_df: pd.DataFrame,
        power_columns: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
        power_unit: str = "W",
        freq: str = "1YE",
        label: str = "right",
    ) -> pd.DataFrame:
        """
        Convert power time-series to annual energy (kWh) by scenario.

        Parameters
        ----------
        generation_df : pandas.DataFrame
            Time-series DataFrame with a DatetimeIndex and power columns
            (for example columns from ``df_1``).
        power_columns : Sequence[str] | Mapping[str, str] | None, optional
            - If ``None``, columns are auto-detected (prefers ``Power_ac*``,
              then ``Power_dc*``).
            - If sequence, each listed column is used as one scenario.
            - If mapping, keys are scenario names and values are column names.
        power_unit : str, optional
            Power unit in ``generation_df``. Supported: ``W``, ``kW``, ``MW``.
        freq : str, optional
            Resample frequency for annual aggregation.
        label : str, optional
            Bin labeling for resampling (``left`` or ``right``).

        Returns
        -------
        pandas.DataFrame
            Annual energy in kWh with one column per scenario.
        """
        if not isinstance(generation_df.index, pd.DatetimeIndex):
            raise ValueError("generation_df must use a DatetimeIndex.")

        if power_columns is None:
            ac_columns = [name for name in generation_df.columns if "Power_ac" in str(name)]
            dc_columns = [name for name in generation_df.columns if "Power_dc" in str(name)]
            selected_columns = ac_columns if ac_columns else dc_columns

            if not selected_columns:
                raise ValueError(
                    "No power columns found. Provide power_columns or include "
                    "columns like 'Power_ac'/'Power_dc'."
                )

            scenario_to_column = {
                cls._scenario_name(column_name): column_name
                for column_name in selected_columns
            }
        elif isinstance(power_columns, Mapping):
            scenario_to_column = dict(power_columns)
        else:
            scenario_to_column = {
                cls._scenario_name(column_name): column_name
                for column_name in power_columns
            }

        missing_columns = [
            column_name
            for column_name in scenario_to_column.values()
            if column_name not in generation_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Columns not found in generation_df: {missing_columns}")

        selected = generation_df[list(scenario_to_column.values())]
        step_hours = cls._infer_step_hours(generation_df.index)

        unit_normalized = power_unit.strip().lower()
        if unit_normalized == "w":
            conversion = step_hours / 1000.0
        elif unit_normalized == "kw":
            conversion = step_hours
        elif unit_normalized == "mw":
            conversion = step_hours * 1000.0
        else:
            raise ValueError("power_unit must be one of: 'W', 'kW', 'MW'.")

        annual_energy = selected.resample(freq, label=label).sum(min_count=1) * conversion

        scenario_columns = {
            column_name: scenario_name
            for scenario_name, column_name in scenario_to_column.items()
        }
        annual_energy = annual_energy.rename(columns=scenario_columns)

        return annual_energy

    @staticmethod
    def _to_lifetime_energy(
        values: Union[pd.Series, Sequence[float], float],
        years: pd.RangeIndex,
        degradation_rate: float,
        production_start_year: int,
        baseline_method: str,
    ) -> pd.Series:
        """Convert input energy to a lifetime yearly schedule in kWh."""
        if isinstance(values, (int, float)):
            baseline = float(values)
            schedule = pd.Series(0.0, index=years, dtype=float)
            for year in years:
                if year >= production_start_year:
                    schedule.loc[year] = baseline * (
                        (1 - degradation_rate) ** (year - production_start_year)
                    )
            return schedule

        series = pd.Series(values).dropna().astype(float).reset_index(drop=True)
        if series.empty:
            raise ValueError("Energy input has no valid values.")

        if len(series) == len(years):
            return pd.Series(series.values, index=years, dtype=float)

        baseline_method_normalized = baseline_method.lower()
        if baseline_method_normalized == "first":
            baseline = float(series.iloc[0])
        elif baseline_method_normalized == "last":
            baseline = float(series.iloc[-1])
        else:
            baseline = float(series.mean())

        schedule = pd.Series(0.0, index=years, dtype=float)
        for year in years:
            if year >= production_start_year:
                schedule.loc[year] = baseline * (
                    (1 - degradation_rate) ** (year - production_start_year)
                )

        return schedule

    @staticmethod
    def _npv(rate: float, cashflows: Sequence[float]) -> float:
        """Compute NPV from cashflows and discount rate."""
        if rate <= -1:
            return float("inf")

        value = 0.0
        for year, cashflow in enumerate(cashflows):
            value += float(cashflow) / ((1 + rate) ** year)
        return value

    @classmethod
    def _irr(
        cls,
        cashflows: Sequence[float],
        lower: float = -0.999,
        upper: float = 1.0,
        tolerance: float = 1e-7,
        max_iter: int = 200,
    ) -> Optional[float]:
        """Estimate IRR using bisection. Returns None if no real bracket is found."""
        npv_low = cls._npv(lower, cashflows)
        npv_high = cls._npv(upper, cashflows)

        if npv_low == 0:
            return lower
        if npv_high == 0:
            return upper

        for _ in range(20):
            if npv_low * npv_high < 0:
                break
            upper *= 2
            npv_high = cls._npv(upper, cashflows)
        else:
            return None

        left, right = lower, upper
        for _ in range(max_iter):
            middle = (left + right) / 2.0
            npv_middle = cls._npv(middle, cashflows)

            if abs(npv_middle) <= tolerance:
                return middle

            if npv_low * npv_middle < 0:
                right = middle
                npv_high = npv_middle
            else:
                left = middle
                npv_low = npv_middle

        return (left + right) / 2.0

    @staticmethod
    def _first_non_negative_year(series: pd.Series) -> Optional[int]:
        """Return first year index where the cumulative series is non-negative."""
        reached = series[series >= 0]
        if reached.empty:
            return None
        return int(reached.index[0])

    def evaluate_project(
        self,
        annual_energy_kwh: Union[
            pd.DataFrame,
            pd.Series,
            Mapping[str, Sequence[float]],
            Sequence[float],
            float,
            int,
        ],
        project_lifetime: int,
        capex: float,
        annual_opex: Optional[float] = None,
        opex_rate: float = 0.03,
        periodic_cost: float = 0.0,
        periodic_cost_frequency: int = 5,
        fuel_cost: float = 0.0,
        discount_rate: Optional[float] = None,
        degradation_rate: float = 0.005,
        electricity_price: float = 0.2,
        price_escalation_rate: float = 0.02,
        production_start_year: int = 1,
        baseline_method: str = "mean",
    ) -> dict:
        """
        Build yearly cashflows and compute financial indicators.

        Parameters
        ----------
        annual_energy_kwh : DataFrame | Series | mapping | sequence | float
            Annual energy input in kWh. Multiple scenarios are supported.
            If historical values are provided (length differs from project lifetime),
            a baseline is inferred and projected with degradation.
        project_lifetime : int
            Project duration in years.
        capex : float
            Initial investment at year 0.
        annual_opex : float | None, optional
            Fixed annual operating cost. If None, uses ``capex * opex_rate``.
        opex_rate : float, optional
            OPEX ratio of CAPEX when ``annual_opex`` is not supplied.
        periodic_cost : float, optional
            Recurring major cost value.
        periodic_cost_frequency : int, optional
            Frequency (years) of periodic cost.
        fuel_cost : float, optional
            Annual fuel cost (useful for hybrid systems).
        discount_rate : float | None, optional
            Discount rate. Defaults to ``self.default_discount_rate``.
        degradation_rate : float, optional
            Annual production degradation.
        electricity_price : float, optional
            Year-0 electricity value in currency/kWh.
        price_escalation_rate : float, optional
            Annual escalation of electricity price.
        production_start_year : int, optional
            Year when production starts.
        baseline_method : str, optional
            Baseline rule when projecting from historical energy.
            Supported: ``mean``, ``first``, ``last``.

        Returns
        -------
        dict
            {
                "kpis": DataFrame indexed by scenario,
                "cashflow": yearly DataFrame with costs/revenues/cashflows
            }
        """
        if project_lifetime < 1:
            raise ValueError("project_lifetime must be >= 1")

        discount = self.default_discount_rate if discount_rate is None else discount_rate
        yearly_opex = capex * opex_rate if annual_opex is None else annual_opex

        years = pd.RangeIndex(0, project_lifetime + 1, name="Year")

        if isinstance(annual_energy_kwh, pd.DataFrame):
            energy_input = annual_energy_kwh.copy()
        elif isinstance(annual_energy_kwh, pd.Series):
            energy_input = annual_energy_kwh.to_frame(name="base")
        elif isinstance(annual_energy_kwh, Mapping):
            energy_input = pd.DataFrame(dict(annual_energy_kwh))
        elif isinstance(annual_energy_kwh, (int, float)):
            energy_input = pd.DataFrame({"base": [float(annual_energy_kwh)]})
        else:
            energy_input = pd.DataFrame({"base": list(annual_energy_kwh)})

        if energy_input.empty:
            raise ValueError("annual_energy_kwh must contain at least one value.")

        cashflow = pd.DataFrame(index=years, dtype=float)
        cashflow["CAPEX"] = 0.0
        cashflow.loc[0, "CAPEX"] = float(capex)

        cashflow["Operations_Cost"] = 0.0
        cashflow.loc[cashflow.index >= production_start_year, "Operations_Cost"] = float(yearly_opex)

        if periodic_cost_frequency <= 0:
            periodic_series = pd.Series(0.0, index=years, dtype=float)
        else:
            periodic_series = pd.Series(
                [
                    float(periodic_cost)
                    if (year > 0 and year % periodic_cost_frequency == 0)
                    else 0.0
                    for year in years
                ],
                index=years,
                dtype=float,
            )

        cashflow["Periodic_Costs"] = periodic_series
        cashflow["Fuel_Cost"] = 0.0
        cashflow.loc[cashflow.index >= production_start_year, "Fuel_Cost"] = float(fuel_cost)

        cashflow["Annual_Cost"] = cashflow[["CAPEX", "Operations_Cost", "Periodic_Costs", "Fuel_Cost"]].sum(axis=1)
        cashflow["Discount_Factor"] = (1.0 + float(discount)) ** cashflow.index.astype(float)
        cashflow["Discounted_Annual_Costs"] = cashflow["Annual_Cost"] / cashflow["Discount_Factor"]
        cashflow["Electricity_Price"] = float(electricity_price) * (
            (1.0 + float(price_escalation_rate)) ** cashflow.index.astype(float)
        )

        scenario_results = {}

        for scenario in energy_input.columns:
            schedule = self._to_lifetime_energy(
                values=energy_input[scenario],
                years=years,
                degradation_rate=degradation_rate,
                production_start_year=production_start_year,
                baseline_method=baseline_method,
            )

            generation_col = f"Annual_Generation_{scenario}_kWh"
            revenue_col = f"Annual_Revenue_{scenario}"
            net_cashflow_col = f"Net_CashFlow_{scenario}"
            discounted_energy_col = f"Discounted_Annual_Energy_{scenario}"
            discounted_revenue_col = f"Discounted_Annual_Revenue_{scenario}"
            discounted_net_cashflow_col = f"Discounted_Net_CashFlow_{scenario}"
            cumulative_col = f"Cumulative_Net_CashFlow_{scenario}"
            discounted_cumulative_col = f"Discounted_Cumulative_Net_CashFlow_{scenario}"

            cashflow[generation_col] = schedule
            cashflow[revenue_col] = cashflow[generation_col] * cashflow["Electricity_Price"]
            cashflow[discounted_energy_col] = cashflow[generation_col] / cashflow["Discount_Factor"]
            cashflow[discounted_revenue_col] = cashflow[revenue_col] / cashflow["Discount_Factor"]

            cashflow[net_cashflow_col] = cashflow[revenue_col] - cashflow["Annual_Cost"]
            cashflow[discounted_net_cashflow_col] = (
                cashflow[discounted_revenue_col] - cashflow["Discounted_Annual_Costs"]
            )

            cashflow[cumulative_col] = cashflow[net_cashflow_col].cumsum()
            cashflow[discounted_cumulative_col] = cashflow[discounted_net_cashflow_col].cumsum()

            npv_costs = float(cashflow["Discounted_Annual_Costs"].sum())
            npv_energy = float(cashflow[discounted_energy_col].sum())
            npv_revenue = float(cashflow[discounted_revenue_col].sum())
            npv = float(cashflow[discounted_net_cashflow_col].sum())

            scenario_results[scenario] = {
                "LCOE_per_kWh": (npv_costs / npv_energy) if npv_energy > 0 else float("nan"),
                "NPV": npv,
                "IRR": self._irr(cashflow[net_cashflow_col].tolist()),
                "Simple_Payback_Year": self._first_non_negative_year(cashflow[cumulative_col]),
                "Discounted_Payback_Year": self._first_non_negative_year(cashflow[discounted_cumulative_col]),
                "Benefit_Cost_Ratio": (npv_revenue / npv_costs) if npv_costs > 0 else float("nan"),
                "NPV_to_CAPEX": (npv / capex) if capex else float("nan"),
            }

        kpis = pd.DataFrame.from_dict(scenario_results, orient="index")
        kpis.index.name = "Scenario"

        return {
            "kpis": kpis,
            "cashflow": cashflow,
        }

    def build_lcoe_evolution(
        self,
        cashflow: pd.DataFrame,
        kpis: pd.DataFrame,
        discounted: bool = True,
    ) -> pd.DataFrame:
        """
        Build running LCOE evolution by scenario over project lifetime.

        Parameters
        ----------
        cashflow : pandas.DataFrame
            Cashflow table returned by ``evaluate_project``.
        kpis : pandas.DataFrame
            KPI table returned by ``evaluate_project`` (indexed by scenario).
        discounted : bool, optional
            If True, uses discounted costs and discounted energy.
            If False, uses non-discounted costs and annual generation.

        Returns
        -------
        pandas.DataFrame
            Long-format DataFrame with columns:
            ``Year``, ``Scenario``, ``Cumulative_Cost``,
            ``Cumulative_Energy_kWh``, ``Running_LCOE_per_kWh``,
            ``Final_LCOE_per_kWh``.
        """
        cost_column = "Discounted_Annual_Costs" if discounted else "Annual_Cost"
        if cost_column not in cashflow.columns:
            raise ValueError(f"'{cost_column}' not found in cashflow DataFrame.")

        if "LCOE_per_kWh" not in kpis.columns:
            raise ValueError("'LCOE_per_kWh' not found in KPI DataFrame.")

        if isinstance(cashflow.index, pd.DatetimeIndex):
            years = cashflow.index.year
        else:
            years = pd.Index(cashflow.index)

        evolution_frames = []

        for scenario in kpis.index:
            scenario_name = str(scenario)
            energy_column = (
                f"Discounted_Annual_Energy_{scenario_name}"
                if discounted
                else f"Annual_Generation_{scenario_name}_kWh"
            )

            if energy_column not in cashflow.columns:
                continue

            cumulative_cost = cashflow[cost_column].cumsum()
            cumulative_energy = cashflow[energy_column].cumsum()
            running_lcoe = cumulative_cost / cumulative_energy.where(cumulative_energy > 0)

            scenario_df = pd.DataFrame(
                {
                    "Year": years,
                    "Scenario": scenario_name,
                    "Cumulative_Cost": cumulative_cost.values,
                    "Cumulative_Energy_kWh": cumulative_energy.values,
                    "Running_LCOE_per_kWh": running_lcoe.values,
                    "Final_LCOE_per_kWh": float(kpis.loc[scenario, "LCOE_per_kWh"]),
                }
            )
            evolution_frames.append(scenario_df)

        if not evolution_frames:
            raise ValueError(
                "No matching scenario energy columns were found in cashflow. "
                "Expected columns like 'Discounted_Annual_Energy_<scenario>' or "
                "'Annual_Generation_<scenario>_kWh'."
            )

        return pd.concat(evolution_frames, ignore_index=True)

    def evaluate_from_generation_df(
        self,
        generation_df: pd.DataFrame,
        power_columns: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
        power_unit: str = "W",
        energy_freq: str = "1YE",
        energy_label: str = "right",
        **evaluate_kwargs,
    ) -> dict:
        """
        Convenience method: compute KPIs directly from generation time-series.

        Returns the same output as ``evaluate_project`` and also includes
        the extracted annual energy DataFrame under ``annual_energy_kwh``.
        """
        annual_energy_kwh = self.prepare_annual_energy(
            generation_df=generation_df,
            power_columns=power_columns,
            power_unit=power_unit,
            freq=energy_freq,
            label=energy_label,
        )

        result = self.evaluate_project(
            annual_energy_kwh=annual_energy_kwh,
            **evaluate_kwargs,
        )
        result["annual_energy_kwh"] = annual_energy_kwh

        return result


class EnvironmentalModels:
    """Environmental and carbon-accounting models for renewable projects."""

    def __init__(
        self,
        carbon_capex: float = 80000.0,
        emissions_file: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize environmental model defaults.

        Parameters
        ----------
        carbon_capex : float, optional
            Embodied carbon factor used in the notebook formula:
            ``factor * installed_capacity_kW``.
        emissions_file : str | Path | None, optional
            Path to emissions factors file. Defaults to ``ESS/data/emissions.txt``.
        """
        self.carbon_capex = carbon_capex
        default_emissions_file = Path(__file__).resolve().parents[1] / "data" / "emissions.txt"
        self.emissions_file = Path(emissions_file) if emissions_file is not None else default_emissions_file

    def _load_emissions_factors(self) -> dict:
        """Load emissions dictionaries from the configured file."""
        if not self.emissions_file.exists():
            raise FileNotFoundError(
                f"Emissions factors file not found: {self.emissions_file}. "
                "Create ESS/data/emissions.txt first."
            )

        with self.emissions_file.open("r", encoding="utf-8") as stream:
            factors = json.load(stream)

        if "panel_type_factors" not in factors or "grid_emission_factors" not in factors:
            raise ValueError(
                "Emissions file must contain 'panel_type_factors' and 'grid_emission_factors'."
            )

        return factors

    def available_emissions_factors(self) -> dict:
        """Return available panel types and grid regions from emissions file."""
        factors = self._load_emissions_factors()
        return {
            "panel_types": list(factors["panel_type_factors"].keys()),
            "grid_regions": list(factors["grid_emission_factors"].keys()),
        }

    def carbon_payback_emissions_factor(
        self,
        pv_type: str,
        capacity_mw: float,
        annual_output_kwh: float,
        lifetime_years: int,
        manufacturing_country: str,
        installation_country: str,
        installation_year: Optional[int] = None,
    ) -> dict:
        """
        Replicate CO2 payback calculations from the SunRISE notebook.

        The method reproduces the referenced formulas:
        - ``lifetime_output = annual_output_kwh * lifetime_years``
        - ``total_lifecycle_co2 = lifetime_output * (pv_emission_factor / 1000)``
        - ``annual_co2_avoided = annual_output_kwh * ((grid - pv) / 1000)``
        - ``co2_payback_time = total_lifecycle_co2 / annual_co2_avoided``

        Returns summary, annual avoided emissions, and cumulative trajectories.
        """
        if annual_output_kwh <= 0:
            raise ValueError("annual_output_kwh must be > 0")
        if lifetime_years <= 0:
            raise ValueError("lifetime_years must be > 0")

        factors = self._load_emissions_factors()
        panel_type_factors = factors["panel_type_factors"]
        grid_emission_factors = factors["grid_emission_factors"]

        if pv_type not in panel_type_factors:
            raise ValueError(
                f"Unknown pv_type '{pv_type}'. Available: {list(panel_type_factors.keys())}"
            )

        manufacturing_factors = panel_type_factors[pv_type]
        if manufacturing_country not in manufacturing_factors:
            raise ValueError(
                f"Unknown manufacturing_country '{manufacturing_country}' for '{pv_type}'. "
                f"Available: {list(manufacturing_factors.keys())}"
            )

        if installation_country not in grid_emission_factors:
            raise ValueError(
                f"Unknown installation_country '{installation_country}'. "
                f"Available: {list(grid_emission_factors.keys())}"
            )

        pv_emission_factor = float(manufacturing_factors[manufacturing_country])
        grid_emission_factor = float(grid_emission_factors[installation_country])

        lifetime_output = float(annual_output_kwh) * int(lifetime_years)
        total_lifecycle_co2 = lifetime_output * (pv_emission_factor / 1000.0)
        annual_co2_avoided = float(annual_output_kwh) * (
            (grid_emission_factor - pv_emission_factor) / 1000.0
        )

        co2_payback_time = (
            total_lifecycle_co2 / annual_co2_avoided
            if annual_co2_avoided > 0
            else float("inf")
        )

        years_index = pd.Index(range(0, int(lifetime_years) + 1), name="Year")
        years_float = pd.Series(years_index.astype(float), index=years_index)

        cumulative_savings = (annual_co2_avoided * years_float) - total_lifecycle_co2
        cumulative_pv_emissions = pd.Series(total_lifecycle_co2, index=years_index, dtype=float)
        cumulative_grid_emissions = (
            float(annual_output_kwh) * grid_emission_factor / 1000.0
        ) * years_float

        crossing = cumulative_savings[cumulative_savings >= 0]
        payback_year = int(crossing.index[0]) if not crossing.empty else None

        annual_index = pd.Index(range(1, int(lifetime_years) + 1), name="Year")
        annual_avoided_emissions = pd.DataFrame(
            {"Annual_CO2_Avoided_kg": float(annual_co2_avoided)},
            index=annual_index,
        )
        annual_avoided_emissions["Cumulative_CO2_Savings_kg"] = (
            annual_avoided_emissions["Annual_CO2_Avoided_kg"].cumsum() - total_lifecycle_co2
        )

        trajectory = pd.DataFrame(
            {
                "Cumulative_CO2_Savings_kg": cumulative_savings.values,
                "Cumulative_PV_Emissions_kg": cumulative_pv_emissions.values,
                "Cumulative_Grid_Emissions_kg": cumulative_grid_emissions.values,
            },
            index=years_index,
        )

        summary = pd.DataFrame(
            {
                "Parameter": [
                    "Type of PV panel",
                    "Installed capacity (MW)",
                    "Annual electricity output (kWh/year)",
                    "System lifetime (years)",
                    "Lifetime electricity output (kWh)",
                    "PV CO2 emission factor (g CO2/kWh)",
                    "Grid CO2 intensity (g CO2/kWh)",
                    "Total life-cycle CO2 emissions (kg)",
                    "Annual CO2 avoided (kg/year)",
                    "CO2 payback time (years)",
                    "CO2 payback year (integer crossing)",
                    "Installation country",
                    "Manufacturing country",
                    "Installation year",
                ],
                "Value": [
                    pv_type,
                    float(capacity_mw),
                    float(annual_output_kwh),
                    int(lifetime_years),
                    float(lifetime_output),
                    float(pv_emission_factor),
                    float(grid_emission_factor),
                    float(total_lifecycle_co2),
                    float(annual_co2_avoided),
                    float(co2_payback_time),
                    payback_year,
                    installation_country,
                    manufacturing_country,
                    installation_year,
                ],
            }
        )

        return {
            "summary": summary,
            "annual_avoided_emissions": annual_avoided_emissions,
            "trajectory": trajectory,
            "metrics": {
                "pv_emission_factor_g_per_kwh": pv_emission_factor,
                "grid_emission_factor_g_per_kwh": grid_emission_factor,
                "total_lifecycle_co2_kg": total_lifecycle_co2,
                "annual_co2_avoided_kg": annual_co2_avoided,
                "co2_payback_time_years": co2_payback_time,
                "co2_payback_year_crossing": payback_year,
            },
        }

    def carbon_payback_avoided(
        self,
        df_co: pd.DataFrame,
        countries: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Compute avoided CO2 and cumulative balance for clear/cloudy scenarios."""
        selected_countries = list(countries) if countries is not None else ["SE", "CZ"]

        # Compute avoided CO2 for each hour under each scenario
        # “SE” and “CZ” are assumed to be emission intensities (kg CO2 per unit power)
        for country in selected_countries:
            df_co[f"Avoided_Co2_{country}_ClearSky"] = df_co["Clear_sky_AC"] * df_co[country]
            df_co[f"Cumulative_{country}_ClearSky"] = (
                df_co[f"Avoided_Co2_{country}_ClearSky"].cumsum() - self.carbon_capex
            )


            df_co[f"Avoided_Co2_{country}_CloudySky"] = df_co["Cloudy_sky_AC"] * df_co[country]
            df_co[f"Cumulative_{country}_CloudySky"] = (
                df_co[f"Avoided_Co2_{country}_CloudySky"].cumsum() - self.carbon_capex
            )

        return df_co
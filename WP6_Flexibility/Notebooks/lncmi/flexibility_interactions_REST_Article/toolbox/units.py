
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# -----------------------------------------------------------------------------
from omegalpes.energy.units.production_units import SeveralProductionUnit
from omegalpes.general.optimisation.elements import Objective
# from omegalpes.energy.units.energy_units import EnergyUnit

from omegalpes.energy.units.consumption_units import  VariableConsumptionUnit,FixedConsumptionUnit, ShiftableConsumptionUnit
from omegalpes.energy.units.production_units import FixedProductionUnit, VariableProductionUnit
from omegalpes.energy.energy_types import elec, thermal
from omegalpes.general.optimisation.elements import  TechnicalConstraint, DefinitionDynamicConstraint, Quantity
from omegalpes.energy.units.conversion_units import  ConversionUnit,HeatPump
from pulp import LpBinary, LpInteger, LpContinuous

from omegalpes.energy.units.energy_units import VariableEnergyUnit
from omegalpes.energy.units.storage_units import StorageUnit
from omegalpes.energy.energy_nodes import EnergyNode
from omegalpes.general.optimisation.elements import *
from omegalpes.general.optimisation.model import OptimisationModel
from omegalpes.general.time import TimeUnit
from pulp import LpStatus, GUROBI_CMD,PULP_CBC_CMD

class ShiftableEnergyUnit(VariableEnergyUnit):
    """
    **Description**

        EnergyUnit with shiftable power profile.

    **Attributes**

        * power_values : power profile to shift (kW)
        * mandatory : indicates if the power is mandatory (True) or not (False)
        * starting_cost : cost of the starting of the EnergyUnit
        * operating_cost : cost of the operation (€/kW)
        * energy_type : type of energy ('Electrical', 'Heat', ...)

    """

    def __init__(self, time, name: str, flow_direction, power_values: list,
                 mandatory=True, co2_out=None, starting_cost=None,
                 operating_cost=None, energy_type=None,binary = True,
                 verbose=True, ):
        # Crop the power profile
        while power_values[0] == 0:
            power_values = power_values[1:]
        while power_values[-1] == 0:
            power_values = power_values[:-1]

        # Works if all values are strictly positives
        epsilon = 0.00001 * min(p > 0 for p in power_values)
        power_profile = [max(epsilon, p) for p in power_values]

        e_max = sum(power_profile) * time.DT
        print (f"emax : {e_max}")
        if mandatory:
            e_min = e_max
        else:
            e_min = 0

        p_min = min(power_profile)
        p_max = max(power_profile)

        # if available_start_up is None:
        #     available_start_up = [1] * time.LEN
        # else:
        #     if len(available_start_up) != time.LEN:
        #         raise ValueError(
        #             f"{name}: 'available_start_up' must have {time.LEN} entries, got {len(available_start_up)}"
        #         )

        VariableEnergyUnit.__init__(self, time, name=name,
                                    flow_direction=flow_direction,
                                    p_min=p_min, p_max=p_max, e_min=e_min,
                                    e_max=e_max, starting_cost=starting_cost,
                                    operating_cost=operating_cost,
                                    min_time_on=None, min_time_off=None,
                                    max_ramp_up=None, max_ramp_down=None,
                                    co2_out=co2_out, availability_hours=None,
                                    energy_type=energy_type,binary = binary,
                                    verbose=verbose,
                                    no_warn=True)

        # self._add_start_up()

        S = np.arange(0, time.LEN ,1 )
        Slen = len(S)
        self.start_up =  Quantity(name='start_up',
                                 description='The EnergyUnit is '
                                             'starting :1 or not :0',
                                 vtype=LpBinary, vlen=Slen, parent=self)

        # self.set_no_overshoot = TechnicalConstraint(name="no_overshoot_constraint"  ,\
        #                            exp ="lpSum({0}_start_up[t] for \
        #                             t in range({1}-{2},{1}, 1) ) == 0".format(self.name,time.LEN,len(power_values)),\
        #                                 description= "ensures that the profile does \
        #                                     not overshoot the horizon", parent = self)
        self.set_one_Startup = TechnicalConstraint(name="onestartup_constraint"  ,\
                                   exp ="lpSum({0}_start_up[t] for \
                                    t in range(0, {1}-{2}, 1) ) == 1".format(self.name,time.LEN,len(power_values)),\
                                        description= "ensures that the profile start \
                                            only once the horizon", parent = self)

        self.power_values = Quantity(name='power_values', opt=False,
                                     value=power_values, parent=self)
        
        for t in time.I:
            cst_name = 'def_{}_p'.format(t)
            rhs_terms = []
            for i, tau in enumerate(S):
                k = t-tau
                if 0 <= k and k < len(power_values):
                    rhs_terms.append(f"{self.name}_power_values[{k}] * {self.name}_start_up[{i}]")
            rhs_str = " + ".join(rhs_terms) if rhs_terms else "0"
            exp = f"{self.name}_p[{t}] == ({rhs_str})"
            cst = TechnicalConstraint(name=cst_name,exp=exp,parent= self)
            setattr(self, cst_name, cst)

class ShiftableConsumptionUnit(ShiftableEnergyUnit, VariableConsumptionUnit):
    """
    **Description**

        Consumption unit with shiftable consumption profile.

    **Attributes**

        * power_values : consumption profile to shift (kW)
        * mandatory : indicates if the consumption is mandatory (True) or not
        (False)
        * starting_cost : cost of the starting of the consumption
        * operating_cost : cost of the operation (€/kW)
        * energy_type : type of energy ('Electrical', 'Heat', ...)
        

    """
    def __init__(self, time, name: str, power_values, mandatory=True,
                co2_out=None, starting_cost=None, operating_cost=None, binary=False,
                energy_type=None, verbose=True, ):
        VariableConsumptionUnit.__init__(self, time=time, name=name, binary=binary,
                                         verbose=verbose)
        
        ShiftableEnergyUnit.__init__(self, time, name=name,
                                     flow_direction='in',
                                     power_values=power_values,
                                     mandatory=mandatory, co2_out=co2_out,
                                     starting_cost=starting_cost,
                                     operating_cost=operating_cost, binary=binary,
                                     energy_type=energy_type,
                                     verbose=False,
                                     )
        



class PhotovoltaicUnit(SeveralProductionUnit):
    """
    PV production unit with a fixed generation profile scaled by capacity.
    Inherits behavior from SeveralProductionUnit.
    """
    def __init__(self, time, name, profile, 
                 co2_out=None, rr_energy=True,energy_type=None,
                 acq_cost_per_kw=None, co2_cost_per_kw=None,
                 p_min=0, p_max=1e5, nb_unit_min=0, nb_unit_max=None,imaginary = True,
                 verbose=True, no_warn=True):
        # Initialize base class with the PV profile as fixed_prod
        super().__init__(
            time=time, name=name, fixed_prod=profile,
            p_min=p_min, p_max=p_max,imaginary = imaginary,
            nb_unit_min=nb_unit_min, nb_unit_max=nb_unit_max,
            co2_out=co2_out, particle_emission=None,
            starting_cost=None, operating_cost=None,
            max_ramp_up=None, max_ramp_down=None, energy_type=energy_type,
            rr_energy=rr_energy,
            verbose=verbose, no_warn=no_warn
        )
        # Save per-capacity costs (could be None if not used)
        self.acquisition_cost = acq_cost_per_kw
        self.co2_capex = co2_cost_per_kw

    def minimize_acquisition_cost(self, weight=1, pareto=False):
        """
        Minimize capital cost = acquisition_cost * capacity.
        Creates an objective: weight * (acquisition_cost * nb_unit).
        """
        if self.acquisition_cost is None:
            raise ValueError("No acquisition_cost defined for unit {}".format(self.name))
        # Define objective expression for capacity cost
        expr = f"{self.name}_nb_unit * {self.acquisition_cost}"
        self.minimize_acq_cost = Objective(
            name='min_acq_cost',
            exp=expr,
            weight=weight,
            pareto=pareto,
            parent=self
        )

    def minimize_co2capex(self, weight=1, pareto=False):
        """
        Minimize CO2 cost = co2_capex * capacity.
        Creates an objective: weight * (co2_capex * nb_unit).
        """
        if self.co2_capex is None:
            raise ValueError("No co2_capex defined for unit {}".format(self.name))
        expr = f"{self.name}_nb_unit * {self.co2_capex}"
        self.minimize_co2capex = Objective(
            name='min_co2capex',
            exp=expr,
            weight=weight,
            pareto=pareto,
            parent=self
        )


class ElectricalToThermalConversionUnit(ConversionUnit):
    """
    **Description**
        DEPRECATED: please now use SingleConversionUnit with relevant energy
        types

        Electrical to thermal Conversion unit with an electricity consumption
        and a thermal production

    **Attributes**

     * thermal_production_unit : thermal production unit (thermal output)
     * elec_consumption_unit : electricity consumption unit (electrical
       input)
     * conversion : Definition Dynamic Constraint linking the electrical
     input to
       the thermal output through the electrical to thermal ratio

    """
    print(
        "DEPRECATED: please now use SingleConversionUnit with relevant energy"
        "types")

    def __init__(self, time, name, pmin_in_elec=0, pmax_in_elec=1e+5,
                 p_in_elec=None, pmin_out_therm=0, pmax_out_therm=1e+5, shiftable = False, binary_elec = False,binary_therm = False,
                 p_out_therm=None, elec_to_therm_ratio=1,e_in_max = None,
                 e_in_min= None, verbose=True):

        """
        :param time: TimeUnit describing the studied time period
        :param name: name of the electrical to thermal conversion unit
        :param pmin_in_elec: minimal incoming electrical power
        :param pmax_in_elec: maximal incoming electrical power
        :param p_in_elec: power input for the electrical consumption unit
        :param pmin_out_therm: minimal power output (thermal)
        :param pmax_out_therm: maximal power output (thermal)
        :param p_out_therm: power output (thermal)
        :param elec_to_therm_ratio: electricity to thermal ratio <=1
        """

        if p_out_therm is None:
            self.thermal_production_unit = VariableProductionUnit(
                time, name + '_therm_prod', energy_type='Thermal', binary=binary_therm,
                p_min=pmin_out_therm, p_max=pmax_out_therm,
                verbose=verbose)
        else:
            self.thermal_production_unit = FixedProductionUnit(
                time, name + '_therm_prod', energy_type='Thermal',
                p=p_out_therm, verbose=verbose)

        if p_in_elec is None and shiftable == False :
            self.elec_consumption_unit = VariableConsumptionUnit(
                time, name + '_elec_cons', p_min=pmin_in_elec,
                p_max=pmax_in_elec, energy_type='Electrical', e_max=e_in_max,
                e_min = e_in_min , verbose=verbose)
            
        elif p_in_elec is not None and shiftable :
            self.elec_consumption_unit = ShiftableConsumptionUnit(
                time = time, name = name + '_elec_cons', power_values=p_in_elec, binary=binary_elec,
                energy_type='Electrical', mandatory=True , verbose=verbose)
        else:
            self.elec_consumption_unit = FixedConsumptionUnit(
                time, name + '_elec_cons', p=p_in_elec,
                energy_type='Electrical', verbose=verbose)

        ConversionUnit.__init__(self, time, name,
                                prod_units=[self.thermal_production_unit],
                                cons_units=[self.elec_consumption_unit])

        if isinstance(elec_to_therm_ratio, (int, float)):  # e2h_ratio is a
            # mean value
            if elec_to_therm_ratio <= 1:
                self.conversion = DefinitionDynamicConstraint(
                    exp_t='{0}_p[t] == {1} * {2}_p[t]'.format(
                        self.thermal_production_unit.name,
                        elec_to_therm_ratio,
                        self.elec_consumption_unit.name),
                    t_range='for t in time.I', name='conversion', parent=self)
            else:
                raise ValueError('The elec_to_therm_ratio should be lower '
                                 'than 1 (therm_production<elec_consumption)')

        elif isinstance(elec_to_therm_ratio, list):  # e2h_ratio is a list of
            # values
            if len(elec_to_therm_ratio) == self.time.LEN:  # it must have the
                #  right size, i.e. the TimeUnit length.
                if all(e <= 1 for e in elec_to_therm_ratio):
                    self.conversion = DefinitionDynamicConstraint(
                        exp_t='{0}_p[t] == {1}[t] * {2}_p[t]'.format(
                            self.thermal_production_unit.name,
                            elec_to_therm_ratio,
                            self.elec_consumption_unit.name),
                        t_range='for t in time.I', name='conversion',
                        parent=self)
                else:
                    raise ValueError(
                        'The elec_to_therm_ratio values should be '
                        'lower than 1 (therm_production<elec_'
                        'consumption)')
            else:
                raise IndexError('The length of the elec_to_therm_ratio '
                                 'vector should be of the same length as the '
                                 'TimeUnit of the studied period')

        elif isinstance(elec_to_therm_ratio, dict):  # e2h_ratio is a dict of
            # values
            if len(elec_to_therm_ratio) == self.time.LEN:
                if all(e <= 1 for e in elec_to_therm_ratio.values()):
                    self.conversion = DefinitionDynamicConstraint(
                        exp_t='{0}_p[t] == {1}[t] * {2}_p[t]'.format(
                            self.thermal_production_unit.name,
                            elec_to_therm_ratio,
                            self.elec_consumption_unit.name),
                        t_range='for t in time.I', name='conversion',
                        parent=self)
                else:
                    raise ValueError(
                        'The elec_to_therm_ratio values should be '
                        'lower than 1 (therm_production<elec_'
                        'consumption)')
            else:
                raise IndexError('The length of the elec_to_therm_ratio '
                                 'dictionary should be of the same length as '
                                 'the TimeUnit of the studied period')
        else:
            raise TypeError(
                "Electricity to thermal ratio should be a mean value or a "
                "vector (list or dict) on the whole time period !")
from res_opt_core.core.components.base import (
    Asset,
    SocManagementInterface,
    ReserveProviderInterface,
    ReserveRequest,
    ReserveRequestType,
    ComponentInputs,
)
from res_opt_core.core.components.input_annotations import VectorOfReals, DefaultToMinusOf, DefaultTo
from res_opt_core.core.utils.time_resolution import ModelTimeResolution
from res_opt_core.core.components.battery import Battery, BatteryInputs
from typing import Annotated, Any, Callable, Type
import pyomo.environ as pyo
from pyomo.core.expr import RelationalExpression
import numpy as np
from numpy.typing import NDArray



class EVInputs(ComponentInputs):
    """EV inputs

    Attributes:
        name: Name of the EV.
        energy_capacity: Energy capacity of the battery, in unit of energy, e.g., MWh.
        power_nominal: Nominal power of the battery, in unit of power, e.g., MW.
        power_max: Maximum power of the battery, in units of power, e.g., MW. If None, defaults to power_nominal.
        power_min: Minimum power of the battery, in units of power, e.g., MW. If None, defaults to -power_max.
        eta: One-way efficiency of the battery. A value between 0 and 1.
        penalty_energy_discharge: Penalty for discharging energy, in units of currency per unit of energy,
            e.g., EUR/MWh.
        soc_initial: Initial state of charge of the battery, between 0 and 1.
        soc_final: Final state of charge of the battery, between 0 and 1.
        soc_min: Minimum state of charge of the battery, between 0 and 1.
        soc_max: Maximum state of charge of the battery, between 0 and 1.
        max_cycles_per_year: Maximum number of cycles per year.

        availability_ev: Binary vector representing if the ev is connected.
        soc_target: always same target?
    """
    name: str = "ev"
    energy_in_slot_start: Annotated[float | list[float], VectorOfReals("num_steps")] = 0.0
    energy_out_slot_end: Annotated[float | list[float], VectorOfReals("num_steps")] = 0.0
    energy_capacity: Annotated[float | list[float], VectorOfReals("num_steps")] = 10.0
    power_nominal: float = 10.0
    power_max: Annotated[
        (float | list[float]) | None,
        VectorOfReals("num_steps"),
        DefaultTo("power_nominal"),
    ] = None
    power_min: Annotated[
        (float | list[float]) | None,
        VectorOfReals("num_steps"),
        DefaultToMinusOf("power_max"),
    ] = None
    eta: float = 0.933
    penalty_energy_discharge: float = 0.0
    soc_initial: float | None = 0.5
    soc_final: float | None = 0.5
    soc_default: Annotated[float | list[float], VectorOfReals("num_steps")] = 0.5
    soc_min: Annotated[float | list[float], VectorOfReals("num_steps")] = 0.0
    soc_max: Annotated[float | list[float], VectorOfReals("num_steps")] = 1.0
    soc_min_soft: Annotated[float | list[float], VectorOfReals("num_steps")] = 0.0
    soc_max_soft: Annotated[float | list[float], VectorOfReals("num_steps")] = 1.0
    soc_min_soft_penalty: Annotated[float | list[float], VectorOfReals("num_steps")] = 0.0
    soc_max_soft_penalty: Annotated[float | list[float], VectorOfReals("num_steps")] = 0.0
    cost_min_energy_violation: float | None = None
    cost_max_energy_violation: float | None = None
    cost_min_power_violation: float | None = None
    cost_max_power_violation: float | None = None
    max_cycles_per_year: float | None = None



class EV(Battery):
    """This is the EV-Class created for generating the new model

    Args:
        name: The name of the component.
        **kwargs: Additional attributes passed on to inputs class.

    Attributes:
        inputs: The inputs to the component.
        inputs_class: The class of the inputs.
        name: The name of the component.
        horizon: The time steps in the optimization horizon.
        dt: The length of each time step in hours.
    """

    inputs: EVInputs
    inputs_class: Type[EVInputs] = EVInputs

    def __init__(
        self,
        *,
        name: str | None = None,
        energy_capacity: float | list[float] = 10.0,
        power_nominal: float = 10.0,
        power_max: float | list[float] | None = None,
        power_min: float | list[float] | None = None,
        eta: float = 0.933,
        penalty_energy_discharge: float = 0.0,
        soc_initial: float | None = 0.5,
        soc_final: float | None = 0.5,
        soc_default: float | list[float] = 0.5,
        soc_min: float | list[float] = 0.0,
        soc_max: float | list[float] = 1.0,
        soc_min_soft: float | list[float] = 0.0,
        soc_max_soft: float | list[float] = 1.0,
        soc_min_soft_penalty: float = 0.0,
        soc_max_soft_penalty: float = 0.0,
        cost_min_energy_violation: float | None = None,
        cost_max_energy_violation: float | None = None,
        cost_min_power_violation: float | None = None,
        cost_max_power_violation: float | None = None,
        max_cycles_per_year: float | None = None,
        energy_in_slot_start: float | list[float] = 0.0,
        energy_out_slot_end: float | list[float] = 0.0, 
        **kwargs: Any,
    ) -> None:
        """Initialize the battery component

        Args:
            name: Name of the battery.
            energy_capacity: Energy capacity of the battery, in unit of energy, e.g., MWh.
            power_nominal: Nominal power of the battery, in unit of power, e.g., MW.
            power_max: Maximum power of the battery, in units of power, e.g., MW. If None, defaults to power_nominal.
            power_min: Minimum power of the battery, in units of power, e.g., MW. If None, defaults to -power_max.
            eta: One-way efficiency of the battery. A value between 0 and 1.
            penalty_energy_discharge: Penalty for discharging energy, in units of currency per unit of energy,
                e.g., EUR/MWh.
            soc_initial: Initial state of charge of the battery, between 0 and 1.
            soc_final: Final state of charge of the battery, between 0 and 1.
            soc_default: Default state of charge of the battery, between 0 and 1.
            soc_min: Minimum state of charge of the battery, between 0 and 1.
            soc_max: Maximum state of charge of the battery, between 0 and 1.
            soc_min_soft: Soft minimum state of charge. Only affects deterministic power flows, not reserves,
                between 0 and 1.
            soc_max_soft: Soft maximum state of charge. Only affects deterministic power flows, not reserves,
                between 0 and 1.
            soc_min_soft_penalty: Penalty for violating the soft minimum state of charge, in units of currency per
                unit of energy, e.g., EUR/MWh.
            soc_max_soft_penalty: Penalty for violating the soft maximum state of charge, in units of currency per
                unit of energy, e.g., EUR/MWh.
            cost_min_energy_violation: Cost for violating the minimum energy, in units of currency per unit of energy,
                e.g., EUR/MWh. Violation not allowed if set to None.
            cost_max_energy_violation: Cost for violating the maximum energy, in units of currency per unit of energy,
                e.g., EUR/MWh.  Violation not allowed if set to None.
            cost_min_power_violation: Cost for violating the minimum power, in units of currency per unit of energy,
                e.g., EUR/MWh.  Violation not allowed if set to None.
            cost_max_power_violation: Cost for violating the maximum power, in units of currency per unit of energy,
                e.g., EUR/MWh.  Violation not allowed if set to None.
            max_cycles_per_year: Maximum number of cycles per year.
        """

        super().__init__(
            name=name,
            energy_capacity=energy_capacity,
            power_nominal=power_nominal,
            power_max=power_max,
            power_min=power_min,
            eta=eta,
            penalty_energy_discharge=penalty_energy_discharge,
            soc_initial=soc_initial,
            soc_final=soc_final,
            soc_default=soc_default,
            soc_min=soc_min,
            soc_max=soc_max,
            soc_min_soft=soc_min_soft,
            soc_max_soft=soc_max_soft,
            soc_min_soft_penalty=soc_min_soft_penalty,
            soc_max_soft_penalty=soc_max_soft_penalty,
            cost_min_energy_violation=cost_min_energy_violation,
            cost_max_energy_violation=cost_max_energy_violation,
            cost_min_power_violation=cost_min_power_violation,
            cost_max_power_violation=cost_max_power_violation,
            max_cycles_per_year=max_cycles_per_year,
            energy_in_slot_start=energy_in_slot_start,
            energy_out_slot_end=energy_out_slot_end,
            **kwargs,
        )

    def init_variables(self) -> None:
        super().init_variables()

        self.var_soc_slot_start = pyo.Var(self.horizon, within=pyo.Reals)
        self.var_soc_slot_end = pyo.Var(self.horizon, within=pyo.Reals)

    def _get_constraints_soc(self) -> list[RelationalExpression]:
        constraints = []

        # inital soc
        if self.inputs.soc_initial is not None:
            constraints += [self.var_soc_slot_start[self.horizon[0]] == self.inputs.soc_initial]

        # final soc
        if self.inputs.soc_final is not None:
            constraints += [self.var_soc_slot_end[self.horizon[-1]] == self.inputs.soc_final]

        # changes due to charing and discharging (real energy flow)
        if self.inputs.cost_max_energy_violation is None:
            # if no cost for violating max energy, then no violation allowed
            constraints += [self.var_max_energy_violation[k] == 0.0 for k in self.horizon]
        if self.inputs.cost_min_energy_violation is None:
            # if no cost for violating min energy, then no violation allowed
            constraints += [self.var_min_energy_violation[k] == 0.0 for k in self.horizon]

        for k in self.horizon:
            if self.inputs.energy_capacity[k] > 0:
                constraints.append(
                    self.var_soc_slot_start[k]
                    + (
                        (self._abs_delta_dc_energy_charge(k) - self._abs_delta_dc_energy_discharge(k) - self.inputs.energy_out_slot_end[k])
                        / self.inputs.energy_capacity[k]
                    )
                    == self.var_soc_slot_end[k]
                )
            else:
                # SoC stays constant
                constraints.append(self.var_soc_slot_start[k] == self.var_soc_slot_end[k])

                # No power flow allowed
                constraints.append(self.var_power_charge[k] == 0.0)
                constraints.append(self.var_power_discharge[k] == 0.0)

        # availability changes
        constraints += [
            self.var_soc_slot_start[k + 1]
            == (self.var_soc_slot_end[k]*self.inputs.energy_capacity[k] + self.inputs.energy_in_slot_start[k+1]) / self.inputs.energy_capacity[k+1]
            if self.inputs.energy_capacity[k+1] > 0 
            else self.var_soc_slot_start[k + 1] == self.var_soc_slot_end[k]
            for k in self.horizon[0:-1]
        ]

        for k in self.horizon:
            if self.inputs.energy_capacity[k] > 0.0:
                # min soc constraints
                constraints.append(
                    self.var_soc_slot_start[k] + (self.var_min_energy_violation[k] / self.inputs.energy_capacity[k])
                    >= self.inputs.soc_min[k]
                    + (self.var_energy_reserve_up_slot_start[k] / self.inputs.energy_capacity[k])
                )
                constraints.append(
                    self.var_soc_slot_end[k] + (self.var_min_energy_violation[k] / self.inputs.energy_capacity[k])
                    >= self.inputs.soc_min[k]
                    + (self.var_energy_reserve_up_slot_end[k] / self.inputs.energy_capacity[k])
                )

                # max soc constraints
                constraints.append(
                    self.var_soc_slot_start[k] - (self.var_max_energy_violation[k] / self.inputs.energy_capacity[k])
                    <= self.inputs.soc_max[k]
                    - (self.var_energy_reserve_down_slot_start[k] / self.inputs.energy_capacity[k])
                )
                constraints.append(
                    self.var_soc_slot_end[k] - (self.var_max_energy_violation[k] / self.inputs.energy_capacity[k])
                    <= self.inputs.soc_max[k]
                    - (self.var_energy_reserve_down_slot_end[k] / self.inputs.energy_capacity[k])
                )


                # soft min soc constraints
                constraints.append(
                    self.var_soc_slot_start[k] + (self.var_soc_min_soft_violation[k] / self.inputs.energy_capacity[k])
                    >= self.inputs.soc_min_soft[k]
                )
                constraints.append(
                    self.var_soc_slot_end[k] + (self.var_soc_min_soft_violation[k] / self.inputs.energy_capacity[k])
                    >= self.inputs.soc_min_soft[k]
                )

                # soft max soc constraints
                constraints.append(
                    self.var_soc_slot_start[k] - (self.var_soc_max_soft_violation[k] / self.inputs.energy_capacity[k])
                    <= self.inputs.soc_max_soft[k]
                )
                constraints.append(
                    self.var_soc_slot_end[k] - (self.var_soc_max_soft_violation[k] / self.inputs.energy_capacity[k])
                    <= self.inputs.soc_max_soft[k]
                )
            else:
                # violations are zero
                constraints.append(self.var_max_energy_violation[k] == 0.0)
                constraints.append(self.var_min_energy_violation[k] == 0.0)
                constraints.append(self.var_soc_min_soft_violation[k] >= 0.0)
                constraints.append(self.var_soc_max_soft_violation[k] >= 0.0)

                # no reserves permitted
                constraints.append(self.var_energy_reserve_up_slot_start[k] == 0.0)
                constraints.append(self.var_energy_reserve_up_slot_end[k] == 0.0)
                constraints.append(self.var_energy_reserve_down_slot_start[k] == 0.0)
                constraints.append(self.var_energy_reserve_down_slot_end[k] == 0.0)


        return constraints


    def _get_constraints_soc__(self) -> list[RelationalExpression]:
        """Overrides Battery._get_constraints_soc() with EV-specific SoC dynamics.

        Two differences from Battery:
        1. SoC update within a slot subtracts energy_out_slot_end (departing EVs).
        2. SoC transition between slots adds energy_in_slot_start (arriving EVs).

        All other SoC constraints (initial, final, min/max, soft, reserves,
        violations) are identical to Battery and are obtained via super().

        Returns:
            List of SoC constraints.
        """
        # get all standard Battery SoC constraints as base
        # (initial, final, min/max, soft limits, violations, reserves)
        constraints = super()._get_constraints_soc()

        # ── override intra-slot SoC update to include energy_out_slot_end ──────
        # Battery:  soc_start + (charge - discharge) / capacity == soc_end
        # EV:       soc_start + (charge - discharge - energy_out) / capacity == soc_end

        for k in self.horizon:
            if self.inputs.energy_capacity[k] > 0:
                constraints.append(
                    self.var_soc_slot_start[k]
                    + (
                        (self._abs_delta_dc_energy_charge(k) - self._abs_delta_dc_energy_discharge(k) - self.inputs.energy_out_slot_end[k])
                        / self.inputs.energy_capacity[k]
                    )
                    == self.var_soc_slot_end[k]
                )
            else:
                # SoC stays constant
                constraints.append(self.var_soc_slot_start[k] == self.var_soc_slot_end[k])

                # No power flow allowed
                constraints.append(self.var_power_charge[k] == 0.0)
                constraints.append(self.var_power_discharge[k] == 0.0)

        # ── override inter-slot SoC transition to include energy_in_slot_start ─
        # Battery:  soc_start[k+1] == soc_end[k] * (cap[k] / cap[k+1]) + soc_default * (...)
        # EV:       soc_start[k+1] == (soc_end[k] * cap[k] + energy_in[k+1]) / cap[k+1]
        constraints += [
            self.var_soc_slot_start[k + 1]
            == (self.var_soc_slot_end[k]*self.inputs.energy_capacity[k] + self.inputs.energy_in_slot_start[k+1]) / self.inputs.energy_capacity[k+1]
            if self.inputs.energy_capacity[k+1] > 0 
            else self.var_soc_slot_start[k + 1] == self.var_soc_slot_end[k]
            for k in self.horizon[0:-1]
        ]

        return constraints

class EV_aggregated(Battery):
    """The crux lays between the indivudal constraints such as that when one car is full the power can not be utilized to fill 
    the SoC of another car. The current class has one SoC and the power inputs aggregated by the entire fleet. Thus, 
    unrealistic results can occur when already full cars are utilized to charge another car. 
    
    How to solve? 
    - model all individually as batteries and then aggregate the fleet as new component for SDL
    - dynamically change the inputs for the next timestep based on the outputs of the previous timestep
    (is this possible? need to make new opt per i.e. every 4hrs -> high pre- and post-processing)
    - is it possible to have power as an input but also dependent on the indivudal cars? (I think not)
    """

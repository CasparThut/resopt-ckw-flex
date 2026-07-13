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


class EV_old(Asset, SocManagementInterface, ReserveProviderInterface):
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
    inputs_class: type[EVInputs] = EVInputs

    def __init__(
        self,
        *,
        name: str | None = None,
        energy_capacity: float | list[float] = 10.0,
        energy_in_slot_start: float | list[float] = 0.0,
        energy_out_slot_end: float | list[float] = 0.0,
        power_nominal: float = 10.0,
        power_max: float | list[float] = None,
        power_min: (float | list[float]) | None = None,
        eta: float = 0.933,
        penalty_energy_discharge: float = 0.0,
        soc_initial: float | None = 0.5,
        soc_final: float | None = 0.5,
        soc_min: float | list[float] = 0.0,
        soc_max: float | list[float] = 1.0,
        max_cycles_per_year: float | None = None,
        **kwargs,
    ):
        """Initialize the EV component

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
            soc_min: Minimum state of charge of the battery, between 0 and 1.
            soc_max: Maximum state of charge of the battery, between 0 and 1.
            max_cycles_per_year: Maximum number of cycles per year.
        """

        super().__init__(
            name=name,
            energy_capacity=energy_capacity,
            energy_in_slot_start=energy_in_slot_start,
            energy_out_slot_end=energy_out_slot_end,
            power_nominal=power_nominal,
            power_max=power_max,
            power_min=power_min,
            eta=eta,
            penalty_energy_discharge=penalty_energy_discharge,
            soc_initial=soc_initial,
            soc_final=soc_final,
            soc_min=soc_min,
            soc_max=soc_max,
            max_cycles_per_year=max_cycles_per_year,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return self.inputs.name

    # Why do we need this?
    @property
    def horizon_duration_in_minutes(self):
        return self.slot_length.in_minutes() * len(self.horizon)


    # TODO: Not yet implemented
    def prepare_for_model_building(self, context: dict) -> None:
        """Prepare the component for model building.

        This method is called before the MILP model is built and is used to set the number of time steps
        in the optimization horizon, and the length of each time step (slot length). Additionally, the inputs
        are finalized.

        Args:
            context: A dictionary containing context information.
        """
        super().prepare_for_model_building(context)

        self.slot_length: ModelTimeResolution = context["slot_length"]

        # power reserves inherited from Asset class
        self._reserve_booking_energy_up_slot_start = [[0.0] for _ in self.horizon]
        self._reserve_booking_energy_down_slot_start = [[0.0] for _ in self.horizon]
        self._reserve_booking_energy_up_slot_end = [[0.0] for _ in self.horizon]
        self._reserve_booking_energy_down_slot_end = [[0.0] for _ in self.horizon]

    def init_variables(self) -> None:
        super().init_variables()

        self.var_soc_slot_start = pyo.Var(self.horizon, within=pyo.Reals)
        self.var_soc_slot_end = pyo.Var(self.horizon, within=pyo.Reals)

        self.var_min_energy_violation = pyo.Var(self.horizon, within=pyo.NonNegativeReals)
        self.var_max_energy_violation = pyo.Var(self.horizon, within=pyo.NonNegativeReals)

        # var_power is inherited from NewAssetComponent
        self.var_power_charge = pyo.Var(self.horizon, within=pyo.NonNegativeReals)
        self.var_power_discharge = pyo.Var(self.horizon, within=pyo.NonNegativeReals)
        self.var_is_charging = pyo.Var(self.horizon, within=pyo.Binary)
        self.var_is_discharging = pyo.Var(self.horizon, within=pyo.Binary)

        self.var_power_max_violation = pyo.Var(self.horizon, within=pyo.NonNegativeReals)
        self.var_power_min_violation = pyo.Var(self.horizon, within=pyo.NonNegativeReals)

        # var_power_reserve_up and var_power_reserve_down are inherited from NewAssetComponent
        self.var_energy_reserve_up_slot_start = pyo.Var(self.horizon, within=pyo.NonNegativeReals)
        self.var_energy_reserve_down_slot_start = pyo.Var(self.horizon, within=pyo.NonNegativeReals)
        self.var_energy_reserve_up_slot_end = pyo.Var(self.horizon, within=pyo.NonNegativeReals)
        self.var_energy_reserve_down_slot_end = pyo.Var(self.horizon, within=pyo.NonNegativeReals)

        # SoC soft limits
        self.var_soc_min_soft_violation = pyo.Var(self.horizon, within=pyo.NonNegativeReals)
        self.var_soc_max_soft_violation = pyo.Var(self.horizon, within=pyo.NonNegativeReals)


    def _abs_delta_dc_energy_charge(self, k: int) -> pyo.expr.ExpressionBase:
        """Delta DC energy charged in time slot k."""
        return self.var_power_charge[k] * self.inputs.eta * self.dt[k]

    def _abs_delta_dc_energy_discharge(self, k: int) -> pyo.expr.ExpressionBase:
        """Delta DC energy discharged in time slot k."""
        return self.var_power_discharge[k] / self.inputs.eta * self.dt[k]



    # TODO: implement reserve markets booking for EV
    # difference to battery?!
    @property
    def reserve_requests(self) -> list[ReserveRequest]:
        return self._reserve_requests

    def _process_reserve_request(self, reserve_request: ReserveRequest):
        reserve_request = reserve_request.get_eta_adjusted(self.inputs.eta)
        self._reserve_requests.append(reserve_request)

        for k in self.horizon:
            self._reserve_booking_power_up[k].append(reserve_request.power_up[k])
            self._reserve_booking_power_down[k].append(reserve_request.power_down[k])
            self._reserve_booking_energy_up_slot_start[k].append(reserve_request.energy_up_slot_start[k])
            self._reserve_booking_energy_down_slot_start[k].append(reserve_request.energy_down_slot_start[k])
            self._reserve_booking_energy_up_slot_end[k].append(reserve_request.energy_up_slot_end[k])
            self._reserve_booking_energy_down_slot_end[k].append(reserve_request.energy_down_slot_end[k])

    def submit_capacity_reserve_request(self, reserve_request: ReserveRequest):
        if reserve_request.request_type != ReserveRequestType.CAPACITY:
            raise ValueError(
                f"Reserve request type {reserve_request.request_type} " f"does not match {ReserveRequestType.CAPACITY}"
            )

        self._process_reserve_request(reserve_request)

    def submit_soc_mgmt_reserve_request(self, reserve_request: ReserveRequest):
        if reserve_request.request_type != ReserveRequestType.SOC_MANAGEMENT:
            raise ValueError(
                f"Reserve request type {reserve_request.request_type} "
                f"does not match {ReserveRequestType.SOC_MANAGEMENT}"
            )

        self._process_reserve_request(reserve_request)



    def _get_constraints_power(self) -> list[RelationalExpression]:
        constraints = []

        # min power constraints
        constraints += [
            self.var_power[k] + self.var_power_min_violation[k]
            >= (self.inputs.power_min[k] + self.var_power_reserve_down[k]) 
            for k in self.horizon
        ]

        # max power constraints
        constraints += [
            self.var_power[k] - self.var_power_max_violation[k]
            <= (self.inputs.power_max[k] - self.var_power_reserve_up[k])
        for k in self.horizon
        ]

        if self.inputs.cost_max_power_violation is None:
            # if no cost for violating max power, then no violation allowed
            constraints += [self.var_power_max_violation[k] == 0.0 for k in self.horizon]
        if self.inputs.cost_min_power_violation is None:
            # if no cost for violating min power, then no violation allowed
            constraints += [self.var_power_min_violation[k] == 0.0 for k in self.horizon]

        # only charing or discharging
        constraints += [
            self.var_power[k] == self.var_power_discharge[k] - self.var_power_charge[k] for k in self.horizon
        ]

        constraints += [
            self.var_power_charge[k] <= self.inputs.power_nominal * self.var_is_charging[k] for k in self.horizon
        ]

        constraints += [
            self.var_power_discharge[k] <= self.inputs.power_nominal * self.var_is_discharging[k] for k in self.horizon
        ]

        constraints += [self.var_is_charging[k] + self.var_is_discharging[k] <= 1 for k in self.horizon]


        # TODO: power is zero when var_availability is zero
        # Create a new constraint or append to existing?
        # THIS IS NOT NEEDED SINCE WE DEFINE THE AVAILABLE POWER THROUGH THE INPUTS power_min, power_max per timestep
        # constraints += [self.var_power_charge[k] == 0.0 for k in self.horizon if self.ev_availability[k] == 0] #in need of [k]?
        # constraints += [self.var_power_discharge[k] == 0.0 for k in self.horizon if self.ev_availability[k] == 0]
        return constraints


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


                # TODO: WHY SOFT CONSTRAINTS?!
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

        # NEW:
        # at the last time steps of a session the soc target has to be reached
        # every session end is when ev_availability goes from 1 to 0
        # for k in self.horizon[:-1]:
        #     if self.inputs.power_max[k] != 0 and self.inputs.power_max[k+1] == 0:
        #         constraints.append(
        #             self.var_soc_slot_end[k] == self.inputs.soc_target[k] if isinstance(self.inputs.soc_target, list) else self.inputs.soc_target
        #         )
                
        # # Handle the very last timestep of the horizon, if EV is available it soc target has to be at last step
        # if self.inputs.power_max[self.horizon[-1]] != 0:
        #     last_target = self.inputs.soc_target[-1] if isinstance(self.inputs.soc_target, list) else self.inputs.soc_target
        #     constraints.append(self.var_soc_slot_end[self.horizon[-1]] == last_target)


        return constraints


    def _get_constraints_soc_additional_one_period_cuts(self) -> list[RelationalExpression]:
        """Additional one-period cuts for SoC constraints to tighten the LP relaxation.

        The extra constraints are mostly useful in energy market optimizations with negative prices where the battery
        would like to fully charge and discharge in the same period.

        The constraints were derived by Reinhard Bauer. The extra cuts are not changing the feasible region of the
        original MILP, but tighten the LP relaxation. In case, there are numerical issues when solving the MILP, these
        constraints can be deactivated/removed.
        """
        constraints = []

        for k in self.horizon:
            if self.inputs.energy_capacity[k] > 0.0:
                # min soc constraints
                constraints.append(
                    self.var_soc_slot_start[k]
                    - (self._abs_delta_dc_energy_discharge(k) / self.inputs.energy_capacity[k])
                    >= self.inputs.soc_min[k] - (self.var_min_energy_violation[k] / self.inputs.energy_capacity[k])
                )

                # max soc constraints
                constraints.append(
                    self.var_soc_slot_start[k] + (self._abs_delta_dc_energy_charge(k) / self.inputs.energy_capacity[k])
                    <= self.inputs.soc_max[k] + (self.var_max_energy_violation[k] / self.inputs.energy_capacity[k])
                )

        return constraints

    def _get_constraints_energy_reserves(self) -> list[RelationalExpression]:
        constraints = []

        constraints += [
            self.var_energy_reserve_up_slot_start[k] == sum(self._reserve_booking_energy_up_slot_start[k])
            for k in self.horizon
        ]

        constraints += [
            self.var_energy_reserve_up_slot_end[k] == sum(self._reserve_booking_energy_up_slot_end[k])
            for k in self.horizon
        ]

        constraints += [
            self.var_energy_reserve_down_slot_start[k] == sum(self._reserve_booking_energy_down_slot_start[k])
            for k in self.horizon
        ]

        constraints += [
            self.var_energy_reserve_down_slot_end[k] == sum(self._reserve_booking_energy_down_slot_end[k])
            for k in self.horizon
        ]

        return constraints

    def _get_constraints_cycle_ageing(self) -> list[RelationalExpression]:
        if self.inputs.max_cycles_per_year is None:
            return []

        max_energy_capacity = max(self.inputs.energy_capacity)

        if max_energy_capacity == 0:
            return []

        return [
            pyo.quicksum(
                (+self._abs_delta_dc_energy_charge(k) + self._abs_delta_dc_energy_discharge(k))
                / (2 * max_energy_capacity)
                for k in self.horizon
            )
            <= self.inputs.max_cycles_per_year * self.horizon_duration_in_minutes / (365 * 24 * 60)
        ]


    def get_constraints(self) -> list[RelationalExpression]:
        constraints = super().get_constraints()
        constraints += self._get_constraints_power()
        constraints += self._get_constraints_soc()
        constraints += self._get_constraints_soc_additional_one_period_cuts()
        constraints += self._get_constraints_energy_reserves()
        constraints += self._get_constraints_cycle_ageing()

        return constraints

    def add_objectives(self, objectives: list[float | pyo.Expression]) -> None:
        super().add_objectives(objectives)

        # add cost for violating min and max power
        if self.inputs.cost_min_power_violation is not None:
            objectives.append(
                -self.inputs.cost_min_power_violation * sum(self.var_power_min_violation[k] for k in self.horizon)
            )
        if self.inputs.cost_max_power_violation is not None:
            objectives.append(
                -self.inputs.cost_max_power_violation * sum(self.var_power_max_violation[k] for k in self.horizon)
            )

        # add cost for violating min and max energy
        if self.inputs.cost_min_energy_violation is not None:
            objectives.append(
                -self.inputs.cost_min_energy_violation * sum(self.var_min_energy_violation[k] for k in self.horizon)
            )
        if self.inputs.cost_max_energy_violation is not None:
            objectives.append(
                -self.inputs.cost_max_energy_violation * sum(self.var_max_energy_violation[k] for k in self.horizon)
            )

        # add cost for violating soft min and max soc
        objectives.append(
            -self.inputs.soc_min_soft_penalty * sum(self.var_soc_min_soft_violation[k] for k in self.horizon)
        )
        objectives.append(
            -self.inputs.soc_max_soft_penalty * sum(self.var_soc_max_soft_violation[k] for k in self.horizon)
        )

        # add battery wear penalty (relative to AC discharged energy)
        objectives.append(
            -self.inputs.penalty_energy_discharge * sum(self.var_power_discharge[k] * self.dt[k] for k in self.horizon)
        )


    # TODO: WHY ARE THESE parts not used in the battery.py module?

    def read_results(self, prefix: str = "") -> dict[str, list[float]]:
        """Return a dictionary of variable values

        Returns the values of all Pyomo variables in the component. The variable values are
        stored in a dictionary with the variable name as the key. This method can only be called
        after the optimization has been solved.

        Returns:
            A dictionary of variable values.
        """

        variable_values = {}
        for attr_name, attr_value in self.__dict__.items():
            if not isinstance(attr_value, pyo.Var):
                continue
            if isinstance(attr_value, pyo.ScalarVar):
                variable_values[f"{prefix}{self.inputs.name}__{attr_name}"] = pyo.value(attr_value, exception=False)
            else:
                variable_values[f"{prefix}{self.inputs.name}__{attr_name}"] = [
                    pyo.value(attr_value[k], exception=False) for k in self.horizon
                ]

        return variable_values

    def read_inputs(self) -> dict:
        return self.inputs.model_dump()

    def read_kpis(self) -> dict[str, Callable]:
        """Return a dictionary of KPIs.

        Subclasses should override this method to define KPIs for the component. This method can only be called
        after the optimization has been solved.

        Returns:
            A dictionary of KPIs.
        """
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


def get_pyo_var_values(var: pyo.Var) -> NDArray[np.float64]:
    return np.array(list(var.get_values().values()))

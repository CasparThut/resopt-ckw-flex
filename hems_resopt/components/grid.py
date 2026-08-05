from typing import Literal

import pyomo.environ as pyo

from res_opt_core.core.components.base import Asset
from res_opt_core.core.components.energy_markets import EnergyMarket
from res_opt_core.core.utils.time_resolution import MarketTimeUnit, TimeBlocks
from res_opt_core.valuator.components.grid import GridWithFees, GridWithFeesInputs


class GridPeakShaveInputs(GridWithFeesInputs):
    """Inputs for the GridPeakShave component.

    Attributes:
        name: Name of the component.
        power_max: Maximum power limits, in units of power, e.g., MW.
        power_min: Minimum power limits, in units of power, e.g., MW.
        fee_import: Fee on imported energy.
        fee_export: Fee on exported energy.
        ramp_step_allowed: Allowed ramp, in units of power per between model time slots, e.g., MW.
        ramp_rate_allowed: Allowed ramp rate, in units of per unit of power per time, e.g., MW/h.
            If ramp_rate_allowed is set, ramp_step_allowed must also be set.
        ramp_imbalance_price_curve: Price curve for imbalance due to ramping. Only relevant
            if ramp_rate_allowed is set.
        peak_power_price: The price of the peak cost per unit of power, e.g., MW.
        time_horizon_peak: Time horizon over which the peak power is evaluated, e.g.,
            "month_naive", "month_daylight_saving", "day_naive", "day_daylight_saving".
    """

    name: str = "grid_peak_shave"
    peak_power_price: float = 10000.0
    time_horizon_peak: MarketTimeUnit | Literal[
        "month_naive",
        "month_daylight_saving",
        "day_naive",
        "day_daylight_saving",
    ] = MarketTimeUnit.MONTH_NAIVE


class GridPeakShave(GridWithFees):
    """Grid peak shave component.

    Extends the GridWithFees component with additional peak shaving constraints
    and objectives. All standard grid energy balance, power limit, import/export
    fee, and ramping constraints are inherited from GridWithFees.

    Attributes:
        inputs: Inputs.
        inputs_class: GridPeakShaveInputs class.
        assets: List of assets.
        markets: List of energy markets.
    """

    inputs: GridPeakShaveInputs
    inputs_class = GridPeakShaveInputs

    def __init__(
        self,
        power_max: (float | list[float]),
        power_min: (float | list[float]),
        assets: list[Asset] | None = None,
        markets: list[EnergyMarket] | None = None,
        name: str | None = None,
        fee_import: (float | list[float]) | None = None,
        fee_export: (float | list[float]) | None = None,
        ramp_step_allowed: (float | list[float]) | None = None,
        ramp_rate_allowed: (float | list[float]) | None = None,
        ramp_imbalance_price_curve: float | list[float] = 0.0,
        peak_power_price: float | None = None,
        time_horizon_peak: MarketTimeUnit
        | Literal["month_naive", "month_daylight_saving", "day_naive", "day_daylight_saving"] = MarketTimeUnit.MONTH_NAIVE,
    ):
        """Initializes the GridPeakShave component.

        Args:
            assets: List of assets.
            markets: List of energy markets.
            name: Name of the component.
            power_max: Maximum power limits, in units of power, e.g., MW.
            power_min: Minimum power limits, in units of power, e.g., MW.
            fee_import: Fee on imported energy.
            fee_export: Fee on exported energy.
            ramp_step_allowed: Allowed ramp, in units of power per between model time slots, e.g., MW.
            ramp_rate_allowed: Allowed ramp rate, in units of per unit of power per time, e.g., MW/h.
                If ramp_rate_allowed is set, ramp_step_allowed must also be set.
            ramp_imbalance_price_curve: Price curve for imbalance due to ramping. Only relevant
                if ramp_rate_allowed is set.
            peak_power_price: Price of the peak power per unit of power, e.g., EUR/MW.
            time_horizon_peak: Time horizon over which the peak power is evaluated, e.g.,
                "month_naive", "month_daylight_saving", "day_naive", "day_daylight_saving".
        """
        super().__init__(
            assets=assets,
            markets=markets,
            name=name,
            power_max=power_max,
            power_min=power_min,
            fee_import=fee_import,
            fee_export=fee_export,
            ramp_step_allowed=ramp_step_allowed,
            ramp_rate_allowed=ramp_rate_allowed,
            ramp_imbalance_price_curve=ramp_imbalance_price_curve,
            # peak_power_price=peak_power_price,
            # time_horizon_peak=time_horizon_peak,
        )

        if peak_power_price is not None:
            self.inputs.peak_power_price = peak_power_price
        self.inputs.time_horizon_peak = time_horizon_peak

    def init_variables(self) -> None:
        """Initializes the peak power variables for the GridPeakShave component.

        Creates one non-negative peak power variable per horizon window.
        Peak power only occurs due to energy consumption / charging behaviour,
        hence the NonNegativeReals domain.

        Populates self.peak_power with one pyo.Var per window.
        """
        super().init_variables()

        self.var_peak_power = pyo.Var(self.horizon, within=pyo.NonPositiveReals)

    def _get_constraints_peak_shave(self) -> list:
        """Adds constraints ensuring var_peak_power[w] >= combined asset power
        at every timestep k within window w.

        Peak power only captures consumption / charging behaviour, consistent
        with the NonNegativeReals domain of var_peak_power.

        Returns:
            List of peak shaving constraints.
        """
        constraints = []


        # 1. block constraints: var_peak_power is constant within each block
        time_blocks = TimeBlocks(market_time_unit=self.inputs.time_horizon_peak)
        constraints += time_blocks.get_block_constraints(
            timestamps=self.timestamp,
            var=self.var_peak_power,
        )

        # Sum of power of assets are smaller / equal peak_power
        constraints += [
            self.var_peak_power[k] <= sum(c.power[k] for c in self.assets) # with negative sign we only penalize charging/consumption behaviour
            for k in self.horizon
        ]

        return constraints


    def get_constraints(self) -> list:
        constraints = super().get_constraints()
        constraints += self._get_constraints_peak_shave()
        return constraints

    def add_objectives(self, objectives: list[float | pyo.Expression]) -> None:
        super().add_objectives(objectives)
        
        # use TimeBlocks to identify the first timestep of each block
        time_blocks = TimeBlocks(market_time_unit=self.inputs.time_horizon_peak)
        blocks = time_blocks._group(self.timestamp)

        for block_indices in blocks.values():
            # all timesteps in the block share the same var_peak_power value
            # so we only need one representative per block (the first)
            representative_k = block_indices[0]
            objectives.append(
                self.inputs.peak_power_price * self.var_peak_power[representative_k]
            )
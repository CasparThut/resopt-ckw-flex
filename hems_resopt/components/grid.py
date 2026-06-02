from typing import Annotated, Literal
from datetime import timedelta
from enum import StrEnum
import pyomo.environ as pyo

from res_opt_core.core.components.input_annotations import VectorOfReals
from res_opt_core.core.components.energy_balance import Grid, GridInputs
from res_opt_core.core.components.base import Component, ComponentInputs, Asset, PowerInterface
from res_opt_core.core.components.energy_markets import EnergyMarket
from res_opt_core.core.utils.time_resolution import ModelTimeResolution


from datetime import timedelta
from enum import StrEnum


class ModelTimeResolution_grid(StrEnum):
    """Extended time resolution enum for grid peak shaving.

    Extends ModelTimeResolution with additional time horizons relevant for
    peak shaving: day, week, and month.

    Members:
        MINUTE: 1-minute resolution.
        MINUTES_15: 15-minute resolution.
        HOUR: Hourly resolution.
        DAY: Daily resolution.
        WEEK: Weekly resolution.
        MONTH: Monthly resolution (approximated as 30 days).
    """

    # --- inherited members (mirrored from ModelTimeResolution) ---
    MINUTE = "1min"
    MINUTES_15 = "15min"
    HOUR = "hour"

    # --- new members ---
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

    def _is_base_resolution(self) -> bool:
        """Returns True if this member exists in the base ModelTimeResolution."""
        return self in (
            ModelTimeResolution_grid.MINUTE,
            ModelTimeResolution_grid.MINUTES_15,
            ModelTimeResolution_grid.HOUR,
        )

    def in_minutes(self) -> int:
        """Returns the number of minutes corresponding to this resolution.

        Returns:
            Number of minutes.
        """
        if self._is_base_resolution():
            # Delegate to base logic by reusing the same mapping
            return ModelTimeResolution(self.value).in_minutes()

        if self == ModelTimeResolution_grid.DAY:
            return 60 * 24
        elif self == ModelTimeResolution_grid.WEEK:
            return 60 * 24 * 7
        
        # TODO: Is approximation for Month reasonable?
        # make it datetime aware? is ResOpt in general datetime aware?
        elif self == ModelTimeResolution_grid.MONTH:
            return 60 * 24 * 30  # approximated as 30 days
        else:
            raise ValueError(f"Unknown time resolution {self}")


    # TODO: Is there a need to create a contrary component, such as hours per time_slot?
    def slots_in_hour(self) -> int:
        """Returns the number of slots per hour for this resolution.

        Returns:
            Number of slots per hour.

        Raises:
            ValueError: If the resolution is coarser than an hour (day/week/month),
                        as slots_in_hour is not meaningful for those.
        """
        if self._is_base_resolution():
            return ModelTimeResolution(self.value).slots_in_hour()

        raise ValueError(
            f"slots_in_hour() is not defined for resolution '{self}'. "
            f"Use in_minutes() or to_timedelta() instead."
        )

    def to_timedelta(self) -> timedelta:
        """Returns the timedelta corresponding to this resolution.

        Returns:
            timedelta object.
        """
        return timedelta(minutes=self.in_minutes())

    @classmethod
    def from_minutes(cls, minutes: int) -> "ModelTimeResolution_grid":
        """Returns the ModelTimeResolution_grid member corresponding to the given minutes.

        Args:
            minutes: Number of minutes.

        Returns:
            Corresponding ModelTimeResolution_grid member.

        Raises:
            ValueError: If no matching resolution is found.
        """

        # TODO: make minutes month dependent... 

        minutes_map = {
            1: cls.MINUTE,
            15: cls.MINUTES_15,
            60: cls.HOUR,
            60 * 24: cls.DAY,
            60 * 24 * 7: cls.WEEK,
            60 * 24 * 30: cls.MONTH,
        }
        if minutes not in minutes_map:
            raise ValueError(f"Unknown time resolution {minutes}")
        return minutes_map[minutes]



class GridPeakShaveInputs(GridInputs):
    """Inputs for the GridPeakShave component.

    Attributes:
        name: Name of the component.
        power_max: Maximum power limits, in units of power, e.g., MW.
        power_min: Minimum power limits, in units of power, e.g., MW.
        peak_power_price: The Price of the peak cost per unit of power, e.g., MW
        time_horizon_peak: The time period where ('day', 'week', 'month')
    """

    name: str = "grid_peak_shave"
    peak_power_price: Annotated[float | list[float], VectorOfReals("num_steps")] = 10000.0
    time_horizon_peak: ModelTimeResolution_grid | Literal["month", "week", "day", "hour", "15min", "1min"] = ModelTimeResolution_grid.MONTH 


class GridPeakShave(Grid):
    """Grid peak shave component.

    Extends the Grid component with additional peak shaving constraints and/or
    objectives. All standard grid energy balance and power limit constraints are
    inherited from the Grid component.

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
        assets: list[Asset] | None = None,
        markets: list[EnergyMarket] | None = None,
        name: str | None = None,
        power_max: (float | list[float]) | None = None,
        power_min: (float | list[float]) | None = None,
        peak_power_price: float | None = None,
        time_horizon_peak: ModelTimeResolution_grid | Literal["month", "week", "day", "hour", "15min", "1min"] = ModelTimeResolution_grid.MONTH 
    ):
        """Initializes the GridPeakShave component.

            Args:
                assets: List of assets.
                markets: List of energy markets.
                name: Name of the component.
                power_max: Maximum power limits, in units of power, e.g., MW.
                power_min: Minimum power limits, in units of power, e.g., MW.
                peak_power_price: Price of the peak power per unit of power, e.g., EUR/MW.
                time_horizon_peak: Time horizon over which the peak power is evaluated.
        """
        super().__init__(
            assets=assets,
            markets=markets,
            name=name,
            power_max=power_max,
            power_min=power_min,
        )

        if peak_power_price is not None:
            self.inputs.peak_power_price = peak_power_price
        if time_horizon_peak is not None:
            self.inputs.time_horizon_peak = ModelTimeResolution_grid(time_horizon_peak)


    def _get_peak_windows(self) -> list[list[int]]:
        """Slices the optimization horizon into windows of size time_horizon_peak.

        Each window is a list of timestep indices (k) belonging to that window.
        The last window may be shorter if the horizon is not evenly divisible.

        Returns:
            List of windows, where each window is a list of timestep indices.
        """
        peak_resolution = ModelTimeResolution_grid(self.inputs.time_horizon_peak)

        # self.dt is set by prepare_for_model_building() as slot_length_in_minutes / 60
        # convert back to minutes to derive the model resolution
        slot_length_in_minutes = round(self.dt[0] * 60)
        model_resolution = ModelTimeResolution_grid.from_minutes(slot_length_in_minutes)

        steps_per_window = peak_resolution.in_minutes() // model_resolution.in_minutes() 

        # Slice self.horizon into chunks of steps_per_window
        horizon = list(self.horizon)
        windows = [
            horizon[i : i + steps_per_window]
            for i in range(0, len(horizon), steps_per_window)
        ]
        return windows

    def init_variables(self) -> None:
        """Initializes the peak power variables for the GridPeakShave component.

        Creates one non-negative peak power variable per horizon window.
        Peak power only occurs due to energy consumption / charging behaviour,
        hence the NonNegativeReals domain.

        Populates self.peak_power with one pyo.Var per window.
        """
        super().init_variables()

        windows = self._get_peak_windows()
        self.var_peak_power = pyo.Var(range(len(windows)), within=pyo.NonNegativeReals)

    def _get_constraints_peak_shave(self) -> list:
        """Adds constraints ensuring var_peak_power[w] >= combined asset power
        at every timestep k within window w.

        Peak power only captures consumption / charging behaviour, consistent
        with the NonNegativeReals domain of var_peak_power.

        Returns:
            List of peak shaving constraints.
        """
        constraints = []

        windows = self._get_peak_windows()

        if len(self.assets) == 0 or len(windows) == 0:
            return constraints

        for w, window in enumerate(windows):
            constraints += [
                self.var_peak_power[w] >= -sum(c.power[k] for c in self.assets) # with negative sign we only penalize charging/consumption behaviour
                for k in window
            ]

        return constraints

    def _get_objectives_peak_shave(self) -> list:
        """Builds the peak shaving cost objective terms.

        The cost per window w is: peak_power_price * var_peak_power[w],
        where peak_power_price is a constant price (e.g., EUR/MW) and
        var_peak_power[w] is the optimized peak power within window w.

        Returns:
            List of objective expressions, one per peak window.
        """
        objectives = []

        windows = self._get_peak_windows()

        if len(windows) == 0:
            return objectives

        objectives += [
            self.inputs.peak_power_price * self.var_peak_power[w]
            for w in range(len(windows))
        ]

        return objectives

    def get_constraints(self) -> list:
        constraints = super().get_constraints()
        constraints += self._get_constraints_peak_shave()
        return constraints

    def get_objectives(self) -> list:
        objectives = super().get_objectives()
        objectives += self._get_objectives_peak_shave()
        return objectives


    def read_results(self, prefix: str = "") -> dict[str, list[float]]:
        """Return a dictionary of variable values.

        Overrides the base implementation to correctly read var_peak_power,
        which is indexed over peak windows rather than the model horizon.

        Returns:
            A dictionary of variable values.
        """
        results = {}

        # replicate base Component.read_results() but skip var_peak_power
        for attr_name, attr_value in self.__dict__.items():
            if not isinstance(attr_value, pyo.Var):
                continue
            if attr_name == "var_peak_power":
                continue  # handled separately below
            if isinstance(attr_value, pyo.ScalarVar):
                results[f"{prefix}{self.inputs.name}__{attr_name}"] = pyo.value(attr_value, exception=False)
            else:
                results[f"{prefix}{self.inputs.name}__{attr_name}"] = [
                    pyo.value(attr_value[k], exception=False) for k in self.horizon
                ]

        # handle var_peak_power with window-based indexing, expanded to horizon length
        windows = self._get_peak_windows()
        peak_power_per_step = []
        for w, window in enumerate(windows):
            peak_value = pyo.value(self.var_peak_power[w], exception=False)
            peak_power_per_step += [peak_value] * len(window) 

        results[f"{prefix}{self.inputs.name}__var_peak_power"] = peak_power_per_step

        return results


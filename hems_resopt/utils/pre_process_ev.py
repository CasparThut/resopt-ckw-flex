import pandas as pd
import logging

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(levelname)s: %(message)s',
    force    = True,          # ← key: overrides any existing logger config in Jupyter
)
logger = logging.getLogger(__name__)

# Round timesteps and keep data only within start and end-time
# ── Helper: Round timestamps to 15-min grid ────────────────────────────────────
def ceil_15min(ts: pd.Timestamp) -> pd.Timestamp:
    """Round a timestamp UP to the next 15-minute boundary (arrival snap)."""
    return ts.ceil('15min')


def floor_15min(ts: pd.Timestamp) -> pd.Timestamp:
    """Round a timestamp DOWN to the previous 15-minute boundary (departure snap)."""
    return ts.floor('15min')


# ── Helper: Filter & snap session times ───────────────────────────────────────
def prepare_sessions(
    pdf: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Filter sessions to the optimisation window and snap connection
    times to the 15-minute grid.

    - carConnected    → ceiled  to next  15-min step (arrival)
    - carDisconnected → floored to last  15-min step (departure)

    Parameters
    ----------
    pdf        : Raw sessions DataFrame from Databricks.
    start_time : Start of the optimisation horizon.
    end_time   : End   of the optimisation horizon.

    Returns
    -------
    Filtered and snapped sessions DataFrame.
    """
    pdf = pdf.copy()
    pdf['carConnected']    = pd.to_datetime(pdf['carConnected'])
    pdf['carDisconnected'] = pd.to_datetime(pdf['carDisconnected'])

    # Keep only sessions whose arrival falls within the window
    mask = (pdf['carConnected'] >= start_time) & (pdf['carConnected'] < end_time)
    pdf  = pdf.loc[mask].copy()

    # Snap to 15-min grid
    pdf['arrival_15min']   = pdf['carConnected'].apply(ceil_15min)
    pdf['departure_15min'] = pdf['carDisconnected'].apply(floor_15min)

    # session duration
    pdf['duration_hours'] = (pdf['departure_15min'] - pdf['arrival_15min']).dt.total_seconds() / 3600

    return pdf.reset_index(drop=True)


# ── Function 1: Build connection_df ───────────────────────────────────────────
def build_connection_df(
    sessions: pd.DataFrame,
    idx: pd.DatetimeIndex,
    charger_ids: list[str] | None = None,       # optional — defaults to all chargers in sessions
) -> pd.DataFrame:
    """
    Build a binary (0/1) DataFrame indicating whether a car is connected
    to each charger at every 15-minute timestep.

    Parameters
    ----------
    sessions    : Prepared sessions DataFrame (output of prepare_sessions).
    idx         : Full 15-minute DatetimeIndex for the optimisation horizon.
    charger_ids : List of charger IDs to include as columns.
                  If None, all unique charger IDs found in sessions are used.

    Returns
    -------
    connection_df : DataFrame[idx x charger_ids] with values 0 or 1.
    """
    # Fall back to all unique chargers present in the sessions data
    if charger_ids is None:
        charger_ids = sessions['chargerId'].unique().tolist()

    connection_df = pd.DataFrame(0, index=idx, columns=charger_ids)

    for _, row in sessions.iterrows():
        cid     = row['chargerId']
        arrival = row['arrival_15min']
        depart  = row['departure_15min']

        if cid in connection_df.columns and arrival <= depart:
            # Mark every timestep in [arrival, departure) as connected
            mask = (connection_df.index >= arrival) & (connection_df.index < depart)
            connection_df.loc[mask, cid] = 1

    return connection_df

def build_input_dict(
    charger_ids: list[str],
    value: float | dict[str, float],
) -> dict[str, float]:
    """
    Build a per-charger input dictionary from either a scalar or an
    existing dict."""

    if isinstance(value, dict):
            return value                                    # already per-charger
    return {cid: value for cid in charger_ids} 

# ── Function 3: Power bounds of the EV pool ───────────────────────────────────
def get_power_bounds(
    connection_df: pd.DataFrame,
    power_max_per_charger: dict[str, float],
    per_charger: bool = False
) -> pd.DataFrame:
    """
    Compute power_min and power_max of the EV pool at every timestep.
    ResOpt sees negative power as charging and thus power_min is the max(all connected chargers)

    - power_min = sum of p_max of all connected chargers *-1
    - power_max = 0  (no mandatory charging enforced at pool level)

    Parameters
    ----------
    connection_df         : Binary connection DataFrame.
    power_max_per_charger : Dict mapping charger_id → max charging power in kW.
                            Modular: each charger can have a different power rating.

    Returns
    -------
    DataFrame with columns ['power_min', 'power_max'].
    """

    if not per_charger:
        power_df = connection_df.multiply(
            pd.Series(power_max_per_charger), axis='columns'
        )
        power_min = power_df.sum(axis=1).rename('power_min') *-1
        power_max = pd.Series(0, index=connection_df.index, name='power_max')

        return pd.concat([power_min, power_max], axis=1)
    
    if per_charger:
        out = {}
        for cid in connection_df.columns:
            p_max = power_max_per_charger.get(cid, 0.0)
            out[f'power_min_{cid}'] = connection_df[cid] * p_max * -1
            out[f'power_max_{cid}'] = pd.Series(0.0, index=connection_df.index)

        return pd.DataFrame(out, index=connection_df.index)


# ── Helper: Feasibility check & capacity adjustment ───────────────────────────
def check_feasibility_of_session(
    session_e_in: float,
    session_e_out: float,
    charger_capacity_series: pd.Series,
) -> tuple[float, float, pd.Series]:
    """
    Check whether e_out is feasible given the charger's energy capacity
    at every timestep, and adjust the capacity series upward where needed
    so that e_out <= e_cap at all timesteps.

    Logic
    -----
    - e_out must not exceed e_cap at any timestep where the charger is active.
    - If e_out > e_cap at any timestep, e_cap is bumped up to e_out at those
      timesteps and the change is logged (old vs new value).
    - e_in is always <= e_out by construction (e_out = e_in + kWh charged >= e_in).

    Parameters
    ----------
    session_e_in             : Energy the car brings in  [kWh].
    session_e_out            : Energy the car leaves with [kWh]  (e_in + kWh charged).
    charger_capacity_series  : pd.Series of battery capacity for this charger
                               across all timesteps (column slice of capacity_df).

    Returns
    -------
    session_e_in             : Unchanged (returned for consistent unpacking).
    session_e_out            : Unchanged (returned for consistent unpacking).
    charger_capacity_series  : Adjusted capacity series (e_cap bumped where needed).
    """

    # ── Check 1: e_in sanity (e_out must be >= e_in) ──────────────────────────
    if session_e_out < session_e_in:
        logger.warning(
            "Infeasible session: e_out (%.2f kWh) < e_in (%.2f kWh). "
            "Clamping e_out to e_in.",
            session_e_out, session_e_in,
        )
        session_e_out = session_e_in  # clamp — no energy was lost

    # ── Check 2: e_out must not exceed e_cap at any active timestep ───────────
    # Only look at timesteps where the charger is actually active (capacity > 0)
    active_mask       = charger_capacity_series > 0
    violated_mask     = active_mask & (charger_capacity_series < session_e_out)
    violated_steps    = charger_capacity_series[violated_mask]

    if not violated_steps.empty:
        for ts, old_cap in violated_steps.items():
            new_cap = session_e_out
            charger_capacity_series.loc[ts] = new_cap

            logger.warning(
                "e_cap adjusted at [%s]: %.2f kWh → %.2f kWh "
                "(e_out = %.2f kWh exceeded old e_cap = %.2f kWh)",
                ts, old_cap, new_cap, session_e_out, old_cap,
            )

    return session_e_in, session_e_out, charger_capacity_series

# ── Function 4: e_in and e_out per timestep ───────────────────────────────────
# this currently works only if only one car arrives at a charger, not build yet for different battery sizes -> not to hard to adapt

def get_e_in_out_capacity(
    sessions: pd.DataFrame,
    connection_df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    battery_sizes: dict[str, float],
    power_sizes: dict[str, float],
    soc_rd_lower: float = 0.2,
    soc_rd_upper: float = 0.5,
    per_charger: bool = False,
) -> tuple[pd.DataFrame, list[float]]:
    """
    Compute e_in, e_out, and e_cap for the EV pool (or per charger) at every
    15-minute timestep.

    - e_in  is placed at the arrival   timestep: energy the car brings into the system.
            e_in  = battery_size - kWh_charged
    - e_out is placed at the departure timestep: energy the car leaves with.
            e_out = battery_size - buffer
    - e_cap is the energy capacity per timestep (pool sum, or per charger).

    Parameters
    ----------
    sessions      : Prepared sessions DataFrame (output of prepare_sessions).
    connection_df : Binary connection DataFrame (columns = charger_ids).
    idx           : Full 15-minute DatetimeIndex.
    battery_sizes : Dict mapping charger_id → battery capacity in kWh.
    power_sizes   : Dict mapping charger_id → max charging power in kW.
    soc_rd_lower  : Lower bound of the random arriving SoC  (default 0.2).
    soc_rd_upper  : Upper bound of the random arriving SoC  (default 0.5).
    per_charger   : If True, return one set of e_in/e_out/e_cap columns PER
                    charger instead of pool-level aggregates.

    Returns
    -------
    (df, capped_kWh_list)
        df : DataFrame with columns ['e_in', 'e_out', 'e_cap'] (pool mode)
             or ['e_in_{cid}', 'e_out_{cid}', 'e_cap_{cid}', ...] (per-charger mode).
        capped_kWh_list : list of kWh amounts capped due to power infeasibility.
    """
    charger_ids = connection_df.columns.tolist()

    e_in  = {cid: pd.Series(0.0, index=idx) for cid in charger_ids}
    e_out = {cid: pd.Series(0.0, index=idx) for cid in charger_ids}

    capacity_df = connection_df.multiply(pd.Series(battery_sizes), axis='columns')

    # Drop sessions smaller than 0.25 hours
    sessions = sessions[sessions['duration_hours'] > 0.5]
    capped_kWh_list = []

    for _, row in sessions.iterrows():
        cid     = row['chargerId']
        arrival = row['arrival_15min']
        depart  = row['departure_15min'] - pd.Timedelta(minutes=15)
        batt    = battery_sizes.get(cid, 80)

        if cid not in e_in:
            continue  # session belongs to a charger not present in connection_df

        if arrival not in idx or depart not in idx:
            continue  # skip sessions outside the optimisation window

        power_session = power_sizes.get(cid, 80)
        kWh_session = row['kiloWattHours']
        time_delta_hours = (depart - arrival).total_seconds() / 3600

        if time_delta_hours > 0:
            min_power_required = kWh_session / time_delta_hours
            if min_power_required > power_session:
                kWh_session_new = power_session * time_delta_hours
                logger.warning(
                    "Power infeasible for charger %s at [%s → %s]: "
                    "required %.2f kW > max %.2f kW. "
                    "kWh capped from %.2f kWh to %.2f kWh.",
                    cid, arrival, depart,
                    min_power_required, power_session,
                    kWh_session, kWh_session_new,
                )
                capped_kWh_list.append(kWh_session - kWh_session_new)
                kWh_session = kWh_session_new

        session_e_in  = max(0.0, batt - kWh_session)
        session_e_out = batt - 0.1

        e_in[cid].loc[arrival]  += session_e_in
        e_out[cid].loc[depart] += session_e_out

    kWh_capped = sum(capped_kWh_list)
    if kWh_capped > 0:
        logger.info(
            "Total kWh removed by power feasibility cap: %.2f kWh across %d sessions.",
            kWh_capped, len(capped_kWh_list),
        )

    if per_charger:
        out = {}
        for cid in charger_ids:
            out[f'e_in_{cid}']  = e_in[cid]
            out[f'e_out_{cid}'] = e_out[cid]
            out[f'e_cap_{cid}'] = capacity_df[cid]
        return pd.DataFrame(out, index=idx), capped_kWh_list

    # Pool-level aggregation (original behaviour)
    e_in_total  = sum(e_in.values()).rename('e_in')
    e_out_total = sum(e_out.values()).rename('e_out')
    e_cap_total = capacity_df.sum(axis=1).rename('e_cap')

    return pd.concat([e_in_total, e_out_total, e_cap_total], axis=1), capped_kWh_list

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



# ── Function 3: Power bounds of the EV pool ───────────────────────────────────
def get_power_bounds(
    connection_df: pd.DataFrame,
    power_max_per_charger: dict[str, float],
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
    power_df = connection_df.multiply(
        pd.Series(power_max_per_charger), axis='columns'
    )
    power_min = power_df.sum(axis=1).rename('power_min') *-1
    power_max = pd.Series(0, index=connection_df.index, name='power_max')

    return pd.concat([power_min, power_max], axis=1)


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
) -> pd.DataFrame:
    """
    Compute e_in and e_out for the EV pool at every 15-minute timestep.
    Compute the energy capacity of the connected EV Pool

    - e_in  is placed at the arrival   timestep: energy the car brings into the system.
            e_in  = battery_size - kWh_charged
    - e_out is placed at the departure timestep: energy the car leaves with.
            e_out = e_in + kiloWattHours charged during the session
    - e_cap is the total energy capacity (max per timestep) of the entire pool that is connected

    - include a controll function that checks if the random defined entry SoC (e_in) aligns with e_out (=e_in + kWh charged) and e_capacity

    Parameters
    ----------
    sessions      : Prepared sessions DataFrame (output of prepare_sessions).
    connection_df : Output of connection function with boolean if charger is connected of not
    idx           : Full 15-minute DatetimeIndex.
    battery_sizes : Dict mapping charger_id → battery capacity in kWh.
    soc_rd_lower  : Lower bound of the random arriving SoC  (default 0.2).
    soc_rd_upper  : Upper bound of the random arriving SoC  (default 0.5).

    Returns
    -------
    DataFrame with columns ['e_in', 'e_out'] indexed by idx.
    """
    e_in  = pd.Series(0.0, index=idx, name='e_in')
    e_out = pd.Series(0.0, index=idx, name='e_out')
    e_cap = pd.Series(0.0, index=idx, name='e_cap')
    
    capacity_df = connection_df.multiply(
        pd.Series(battery_sizes), axis='columns')
    
    # Drop sessions smaller 0.25 hours
    sessions = sessions[sessions['duration_hours'] > 0.25]
    capped_kWh_list = []

    for _, row in sessions.iterrows():
        cid     = row['chargerId']
        arrival = row['arrival_15min']
        depart  = row['departure_15min'] - pd.Timedelta(minutes=15) # k-1 is needed to align e_out with the optimization variables soc_slot_end
        batt    = battery_sizes.get(cid, 80)  # fallback to 80 kWh

        if arrival not in idx or depart not in idx:
            continue  # skip sessions outside the optimisation window

        # Feasibility check: charging is possible during connected time 
        power_session = power_sizes.get(cid, 80)
        kWh_session = row['kiloWattHours']
        time_delta_hours = (depart - arrival).total_seconds() / 3600 # value as hour
        min_power_required = kWh_session / time_delta_hours
        
        if time_delta_hours > 0:
            min_power_required = kWh_session / time_delta_hours  # kW
            if min_power_required > power_session:
                # Cap kWh to what is physically deliverable within the session
                kWh_session_new = power_session * time_delta_hours
                logger.warning(
                    "Power infeasible for charger %s at [%s → %s]: "
                    "required %.2f kW > max %.2f kW. "
                    "kWh capped from %.2f kWh to %.2f kWh.",
                    cid, arrival, depart,
                    min_power_required, power_session,
                    kWh_session, kWh_session_new,
                )

                # Append to list
                capped_kWh_list.append(kWh_session - kWh_session_new)
                kWh_session = kWh_session_new


        # Focus on kWh per Session, start-, end- and e_cap are only relevant for the optimization to work smoothly
        # assumption car will always be full when leaving
        session_e_in  = max(0.0, batt - kWh_session)
        session_e_out = batt - 0.1

        
        # session_e_in, session_e_out, capacity_df[cid] = check_feasibility_of_session(session_e_in, session_e_out, charger_cap_series) # not needed as restrictions can't be violated like this

        e_in.loc[arrival] += session_e_in
        e_out.loc[depart] += session_e_out

    e_cap = capacity_df.sum(axis=1)

    kWh_capped = sum(capped_kWh_list)

    if kWh_capped > 0:
        logger.info(
            "Total kWh removed by power feasibility cap: %.2f kWh across %d sessions.",
            kWh_capped, len(capped_kWh_list),
        )

    return pd.concat([e_in, e_out, e_cap], axis=1), capped_kWh_list
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import matplotlib as mpl

# Set default plot properties (rcParams)
plt.rcParams['figure.figsize'] = [15, 8]  # Set the default figure size (width, height)
plt.rcParams['figure.dpi'] = 300  # Set the resolution (300 DPI for high-quality output)
plt.rcParams['lines.linewidth'] = 1     # Set default line width
plt.rcParams['lines.color'] = 'b'         # Set default line color (blue in this case)
plt.rcParams['axes.grid'] = True 
plt.rcParams['axes.grid.axis'] = 'y'
plt.rcParams['grid.alpha'] = 0.3    
plt.rcParams['axes.titlecolor'] = '#616161'
plt.rcParams['axes.labelcolor'] = '#616161'
plt.rcParams['xtick.color'] = '#616161'
plt.rcParams['ytick.color'] = '#616161'
plt.rcParams['text.color'] = '#616161'
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.markerscale'] = 1
plt.rcParams['legend.framealpha'] = 0.5
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#BCCF02", "#6C9C30", "#3E5C17", "#a7b805"]) 
# plt.rcParams["patch.facecolor"] = "#BCCF02"

plt.rcParams['figure.constrained_layout.use'] = True # alternativ zu tight_layout(). noch testen sonst deaktivieren



def plot_charger_usage(easee_sessions, df_energy, idx, connection_df, sessions=None, session_kWh_col='kiloWattHours'):
    """
    Plot charger usage diagnostics.

    Parameters
    ----------
    easee_sessions  : Raw Easee sessions DataFrame (carConnected / carDisconnected).
    df_energy       : Optimisation output DataFrame with 'energy_charged' column.
    idx             : Full 15-minute DatetimeIndex for the optimisation horizon.
    connection_df   : Binary (0/1) DataFrame [idx x charger_ids] from build_connection_df().
    sessions        : Prepared sessions DataFrame (used for daily energy comparison).
    session_kWh_col : Column name in sessions holding per-session energy [kWh].
    """

    # ── Subplot 1: Concurrent active charging sessions ────────────────────────
    # Sum across all charger columns → number of connected chargers at each timestep
    concurrent = connection_df.sum(axis=1)  # pd.Series aligned to idx

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    ax = axes[0]
    ax.plot(concurrent.index, concurrent.values, color="#3E5C17")
    ax.set_title("Concurrent active charging sessions over time")
    ax.set_ylabel("Number of active sessions")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Subplot 2: Session duration distribution ──────────────────────────────
    durations = (easee_sessions['carDisconnected'] - easee_sessions['carConnected']).dt.total_seconds().div(3600).dropna()
    ax2 = axes[1]
    ax2.hist(durations.clip(upper=24), bins=30, color="#6C9C30", alpha=0.8)
    ax2.set_xlabel("Session duration [h]")
    ax2.set_ylabel("Count")
    ax2.set_title("Session duration distribution (capped at 24h)")

    # ── Subplot 3: Daily energy — Optimized vs Sessions ───────────────────────
    ax3 = axes[2]

    daily_opt = df_energy['energy_charged'].fillna(0).resample('D').sum()

    if sessions is None:
        try:
            sess_df = globals().get('sessions', None)
        except Exception:
            sess_df = None
    else:
        sess_df = sessions

    if sess_df is not None and session_kWh_col in sess_df.columns:
        sess_days = sess_df.copy()
        sess_days['day'] = sess_days['carConnected'].dt.normalize()
        daily_sess = sess_days.groupby('day')[session_kWh_col].sum()
        daily_index = pd.date_range(start=idx[0].normalize(), end=idx[-1].normalize(), freq='D', tz=idx.tz)
        daily_sess = daily_sess.reindex(daily_index, fill_value=0)
    else:
        daily_index = daily_opt.index.union(pd.date_range(start=idx[0].normalize(), end=idx[-1].normalize(), freq='D', tz=idx.tz))
        daily_sess = pd.Series(0.0, index=daily_index)

    all_days = daily_opt.index.union(daily_sess.index).sort_values()
    daily_opt  = daily_opt.reindex(all_days, fill_value=0)
    daily_sess = daily_sess.reindex(all_days, fill_value=0)

    x     = np.arange(len(all_days))
    width = 0.4
    ax3.bar(x - width/2, daily_opt.values,  width=width, label='Optimized energy charged (kWh/day)', color='#3E5C17')
    ax3.bar(x + width/2, daily_sess.values, width=width, label='Sessions energy (kWh/day)',           color='#BCCF02', alpha=0.9)

    if len(all_days) > 12:
        step        = max(1, len(all_days) // 12)
        xticks      = x[::step]
        xticklabels = [all_days[i].strftime('%Y-%m-%d') for i in xticks]
    else:
        xticks      = x
        xticklabels = [d.strftime('%Y-%m-%d') for d in all_days]

    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticklabels, rotation=30, ha='right')
    ax3.set_ylabel('kWh/day')
    ax3.set_title('Daily energy: Optimized vs Sessions')
    ax3.legend()

    plt.show()





# ---------- Representative week: prices and energy charged ----------
def plot_representative_week(df_prices, df_energy, use_second_half=True):
    """
    Produce a representative week for price and energy_charged following your template.
    df_prices: Series (index=DatetimeIndex) price
    df_energy: Series energy_charged (index=DatetimeIndex)
    """
    # build a combined frame
    combo = pd.DataFrame({
        'price': df_prices,
        'energy_charged': df_energy
    }).dropna(how='all')

    # pick second half (or first)
    half = combo.iloc[len(combo)//2:] if use_second_half else combo.iloc[:len(combo)//2]

    # keys
    half = half.copy()
    half['weekday'] = half.index.weekday
    half['hour'] = half.index.hour

    # ensure numeric
    for c in ['price','energy_charged']:
        half[c] = pd.to_numeric(half[c], errors='coerce')

    # group mean
    rep = half.groupby(['weekday','hour'])[['price','energy_charged']].mean()

    # map to continuous week
    start = pd.Timestamp('2024-01-01')  # Monday
    idx_week = pd.date_range(start, periods=7*24, freq='h', tz='UTC')
    map_df = pd.DataFrame({'weekday': idx_week.weekday, 'hour': idx_week.hour}, index=idx_week)

    rep_week = (
        map_df
        .merge(rep, left_on=['weekday','hour'], right_index=True, how='left')
        .drop(columns=['weekday','hour'])
    )

    # plot
    fig, ax1 = plt.subplots(figsize=(12,4))
    ax2 = ax1.twinx()
    ax1.plot(rep_week.index, rep_week['price'], color='#DD8500', label='Price (avg)')
    ax2.plot(rep_week.index, rep_week['energy_charged'], color='#3E5C17', label='Energy charged (avg)')
    ax1.set_title("Representative week — avg price and energy charged (hourly)")
    ax1.set_ylabel("Price CHF/kWh", color='#DD8500')
    ax2.set_ylabel("Energy charged kWh", color='#3E5C17')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()



def compute_ev_optimization_summary(
    df_results: pd.DataFrame,
    df_ev_inputs: pd.DataFrame,
    idx: pd.DatetimeIndex,
    sessions: pd.DataFrame,
    capped_kWh_list: list,
    peak_power_price_dyn: float = 1.0,
    peak_power_price_stat: float = 1.5,
    energy_costs: float = 0.11,
    fix_costs: float = 0.07226,
    EV_opt_name: str = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Compute post-processing metrics for EV charging optimization.

    Parameters
    ----------
    df_results : pd.DataFrame
        Optimization results; must contain column 'dynamic_tariff__var_power' (kW; negative convention used in your code).
    df_ev_inputs : pd.DataFrame
        Inputs with index = idx and column 'dyn_tarif' (CHF/kWh).
    idx : pd.DatetimeIndex
        Full datetime index for the optimization horizon (tz-aware recommended).
    sessions : pd.DataFrame
        Sessions dataframe with columns 'kiloWattHours', 'carConnected', 'carDisconnected'.
    capped_kWh_list : list
        List of kWh values lost due to capping (numeric).
    peak_power_price : float
        CHF per kW for monthly peak tariff (default 10 CHF/kW).
    energy_costs : float
        additional ELV or energy cost per kWh (default 0.2 CHF/kWh).

    Returns
    -------
    df : pd.DataFrame
        Time series DataFrame indexed by idx with columns: power_charged, energy_charged, charging_costs, price.
    summary : dict
        Summary KPIs (same keys as in your original 'summary').
    monthly : pd.DataFrame
        Monthly aggregated DataFrame with energy and cost components.
    """
    # Validate inputs
    if 'dyn_tarif' not in df_ev_inputs.columns:
        raise ValueError("df_ev_inputs must contain 'dyn_tarif' column")
    if 'dynamic_tariff__var_power' not in df_results.columns:
        raise ValueError("df_results must contain 'dynamic_tariff__var_power' column")
    if 'kiloWattHours' not in sessions.columns:
        raise ValueError("sessions must contain 'kiloWattHours' column")
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("idx must be a pandas DatetimeIndex")

    # Parameters
    mean_costs_tariff = df_ev_inputs['dyn_tarif'].mean()
    fixed_grid_costs = mean_costs_tariff


    # Time series df
    df = pd.DataFrame(index=idx)
    # 1. Core charging metrics
    df['power_charged']  = df_results['dynamic_tariff__var_power'] * -1
    df['energy_charged'] = df['power_charged'] / 4.0
    df['charging_costs'] = df['energy_charged'] * df_ev_inputs['dyn_tarif']
    df['price']          = df_ev_inputs['dyn_tarif']

    # 2. Monthly peaks and peak costs
    peaks_month = df['power_charged'].resample('ME').max()
    peak_costs_opt  = peaks_month * peak_power_price_dyn
    peak_costs_stand = peaks_month * peak_power_price_stat

    # 3. Total optimized costs
    energy_costs_grid_opt = df['charging_costs'].sum()
    energy_costs_elv_opt  = df['energy_charged'].sum() * energy_costs
    energy_costs_tot_opt  = energy_costs_grid_opt + energy_costs_elv_opt
    fix_costs_kWh         = df['energy_charged'].sum() * fix_costs
    total_peak_costs_opt  = peak_costs_opt.sum()
    total_costs_opt       = energy_costs_grid_opt + energy_costs_elv_opt + total_peak_costs_opt + fix_costs_kWh

    # 4. Baseline (flat tariff)
    peaks_month_stand = peaks_month
    total_peak_costs_stand = peak_costs_stand.sum()
    total_energy_charged = df['energy_charged'].sum()
    energy_costs_grid_stand = total_energy_charged * fixed_grid_costs
    energy_costs_elv_stand  = total_energy_charged * energy_costs

    total_costs_stand = energy_costs_grid_stand + energy_costs_elv_stand + total_peak_costs_stand + fix_costs_kWh

    # 5. Optimization savings
    # savings_opt_rl = total_costs_stand - energy_costs_tot_opt
    # savings_opt_rl_pct = (savings_opt_rl / total_costs_stand) * 100 if total_costs_stand != 0 else np.nan
    # savings_opt_incl_peak = total_costs_stand - total_costs_opt
    # savings_opt_incl_peak_pct = (savings_opt_incl_peak / total_costs_stand) * 100 if total_costs_stand != 0 else np.nan
    # savings_opt_energy = energy_costs_tot_stand - energy_costs_tot_opt
    # savings_opt_energy_pct = (savings_opt_energy / energy_costs_tot_stand) * 100 if energy_costs_tot_stand != 0 else np.nan

    # 6. Session energy audit
    kWh_in_sessions = sessions['kiloWattHours'].sum()
    kWh_capped_delta = sum(capped_kWh_list) if getattr(capped_kWh_list, '__len__', None) else float(capped_kWh_list or 0)
    kWh_capped_pct = (kWh_capped_delta / kWh_in_sessions) * 100 if kWh_in_sessions != 0 else np.nan

    # 7. Average charging price
    avg_energy_grid_price_opt = energy_costs_grid_opt / total_energy_charged if total_energy_charged != 0 else np.nan
    avg_energy_price_opt      = energy_costs_elv_opt / total_energy_charged if total_energy_charged != 0 else np.nan
    avg_tot_price_opt  = total_costs_opt / total_energy_charged if total_energy_charged != 0 else np.nan
    avg_peak_costs_opt        = total_peak_costs_opt / total_energy_charged if total_energy_charged != 0 else np.nan
    avg_energy_grid_price_stand = fixed_grid_costs
    avg_energy_price_stand      = energy_costs
    avg_peak_costs_stand        = total_peak_costs_stand / total_energy_charged if total_energy_charged != 0 else np.nan
    avg_energy_tot_price_stand  = fixed_grid_costs + energy_costs + avg_peak_costs_stand + fix_costs

    # 8. Load profile quality
    # avoid division by zero
    par_optimized = (df['power_charged'].max() / df['power_charged'].mean()) if df['power_charged'].mean() != 0 else np.nan
    off_peak_mask = df.index.hour.isin(range(22,24)) | df.index.hour.isin(range(0,6))
    kWh_off_peak = df.loc[off_peak_mask, 'energy_charged'].sum()
    kWh_on_peak  = df.loc[~off_peak_mask, 'energy_charged'].sum()
    off_peak_share = (kWh_off_peak / total_energy_charged) * 100 if total_energy_charged != 0 else np.nan

    # 9. Metadata of the run
    first_session = sessions['carConnected'].min()
    last_session = sessions['carConnected'].max()
    number_of_sessions = len(sessions)
    average_kWh_session = sessions['kiloWattHours'].mean()
    involved_chargers = sessions['charger_id'].unique()
    number_of_chargers = len(involved_chargers)

    # ──SoC Violations ────────────────────
    number_of_soc_violations = None

    if EV_opt_name is not None:
        min_energy_violations = f'{EV_opt_name}__var_min_energy_violation'
        max_energy_violations = f'{EV_opt_name}__var_max_energy_violation'
        n_violations = 0

        if min_energy_violations in df_results.columns:
            series_min = df_results[min_energy_violations]
            
            tot_min_energy_viol = sum(series_min)
            
            n_min_energy_viol = series_min != 0
            n_violations += n_min_energy_viol.sum()
        if max_energy_violations in df_results.columns:
            series_max = df_results[max_energy_violations]
            
            tot_max_energy_viol = sum(series_max)

            n_max_energy_viol = series_max != 0
            n_violations += n_max_energy_viol.sum()


    # 9. Summary
    summary = {
        # Energy
        'Total Energy Charged [kWh]'      : round(total_energy_charged, 2),
        'Energy in Raw Sessions [kWh]'    : round(kWh_in_sessions, 2),
        'Energy Lost to Power Cap [kWh]'  : round(kWh_capped_delta, 2),
        'Energy Lost to Power Cap [%]'    : round(kWh_capped_pct, 2),

        # Costs
        'Energy Costs Grid Opt [CHF]'     : round(energy_costs_grid_opt, 2),
        'Energy Costs ELV Opt [CHF]'      : round(energy_costs_elv_opt, 2),
        'Peak Costs Opt [CHF]'            : round(total_peak_costs_opt, 2),
        'Fix Costs (Abgaben etc.) [CHF]'  : round(fix_costs_kWh, 2),
        'Total Optimized Costs [CHF]'     : round(total_costs_opt, 2),
        'Energy Costs Grid (flat tariff) [CHF]': round(energy_costs_grid_stand, 2),
        'Energy Costs ELV (flat tariff) [CHF]': round(energy_costs_elv_stand, 2),
        'Peak costs (flat tariff) [CHF]'       : round(total_peak_costs_stand, 2),
        'Total costs (flat tariff) [CHF]'      : round(total_costs_stand, 2),

        # Prices
        'Avg Charging Price Grid Opt [CHF/kWh]': round(avg_energy_grid_price_opt, 5),
        'Avg Charging Price ELV Opt [CHF/kWh]' : round(avg_energy_price_opt, 5),
        'Avg Charging Price Peak Costs Opt [CHF/kWh]': round(avg_peak_costs_opt, 5),
        'Avg Charging Price Abgaben etc. [CHF/kWh]': round(fix_costs, 5),
        'Avg Charging Price Opt [CHF/kWh]'     : round(avg_tot_price_opt, 5),
        'Avg Charging Price Grid Standard [CHF/kWh]': round(avg_energy_grid_price_stand, 5),
        'Avg Charging Price ELV Standard [CHF/kWh]' : round(avg_energy_price_stand, 5),
        'Avg Charging Price Peak Costs Standard [CHF/kWh]': round(avg_peak_costs_stand, 5),
        'Avg Charging Price Abgaben etc. [CHF/kWh]': round(fix_costs, 5),
        'Avg Charging Price Standard [CHF/kWh]'     : round(avg_energy_tot_price_stand, 5),

        # Load profile quality
        'Peak-to-Average Ratio [-]' : round(par_optimized, 2),
        'Off-Peak Charging Share [%]': round(off_peak_share, 2),
        'On-Peak  Charging Share [%]': round(100 - off_peak_share, 2),

        # Charger and session data
        'Number of Chargers':  number_of_chargers ,
        'Sessions': number_of_sessions,
        'First Session' : first_session,
        'Last Session':last_session,
        'Average kWh per Session':average_kWh_session,
        'Number of SoC violations': n_violations,
        'Total Energy of min SoC violations': round(tot_min_energy_viol,5),
        'Total Energy of max SoC violations': round(tot_max_energy_viol,5)
    }

    # Monthly breakdown
    monthly = df['energy_charged'].resample('ME').sum().rename('energy_charged_kWh').to_frame()
    monthly['energy_costs_grid_opt_CHF'] = df['charging_costs'].resample('ME').sum()
    monthly['energy_costs_grid_stand_CHF'] = df['energy_charged'].resample('ME').sum() * fixed_grid_costs
    monthly['energy_costs_ELV_opt_CHF'] = df['energy_charged'].resample('ME').sum() * energy_costs
    monthly['energy_costs_ELV_stand_CHF'] = df['energy_charged'].resample('ME').sum() * energy_costs
    monthly['peak_kW'] = peaks_month
    monthly['peak_costs_opt_CHF'] = peak_costs_opt
    monthly['peak_costs_stand_CHF'] = peak_costs_stand
    monthly['fix_costs_CHF']            = df['energy_charged'].resample('ME').sum() * fix_costs
    monthly['total_costs_optimized_CHF'] = monthly['energy_costs_grid_opt_CHF'] + monthly['energy_costs_ELV_opt_CHF'] + monthly['peak_costs_opt_CHF'] + monthly['fix_costs_CHF'] 
    monthly['total_costs_standard_CHF'] = monthly['energy_costs_grid_stand_CHF'] + monthly['energy_costs_ELV_stand_CHF'] + monthly['peak_costs_stand_CHF'] + monthly['fix_costs_CHF'] 


    return df, summary, monthly


def print_summary(summary: dict):
    """
    Nicely formatted printout for the summary dict produced by the optimizer.
    Groups metrics, aligns values, adds units and thousands separators.
    """
    # Define groups and display formatting per key pattern
    energy_keys = [
        'Total Energy Charged [kWh]',
        'Energy in Raw Sessions [kWh]',
        'Energy Lost to Power Cap [kWh]',
        'Energy Lost to Power Cap [%]'
    ]
    cost_keys = [
        'Energy Costs Grid Opt [CHF]',
        'Energy Costs ELV Opt [CHF]',
        'Peak Costs Opt [CHF]',
        'Fix Costs (Abgaben etc.) [CHF]',
        'Total Optimized Costs [CHF]',
        'Energy Costs Grid (flat tariff) [CHF]',
        'Energy Costs ELV (flat tariff) [CHF]',
        'Peak costs (flat tariff) [CHF]',
        'Fix Costs (Abgaben etc.) [CHF]',
        'Total costs (flat tariff) [CHF]'
    ]

    price_keys = [
        'Avg Charging Price Grid Opt [CHF/kWh]',
        'Avg Charging Price ELV Opt [CHF/kWh]',
        'Avg Charging Price Peak Costs Opt [CHF/kWh]',
        'Avg Charging Price Abgaben etc. [CHF/kWh]',
        'Avg Charging Price Opt [CHF/kWh]',
        'Avg Charging Price Grid Standard [CHF/kWh]',
        'Avg Charging Price ELV Standard [CHF/kWh]',
        'Avg Charging Price Peak Costs Standard [CHF/kWh]',
        'Avg Charging Price Abgaben etc. [CHF/kWh]',
        'Avg Charging Price Standard [CHF/kWh]'
    ]
    load_keys = [
        'Peak-to-Average Ratio [-]',
        'Off-Peak Charging Share [%]',
        'On-Peak  Charging Share [%]'
    ]
    meta_data_keys = [        # Charger and session data
        'Number of Chargers',
        'Sessions',
        'First Session',
        'Last Session',
        'Average kWh per Session',
        'Number of SoC violations',
        'Total Energy of min SoC violations',
        'Total Energy of max SoC violations']

    def fmt(val, key):
        # None or nan
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        # percentages
        if isinstance(key, str) and key.strip().endswith('[%]'):
            return f"{val:,.2f}%"
        # CHF currency
        if isinstance(key, str) and 'CHF' in key:
            # show thousands separators, 0 decimals if large
            if abs(val) >= 1000:
                return f"{val:,.0f} CHF"
            return f"{val:,.2f} CHF"
        # kWh
        if isinstance(key, str) and '[kWh]' in key:
            return f"{val:,.2f} kWh"
        # price per kWh
        if isinstance(key, str) and 'CHF/kWh' in key:
            return f"{val:,.4f} CHF/kWh"
        # ratio
        if isinstance(key, str) and 'Ratio' in key:
            return f"{val:,.2f}"
        # fallback numeric
        if isinstance(val, (int, float)):
            return f"{val:,.2f}"
        return str(val)

    # prepare ordered groups
    groups = [
        ("Energy", energy_keys),
        ("Costs (optimized vs flat)", cost_keys),
        # ("Savings", savings_keys),
        ("Avg Prices", price_keys),
        ("Load profile quality", load_keys),
        ("Site Data", meta_data_keys)
    ]

    # compute column width
    label_width = max(len(k) for k in summary.keys()) + 2
    value_width = 18

    # Header
    sep = "=" * (label_width + value_width + 6)
    print("\n" + sep)
    print(f"{'EV CHARGING OPTIMISATION — SUMMARY':^{label_width + value_width + 6}}")
    print(sep)

    for title, keys in groups:
        print(f"\n-- {title} --")
        for k in keys:
            if k not in summary:
                continue
            v = summary[k]
            print(f"  {k:<{label_width}} {fmt(v, k):>{value_width}}")
    print(sep + "\n")
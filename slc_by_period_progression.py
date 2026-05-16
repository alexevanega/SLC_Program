import pandas as pd
import math
from utility import calculate_shot_probability, calculate_service_tax, get_total_seconds, calculate_slc_score

def _empty_period_stats():
    return {
        'S_saves': 0,
        'S_goals': 0,
        'NPW': 0,
        'NPW_SH': 0,
        'iTkA': 0,
        'iGvA': 0,
        'UA': 0,
        'UA_SH': 0,
        'RP': 0,
        'RP_SH': 0,
        'iGvA_SH': 0,
        'xG': 0.0,
        'xS': 0.0,
        'SLC_event': 0.0,
        'Survival_Event': 0.0,
        'Relief_Event': 0.0,
        'Pressure_Cost_Event': 0.0,
        'Pressure_Weight': 0.0,
        'Sequences': 0,
        'Lateral_Adjustments': 0,
        'Sequence_Fatigue': 0.0,
        'xG_Calibration': 0.0
    }


def _details(event):
    details = event.get('details', {}) if isinstance(event, dict) else {}
    return details if isinstance(details, dict) else {}


def _infer_goalie_team_id(raw_game_data, goalie_id):
    home_id = raw_game_data.get('homeTeam', {}).get('id')
    away_id = raw_game_data.get('awayTeam', {}).get('id')

    for event in raw_game_data.get('plays', []):
        details = _details(event)
        if details.get('goalieInNetId') != goalie_id:
            continue

        shooter_team_id = details.get('eventOwnerTeamId')
        if shooter_team_id == home_id:
            return away_id
        if shooter_team_id == away_id:
            return home_id

    return None


def _is_goalie_team_short_handed(situation_code, raw_game_data, goalie_team_id):
    if goalie_team_id is None:
        return False

    sit_code = str(situation_code or '1551')
    if len(sit_code) < 3 or not sit_code[1].isdigit() or not sit_code[2].isdigit():
        return False

    away_skaters = int(sit_code[1])
    home_skaters = int(sit_code[2])
    home_id = raw_game_data.get('homeTeam', {}).get('id')
    away_id = raw_game_data.get('awayTeam', {}).get('id')

    if goalie_team_id == home_id:
        return home_skaters < away_skaters
    if goalie_team_id == away_id:
        return away_skaters < home_skaters
    return False


def _time_gap_seconds(current_event, previous_event):
    if not previous_event:
        return None
    current_period = current_event.get('period') or current_event.get('periodDescriptor', {}).get('number', 1)
    previous_period = previous_event.get('period') or previous_event.get('periodDescriptor', {}).get('number', current_period)
    current_time = current_event.get('time_sec', get_total_seconds(current_event.get('timeInPeriod'), current_period))
    previous_time = previous_event.get('time_sec', get_total_seconds(previous_event.get('timeInPeriod'), previous_period))
    return current_time - previous_time


def _same_pressure_sequence(event, prev_event):
    if not prev_event:
        return False
    if event.get('sortOrder', 0) - prev_event.get('sortOrder', 0) > 12:
        return False
    gap = _time_gap_seconds(event, prev_event)
    if gap is None or gap < 0 or gap >= 10:
        return False
    if prev_event.get('typeDescKey') not in {'shot-on-goal', 'blocked-shot'}:
        return False
    return _details(event).get('zoneCode') == _details(prev_event).get('zoneCode')


def _is_breakdown(event, prev_event):
    if not prev_event:
        return False

    details = _details(event)
    prev_details = _details(prev_event)
    gap = _time_gap_seconds(event, prev_event)
    if gap is None or gap < 0:
        return False

    transition_gate = (
        prev_event.get('typeDescKey') in {'giveaway', 'takeaway'} and
        prev_details.get('eventOwnerTeamId') == details.get('eventOwnerTeamId') and
        prev_details.get('zoneCode') == 'D' and
        details.get('zoneCode') == 'O' and
        gap < 4
    )

    royal_road_gate = (
        prev_event.get('typeDescKey') in {'pass', 'takeover', 'takeaway'} and
        gap < 1 and
        abs(details.get('yCoord', 0) - prev_details.get('yCoord', 0)) > 20
    )

    return transition_gate or royal_road_gate


def _has_lateral_pressure(event, prev_event):
    if not prev_event:
        return False

    if prev_event.get('typeDescKey') not in {'pass', 'giveaway'}:
        return False

    details = _details(event)
    prev_details = _details(prev_event)
    gap = _time_gap_seconds(event, prev_event)
    if gap is None or gap < 0 or gap > 1.5:
        return False

    if prev_details.get('zoneCode') not in {'D', 'O'}:
        return False

    current_y = details.get('yCoord', 0)
    previous_y = prev_details.get('yCoord', 0)
    crossed_slot = (current_y * previous_y < 0) or abs(current_y - previous_y) > 20
    return crossed_slot


def _calibrated_game_xg_factor(raw_game_data):
    shots = 0
    goals = 0
    model_xg = 0.0
    prev_event = None

    for event in raw_game_data.get('plays', []):
        period_desc = event.get('periodDescriptor', {})
        period_type = period_desc.get('periodType') or period_desc.get('type')
        period_number = period_desc.get('number', 1)
        event_type = event.get('typeDescKey')

        if period_type == 'SO' or period_number > 4:
            prev_event = event
            continue

        if event_type not in {'shot-on-goal', 'goal'}:
            prev_event = event
            continue

        details = _details(event)
        if details.get('xCoord') is None or details.get('yCoord') is None:
            prev_event = event
            continue

        shots += 1
        goals += 1 if event_type == 'goal' else 0
        model_xg += calculate_shot_probability(event, prev_event)
        prev_event = event

    if shots == 0 or model_xg <= 0:
        return 1.0

    model_goal_rate = model_xg / shots
    actual_goal_rate = goals / shots
    smoothed_actual_rate = ((actual_goal_rate * shots) + (model_goal_rate * 40)) / (shots + 40)
    factor = smoothed_actual_rate / model_goal_rate if model_goal_rate else 1.0
    return max(0.75, min(1.75, factor))


def _event_slc_contribution(event, prev_event, system_state, is_sh=False):
    details = _details(event)
    current_time = event.get('time_sec', 0)
    base_xg = calculate_shot_probability(event, prev_event)
    lateral_pressure = _has_lateral_pressure(event, prev_event)
    if lateral_pressure:
        base_xg *= 2.0

    sequence_shot_count = system_state.get("sequence_shot_count", 0) + 1
    sequence_fatigue_factor = 1.0 + (min(sequence_shot_count, 5) - 1) * 0.12
    base_xg *= sequence_fatigue_factor
    base_xg *= system_state.get("xg_calibration_factor", 1.0)
    base_xg = min(round(float(base_xg), 4), 0.999)

    adjusted_xs = 1.0 - base_xg
    breakdown = _is_breakdown(event, prev_event)
    if breakdown:
        adjusted_xs = min(adjusted_xs, 0.15)

    t_gap = current_time - system_state["last_defensive_event_time"]
    tau_inactivity = 1.0
    if t_gap > 300:
        tau_inactivity = 1.0 + min(1.0, (t_gap - 300) / 600)

    tau_sequence = 1.5 if _same_pressure_sequence(event, prev_event) else 0.0
    result = 0 if event.get('typeDescKey') == 'goal' else 1
    tax = tau_inactivity + tau_sequence
    if breakdown:
        tax = max(tax, 2.0)
    if is_sh:
        tax *= 1.5

    if result == 1:
        survival_event = 1 - adjusted_xs
    else:
        survival_event = 0 - adjusted_xs

    pressure_cost_event = max(tax - 1.0, 0.0)
    if result == 0:
        pressure_cost_event += 0.50
    if breakdown:
        pressure_cost_event += 0.35

    relief_event = 0.0
    slc_event = survival_event + relief_event - pressure_cost_event

    return {
        "base_xg": base_xg,
        "adjusted_xs": adjusted_xs,
        "tau_inactivity": tau_inactivity,
        "tau_sequence": tau_sequence,
        "sequence_fatigue_factor": sequence_fatigue_factor,
        "lateral_pressure": lateral_pressure,
        "xg_calibration_factor": system_state.get("xg_calibration_factor", 1.0),
        "tax": tax,
        "short_handed": is_sh,
        "breakdown": breakdown,
        "slc_event": slc_event,
        "survival_event": survival_event,
        "relief_event": relief_event,
        "pressure_cost_event": pressure_cost_event,
        "result": result
    }

def analyze_progression_from_raw(raw_game_data, goalie_id):
    """
    Optimized with Pandas to eliminate loop-based time complexity.
    """
    if not raw_game_data or 'plays' not in raw_game_data:
        return None
    
    game_id = raw_game_data.get('id')
    # Load all plays into a DataFrame
    df = pd.DataFrame(raw_game_data['plays'])
    if df.empty or 'periodDescriptor' not in df.columns:
        print(f"Skipping Game ID {game_id}: play-by-play is not available yet.")
        return None

    # --- VECTORIZED PRE-PROCESSING ---
    # Extract period and period type
    df['period'] = df['periodDescriptor'].apply(lambda x: x.get('number') if isinstance(x, dict) else None)
    df['periodType'] = df['periodDescriptor'].apply(lambda x: x.get('periodType') if isinstance(x, dict) else None)
    
    # Filter out Shootouts and irrelevant periods
    df = df[(df['periodType'] != 'SO') & (df['period'] <= 4)].copy()
    if df.empty:
        print(f"Skipping Game ID {game_id}: no regulation or overtime plays found.")
        return None
    
    # Extract total seconds for time-based logic[cite: 11, 12]
    df['time_sec'] = df.apply(lambda row: get_total_seconds(row['timeInPeriod'], row['period']), axis=1)

    # --- GOALIE EVENT FILTERING ---
    # We identify plays where the goalie is the primary actor (giveaway/takeaway) 
    # or the one in net (shots/goals)
    def is_goalie_involved(row):
        details = row.get('details')
        # This check prevents the 'float' object error by ensuring details is a dictionary
        if not isinstance(details, dict): 
            return False
        return details.get('playerId') == goalie_id or details.get('goalieInNetId') == goalie_id

    df['is_goalie_action'] = df.apply(is_goalie_involved, axis=1)
    
    progression = {p: _empty_period_stats() for p in range(1, 5)}
    goalie_team_id = _infer_goalie_team_id(raw_game_data, goalie_id)
    system_state = {
        "last_defensive_event_time": 0,
        "last_turnover_zone": None,
        "in_zone_pressure": False,
        "sequence_start_time": 0,
        "sequence_shot_count": 0,
        "xg_calibration_factor": _calibrated_game_xg_factor(raw_game_data)
    }

    prev_event = None
    for _, row in df.iterrows():
        event = row.to_dict()
        details = event.get('details')
        if not isinstance(details, dict):
            prev_event = event
            continue

        event_type = event.get('typeDescKey')

        if event_type == 'stoppage':
            system_state["last_defensive_event_time"] = event.get('time_sec', system_state["last_defensive_event_time"])
            system_state["in_zone_pressure"] = False
            system_state["sequence_shot_count"] = 0
            prev_event = event
            continue

        if event_type in {'giveaway', 'takeaway'}:
            system_state["last_turnover_zone"] = details.get('zoneCode')

        if event_type not in {'shot-on-goal', 'goal'}:
            prev_event = event
            continue

        if details.get('goalieInNetId') != goalie_id:
            prev_event = event
            continue

        if details.get('xCoord') is None or details.get('yCoord') is None:
            prev_event = event
            continue

        p_num = row['period']
        stats = progression[p_num]
        is_sh = _is_goalie_team_short_handed(event.get('situationCode'), raw_game_data, goalie_team_id)
        contribution = _event_slc_contribution(event, prev_event, system_state, is_sh=is_sh)
        stats['xG'] += contribution["base_xg"]
        stats['xS'] += contribution["adjusted_xs"]
        stats['SLC_event'] += contribution["slc_event"]
        stats['Survival_Event'] += contribution["survival_event"]
        stats['Relief_Event'] += contribution["relief_event"]
        stats['Pressure_Cost_Event'] += contribution["pressure_cost_event"]
        stats['Pressure_Weight'] += contribution["tax"]
        if contribution["short_handed"]:
            stats['SH_Shots'] = stats.get('SH_Shots', 0) + 1
            stats['SH_Tax'] = stats.get('SH_Tax', 0) + (contribution["tax"] - (contribution["tax"] / 1.5))
        if contribution["breakdown"]:
            stats['Breakdowns'] = stats.get('Breakdowns', 0) + 1
        if contribution["tau_inactivity"] > 1:
            stats['Carolina_Tax'] = stats.get('Carolina_Tax', 0) + (contribution["tau_inactivity"] - 1)
        if contribution["tau_sequence"] > 0:
            stats['Brunt'] = stats.get('Brunt', 0) + 1
        if contribution["lateral_pressure"]:
            stats['Lateral_Adjustments'] = stats.get('Lateral_Adjustments', 0) + 1
        if contribution["sequence_fatigue_factor"] > 1:
            stats['Sequence_Fatigue'] = stats.get('Sequence_Fatigue', 0) + (contribution["sequence_fatigue_factor"] - 1)
        stats['xG_Calibration'] = contribution["xg_calibration_factor"]
        if not system_state["in_zone_pressure"]:
            stats['Sequences'] += 1
            system_state["sequence_start_time"] = event.get('time_sec', 0)
            system_state["in_zone_pressure"] = True
            system_state["sequence_shot_count"] = 0
        system_state["sequence_shot_count"] += 1

        system_state["last_defensive_event_time"] = event.get('time_sec', system_state["last_defensive_event_time"])
        if event_type == 'goal':
            system_state["in_zone_pressure"] = False
            system_state["sequence_shot_count"] = 0
        prev_event = event

    # --- TIME-SENSITIVE REBOUND LOGIC ---
    # Track the next puck touch after a save so clean teammate handoffs are service,
    # while pure opponent recoveries/second shots remain tax.
    possession_events = {'takeaway', 'giveaway', 'pass', 'hit', 'blocked-shot', 'missed-shot'}
    tracked_events = possession_events | {'shot-on-goal', 'goal', 'stoppage'}
    goalie_df = df[(df['is_goalie_action']) | (df['typeDescKey'].isin(tracked_events))].copy()
    last_save_time = -999
    active_window = False
    awaiting_possession = False
    teammate_touched_between = False
    last_save_is_sh = False

    for _, row in goalie_df.iterrows():
        p_num = row['period']
        stats = progression[p_num]
        event = row['typeDescKey']
        details = row.get('details', {})
        if not isinstance(details, dict):
            details = {}
        is_sh = _is_goalie_team_short_handed(row.get('situationCode'), raw_game_data, goalie_team_id)
        event_team = details.get('eventOwnerTeamId')
        is_shot_against_goalie = event in {'shot-on-goal', 'goal'} and details.get('goalieInNetId') == goalie_id

        if active_window and awaiting_possession and not is_shot_against_goalie and event != 'stoppage':
            if event in possession_events and event_team == goalie_team_id:
                stats['NPW'] += 1
                stats['Relief_Event'] += 0.80
                if last_save_is_sh:
                    stats['NPW_SH'] += 1
                    stats['Relief_Event'] += 0.20
                active_window = False
                awaiting_possession = False
                teammate_touched_between = True
                continue
            if event in possession_events and event_team is not None and event_team != goalie_team_id:
                awaiting_possession = False

        # Service & Tax logic
        if event == 'giveaway' and details.get('playerId') == goalie_id:
            if is_sh: stats['iGvA_SH'] += 1
            else: stats['iGvA'] += 1
            stats['Pressure_Cost_Event'] += 1.25 if is_sh else 0.75
        
        elif event == 'takeaway' and details.get('playerId') == goalie_id:
            stats['iTkA'] += 1
            stats['Relief_Event'] += 0.75

        elif event == 'stoppage' and active_window:
            if (row['time_sec'] - last_save_time) <= 3:
                stats['NPW'] += 1
                stats['Relief_Event'] += 1.00
                if last_save_is_sh:
                    stats['NPW_SH'] += 1
                    stats['Relief_Event'] += 0.25
            active_window = False
            awaiting_possession = False
            teammate_touched_between = False
            last_save_is_sh = False

        elif event == 'goal' and details.get('goalieInNetId') == goalie_id:
            stats['S_goals'] += 1
            active_window = False
            awaiting_possession = False
            teammate_touched_between = False
            last_save_is_sh = False

        elif event == 'shot-on-goal' and details.get('goalieInNetId') == goalie_id:
            # Rebound Detection
            if active_window and (row['time_sec'] - last_save_time) <= 3:
                if not teammate_touched_between:
                    x, y = details.get('xCoord', 0), details.get('yCoord', 0)
                    dist = math.sqrt((x - (89 if x > 0 else -89))**2 + (y - 0)**2)
                    if dist < 25:
                        stats['RP'] += 1
                        stats['Pressure_Cost_Event'] += 0.85
                        if is_sh:
                            stats['RP_SH'] += 1
                            stats['Pressure_Cost_Event'] += 0.30
                    else:
                        stats['UA'] += 1
                        stats['Pressure_Cost_Event'] += 0.45
                        if is_sh:
                            stats['UA_SH'] += 1
                            stats['Pressure_Cost_Event'] += 0.20
            
            stats['S_saves'] += 1
            last_save_time = row['time_sec']
            active_window = True
            awaiting_possession = True
            teammate_touched_between = False
            last_save_is_sh = is_sh

        # UA Timeout
        if active_window and (row['time_sec'] - last_save_time) > 5:
            current_zone = details.get('zoneCode')
            if current_zone == 'D':
                stats['UA'] += 1
                stats['Pressure_Cost_Event'] += 0.45
                if last_save_is_sh:
                    stats['UA_SH'] += 1
                    stats['Pressure_Cost_Event'] += 0.20
            else:
                stats['NPW'] += 1
                stats['Relief_Event'] += 0.80
                if last_save_is_sh:
                    stats['NPW_SH'] += 1
                    stats['Relief_Event'] += 0.20
            active_window = False
            awaiting_possession = False
            teammate_touched_between = False
            last_save_is_sh = False

    return compile_final_report(game_id, goalie_id, progression)

def compile_final_report(game_id, goalie_id, progression):
    """Summarizes stats into SLC format."""
    total_stats = _empty_period_stats()
    summaries = {}
    calibration_values = []

    for p_num, p_stats in progression.items():
        if (p_stats['S_saves'] + p_stats['S_goals']) == 0: continue
        p_stats['xG'] = round(float(p_stats.get('xG', 0)), 4)
        p_stats['xS'] = round(float(p_stats.get('xS', 0)), 4)
        p_stats['SLC_event'] = round(float(p_stats.get('SLC_event', 0)), 4)
        p_stats['Survival_Event'] = round(float(p_stats.get('Survival_Event', 0)), 4)
        p_stats['Relief_Event'] = round(float(p_stats.get('Relief_Event', 0)), 4)
        p_stats['Pressure_Cost_Event'] = round(float(p_stats.get('Pressure_Cost_Event', 0)), 4)
        p_stats['Pressure_Weight'] = round(float(p_stats.get('Pressure_Weight', 0)), 4)
        
        score, _ = calculate_slc_score(p_stats)
        service, tax = calculate_service_tax(p_stats)
        summaries[p_num] = {
            'score': score,
            'stats': p_stats,
            'tax': tax,
            'service': service
        }
        if p_stats.get('xG_Calibration'):
            calibration_values.append(float(p_stats['xG_Calibration']))
        for key, value in p_stats.items():
            if key == 'xG_Calibration':
                continue
            if isinstance(value, (int, float)):
                total_stats[key] = total_stats.get(key, 0) + value

    total_stats['xG'] = round(float(total_stats.get('xG', 0)), 4)
    total_stats['xS'] = round(float(total_stats.get('xS', 0)), 4)
    total_stats['SLC_event'] = round(float(total_stats.get('SLC_event', 0)), 4)
    total_stats['Survival_Event'] = round(float(total_stats.get('Survival_Event', 0)), 4)
    total_stats['Relief_Event'] = round(float(total_stats.get('Relief_Event', 0)), 4)
    total_stats['Pressure_Cost_Event'] = round(float(total_stats.get('Pressure_Cost_Event', 0)), 4)
    total_stats['Pressure_Weight'] = round(float(total_stats.get('Pressure_Weight', 0)), 4)
    total_stats['Sequence_Fatigue'] = round(float(total_stats.get('Sequence_Fatigue', 0)), 4)
    total_stats['xG_Calibration'] = round(sum(calibration_values) / len(calibration_values), 4) if calibration_values else 1.0
    final_score, _ = calculate_slc_score(total_stats)
    return {
        'game_id': game_id, 'goalie_id': goalie_id, 
        'periods': summaries, 'total': {'score': final_score, 'stats': total_stats}
    }

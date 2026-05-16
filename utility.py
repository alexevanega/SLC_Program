import requests # type: ignore
import os
import numpy as np
import json
import pandas as pd
from pathlib import Path

# NEW VAULT PATH
RAW_DATA_BASE_DIR = "./data/raw_games/"
RAW_DATA_DIR = "./data/raw_games/20252026/"
SCHEDULE_FILE = Path("./data/schedule/master_schedule.json")
DERIVED_STAT_FIELDS = {
    "Service_Weighted",
    "Tax_Weighted",
    "Reset_Component",
    "Survival_Component",
    "Relief_Component",
    "Pressure_Cost_Component",
    "Survival_Rate",
    "Relief_Rate",
    "Pressure_Cost_Rate",
    "Raw_SLC",
    "SLC_Confidence",
    "Opportunity_Multiplier",
}

SLC_SURVIVAL_WEIGHT = 0.25
SLC_RELIEF_WEIGHT = 0.70
SLC_PRESSURE_COST_WEIGHT = 0.55

PROFILE_SLC_WEIGHTS = {
    "Stable": {
        "survival": 0.25,
        "relief": 0.75,
        "pressure_cost": 0.0,
    },
    "Brittle": {
        "survival": 0.50,
        "relief": 0.0,
        "pressure_cost": 0.50,
    },
}


def _has_play_by_play_data(data):
    plays = data.get('plays') if isinstance(data, dict) else None
    if not plays:
        return False
    return any(isinstance(play, dict) and play.get('periodDescriptor') for play in plays)

def fetch_and_vault_raw_data(game_id):
    """
    Checks if raw game data exists. If not, fetches from NHL API and vaults it.
    This is the 'Dump' part of your logic.
    """
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
    
    file_path = os.path.join(RAW_DATA_DIR, f"{game_id}.json")
    
    # 1. Gain: Check local first
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            cached_data = json.load(f)
        if _has_play_by_play_data(cached_data):
            return cached_data
        print(f"Cached raw game {game_id} has no play-by-play yet. Refreshing from API.")
    
    # 2. Dump: Fetch and Save
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    try:
        response = requests.get(url)
        data = response.json()
        if _has_play_by_play_data(data):
            with open(file_path, 'w') as f:
                json.dump(data, f)
        else:
            print(f"Game {game_id} has no usable play-by-play yet; skipping vault write.")
        return data
    except Exception as e:
        print(f"Error vaulting raw data for {game_id}: {e}")
        return None
    
def get_total_seconds(time_str, period):
    """Converts MM:SS + period to total game seconds[cite: 3]."""
    if not time_str or ":" not in time_str:
        return 0
    m,s = map(int, time_str.split(":"))
    return (period - 1) * 1200 + (m * 60 + s)

def calculate_service_tax(stats, is_sh=False, include_baseline=False):
    """Return weighted Service and Tax with short-handed modifiers applied."""
    if is_sh:
        service = (stats.get('NPW', 0) * 1.5) + (stats.get('NPW_SH', 0) * 1.5) + stats.get('iTkA', 0)
        tax = (
            (stats.get('UA', 0) * 2.0) +
            (stats.get('UA_SH', 0) * 2.0) +
            (stats.get('RP', 0) * 3.0) +
            (stats.get('RP_SH', 0) * 3.0) +
            (stats.get('iGvA', 0) * 2.0) +
            (stats.get('iGvA_SH', 0) * 4.5)
        )
    else:
        service = stats.get('NPW', 0) + (stats.get('NPW_SH', 0) * 1.5) + stats.get('iTkA', 0)
        tax = (
            (stats.get('UA', 0) * 1.0) +
            (stats.get('UA_SH', 0) * 2.0) +
            (stats.get('RP', 0) * 2.0) +
            (stats.get('RP_SH', 0) * 3.0) +
            (stats.get('iGvA', 0) * 2.0) +
            (stats.get('iGvA_SH', 0) * 4.5)
        )

    if include_baseline:
        tax += 1
    return round(float(service), 3), round(float(tax), 3)


def _slc_weights_for_profile(team_profile=None):
    return PROFILE_SLC_WEIGHTS.get(team_profile or "", {
        "survival": SLC_SURVIVAL_WEIGHT,
        "relief": SLC_RELIEF_WEIGHT,
        "pressure_cost": SLC_PRESSURE_COST_WEIGHT,
    })


def calculate_slc_score(stats, is_sh=False, team_profile=None):
    """Core SLC math using pressure-density scoring."""
    score_stats = dict(stats or {})
    total_shots = score_stats.get('S_saves', 0) + score_stats.get('S_goals', 0)
    if total_shots == 0:
        return 0, score_stats

    service, reset_tax = calculate_service_tax(score_stats, is_sh=is_sh)
    pressure_weight = float(score_stats.get('Pressure_Weight', 0) or total_shots)
    if pressure_weight <= 0:
        return 0, score_stats

    reset_component = service / (service + reset_tax + 1)
    sequences = float(score_stats.get('Sequences', 0) or 0)
    workload_units = max(sequences, float(total_shots), 1.0)
    if all(key in score_stats for key in ["Survival_Event", "Relief_Event", "Pressure_Cost_Event"]):
        survival_component = float(score_stats.get("Survival_Event", 0))
        relief_component = float(score_stats.get("Relief_Event", 0)) + reset_component
        pressure_cost_component = float(score_stats.get("Pressure_Cost_Event", 0))
        survival_rate = survival_component / pressure_weight
        relief_rate = relief_component / workload_units
        pressure_cost_rate = pressure_cost_component / workload_units
        weights = _slc_weights_for_profile(team_profile)
        raw_slc = (
            (weights["survival"] * survival_rate) +
            (weights["relief"] * relief_rate) -
            (weights["pressure_cost"] * pressure_cost_rate)
        ) * 10
    elif 'SLC_event' in score_stats:
        survival_component = float(score_stats.get('SLC_event', 0))
        relief_component = reset_component
        pressure_cost_component = 0.0
        survival_rate = survival_component / pressure_weight
        relief_rate = relief_component / workload_units
        pressure_cost_rate = 0.0
        net_performance = score_stats.get('SLC_event', 0) + reset_component
        raw_slc = (net_performance / pressure_weight) * 10
    else:
        expected_saves = total_shots - score_stats.get('xG', 0)
        survival_component = float(score_stats.get('S_saves', 0) - expected_saves)
        relief_component = reset_component
        pressure_cost_component = 0.0
        survival_rate = survival_component / pressure_weight
        relief_rate = relief_component / workload_units
        pressure_cost_rate = 0.0
        net_performance = (score_stats.get('S_saves', 0) - expected_saves) + reset_component
        raw_slc = (net_performance / pressure_weight) * 10

    confidence = workload_units / (workload_units + 6.0)
    slc = raw_slc * confidence

    score_stats['Service_Weighted'] = service
    score_stats['Tax_Weighted'] = reset_tax
    score_stats['Reset_Component'] = round(float(reset_component), 4)
    score_stats['Survival_Component'] = round(float(survival_component), 4)
    score_stats['Relief_Component'] = round(float(relief_component), 4)
    score_stats['Pressure_Cost_Component'] = round(float(pressure_cost_component), 4)
    score_stats['Survival_Rate'] = round(float(survival_rate), 4)
    score_stats['Relief_Rate'] = round(float(relief_rate), 4)
    score_stats['Pressure_Cost_Rate'] = round(float(pressure_cost_rate), 4)
    score_stats['Raw_SLC'] = round(float(raw_slc), 3)
    score_stats['SLC_Confidence'] = round(float(confidence), 4)
    return round(float(slc), 3), score_stats


def strip_derived_stats(stats):
    """Keep extracted stat fields and remove formula/display fields."""
    return {
        key: value
        for key, value in (stats or {}).items()
        if key not in DERIVED_STAT_FIELDS
    }


def prepare_report_for_master(report):
    """Return a master-safe report containing base extracted stats only."""
    if not report:
        return report

    clean_report = {
        key: value
        for key, value in report.items()
        if key not in {"score", "service", "tax"}
    }

    clean_periods = {}
    for period, period_data in report.get("periods", {}).items():
        clean_periods[str(period)] = {
            "stats": strip_derived_stats(period_data.get("stats", {}))
        }

    clean_report["periods"] = clean_periods
    clean_report["total"] = {
        "stats": strip_derived_stats(report.get("total", {}).get("stats", {}))
    }
    return clean_report


def report_team_profile(report):
    context = report.get("team_context", {}) if isinstance(report, dict) else {}
    return context.get("volatility_profile")


def current_report_score(report):
    """Calculate current total SLC from stored master stats."""
    return calculate_slc_score(
        report.get("total", {}).get("stats", {}),
        team_profile=report_team_profile(report),
    )[0]


def current_period_score(report, period):
    """Calculate current period SLC from stored master stats."""
    stats = report.get("periods", {}).get(str(period), {}).get("stats", {})
    return calculate_slc_score(stats, team_profile=report_team_profile(report))[0]

def calculate_cumulative_slc(report_list):
    """Sums the total SLC scores across a list of game reports."""
    return round(sum(current_report_score(report) for report in report_list), 3)

def calculate_average_slc(report_list):
    """Calculates the mean SLC score across a list of game reports."""
    if not report_list: return 0
    total = sum(current_report_score(report) for report in report_list)
    return round(total / len(report_list), 3)

def get_seasonal_stats(report_list):
    """
    Returns a dictionary of the fundamental seasonal metrics.
    Aggregates Saves, Goals, and Systemic Tax.
    """
    seasonal_totals = {
        'games_played': len(report_list),
        'total_saves': sum(r['total']['stats']['S_saves'] for r in report_list),
        'total_goals': sum(r['total']['stats']['S_goals'] for r in report_list),
        'total_service': sum(calculate_service_tax(r['total']['stats'])[0] for r in report_list),
        'total_tax': sum(calculate_service_tax(r['total']['stats'])[1] for r in report_list)
    }
    return seasonal_totals

BASE_DATA_DIR = "./data/processedGames/"
MASTER_REPORT_FILE = os.path.join(BASE_DATA_DIR, "master_report.json")

def sanitize_goalie_name(goalie_name):
    """Normalize goalie names for consistent folder and master file keys."""
    return goalie_name.strip().replace(" ", "_").lower()

def load_master_reports():
    """Load the central master vault for all processed goalie reports."""
    if not os.path.exists(MASTER_REPORT_FILE):
        return {}

    with open(MASTER_REPORT_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_master_reports(master):
    """Persist the central master vault."""
    if not os.path.exists(BASE_DATA_DIR):
        os.makedirs(BASE_DATA_DIR)
    with open(MASTER_REPORT_FILE, 'w') as f:
        json.dump(master, f, indent=4)

def save_report_to_master(report, goalie_name):
    """Add a single report to the master file for fast lookup."""
    if not report:
        return None

    master = load_master_reports()
    goalie_key = sanitize_goalie_name(goalie_name)
    if goalie_key not in master:
        master[goalie_key] = {}

    master[goalie_key][str(report['game_id'])] = prepare_report_for_master(report)
    save_master_reports(master)
    return MASTER_REPORT_FILE


def get_goalie_dir(goalie_name):
    # Sanitize the name for folder compatibility
    folder_name = sanitize_goalie_name(goalie_name)
    path = os.path.join(BASE_DATA_DIR, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_report_locally(report, goalie_name):
    """Persist active goalie performances to the master report only."""
    total_shots = report['total']['stats']['S_saves'] + report['total']['stats']['S_goals']
    if total_shots <= 0:
        return None # Logic: Do not vault empty performances

    return save_report_to_master(report, goalie_name)

def load_local_report(game_id, goalie_name):
    """Checks the master report for a cached game result."""
    master = load_master_reports()
    goalie_key = sanitize_goalie_name(goalie_name)
    return master.get(goalie_key, {}).get(str(game_id))


def clear_data_caches():
    """Clear local function caches when dashboard data is updated."""
    for func_name in [
        "_load_master_reports_cached",
        "_load_schedule_map_cached",
        "_load_raw_game_cached",
        "_load_goalie_data_cached",
        "_get_team_sequence_data_cached",
        "_get_pressure_sequence_metrics_cached",
        "_build_high_interest_loan_data_cached",
        "_build_slc_hypothesis_data_cached",
        "_build_game_slc_xga_data_cached",
    ]:
        func = globals().get(func_name)
        if hasattr(func, "cache_clear"):
            func.cache_clear()


def _load_schedule_map():
    if not SCHEDULE_FILE.exists():
        return {}

    with SCHEDULE_FILE.open('r', encoding='utf-8') as f:
        return json.load(f)


def _load_raw_game(game_id):
    raw_path = os.path.join(RAW_DATA_DIR, f"{game_id}.json")
    if not os.path.exists(raw_path):
        return {}

    with open(raw_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _raw_game_exists(game_id):
    return os.path.exists(os.path.join(RAW_DATA_DIR, f"{game_id}.json"))


def _get_goalie_team_id(raw_game, goalie_id):
    for player in raw_game.get('rosterSpots', []):
        if player.get('playerId') == goalie_id:
            return player.get('teamId')
    return None


def _get_goalie_team_abbrev(raw_game, goalie_id):
    goalie_team_id = _get_goalie_team_id(raw_game, goalie_id)
    for team_key in ['homeTeam', 'awayTeam']:
        team = raw_game.get(team_key, {})
        if team.get('id') == goalie_team_id:
            return team.get('abbrev')
    return None


def get_available_teams():
    """Return team abbreviations found in the local schedule."""
    teams = set()
    for game in _load_schedule_map().values():
        matchup = game.get('matchup', '')
        if ' @ ' not in matchup:
            continue
        away, home = matchup.split(' @ ', 1)
        teams.add(away)
        teams.add(home)
    return sorted(teams)


def get_team_games(team_abbrev):
    """Return locally vaulted games for a team, newest first."""
    games = []
    for game_id, game in _load_schedule_map().items():
        matchup = game.get('matchup', '')
        if f"{team_abbrev} @ " not in matchup and f" @ {team_abbrev}" not in matchup:
            continue
        if not _raw_game_exists(game_id):
            continue
        games.append({
            "Game_ID": str(game_id),
            "Date": game.get('date', '0000-00-00'),
            "Matchup": matchup,
            "Label": f"{game.get('date', '0000-00-00')} {matchup}"
        })
    return sorted(games, key=lambda item: item["Date"], reverse=True)


def _iter_non_shootout_plays(raw_game):
    for event in raw_game.get('plays', []):
        period_desc = event.get('periodDescriptor', {})
        period_type = period_desc.get('periodType') or period_desc.get('type')
        period_number = period_desc.get('number', 1)
        if period_type == "SO" or period_number > 4:
            continue
        yield event


def _get_game_duration_minutes(raw_game):
    max_seconds = 0
    for event in _iter_non_shootout_plays(raw_game):
        period = event.get('periodDescriptor', {}).get('number', 1)
        max_seconds = max(max_seconds, get_total_seconds(event.get('timeInPeriod'), period))

    return max(max_seconds / 60.0, 60.0)


def _get_team_id(raw_game, team_abbrev):
    for team_key in ['homeTeam', 'awayTeam']:
        team = raw_game.get(team_key, {})
        if team.get('abbrev') == team_abbrev:
            return team.get('id')
    return None


def get_team_sequence_data(team_abbrev, game_id, phase="All"):
    """
    Estimate defensive-zone sequence durations for a selected team.

    A sequence starts on opponent shot pressure and ends on a whistle, goal, or
    selected-team neutral/offensive-zone touch. This is a local play-by-play proxy
    for defensive-zone entries.
    """
    raw_game = _load_raw_game(game_id)
    team_id = _get_team_id(raw_game, team_abbrev)
    if not raw_game or not team_id:
        return pd.DataFrame()

    rows = []
    active = None
    last_opponent_shot_time = None
    opponent_pressure_events = {'shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'}
    ending_stoppages = {'stoppage', 'period-end'}

    for event in _iter_non_shootout_plays(raw_game):
        event_type = event.get('typeDescKey', '').lower()
        details = event.get('details', {})
        owner_id = details.get('eventOwnerTeamId')
        zone_code = details.get('zoneCode')
        period = event.get('periodDescriptor', {}).get('number', 1)
        event_time = get_total_seconds(event.get('timeInPeriod'), period)

        if phase == "3rd Period" and period != 3:
            continue

        if active is not None and active["period"] != period:
            active["end"] = (active["period"] * 1200)
            active["end_reason"] = "period-end"
            rows.append(active)
            active = None
            last_opponent_shot_time = None

        is_opponent_pressure = owner_id and owner_id != team_id and event_type in opponent_pressure_events
        if is_opponent_pressure:
            if active is None:
                active = {
                    "start": event_time,
                    "end": event_time,
                    "period": period,
                    "stress": False,
                    "stress_reasons": set(),
                    "end_reason": ""
                }
            active["end"] = event_time

            if event_type in {'shot-on-goal', 'goal'}:
                if last_opponent_shot_time is not None and (event_time - last_opponent_shot_time) <= 3:
                    active["stress"] = True
                    active["stress_reasons"].add("RP")
                last_opponent_shot_time = event_time

            if event_type == 'goal':
                active["end_reason"] = "goal"
                rows.append(active)
                active = None
                last_opponent_shot_time = None
            continue

        if active is None:
            continue

        if event_type == 'giveaway' and owner_id == team_id and zone_code == 'D':
            active["stress"] = True
            active["stress_reasons"].add("iGvA")

        is_clearance = owner_id == team_id and zone_code in {'N', 'O'}
        if event_type in ending_stoppages or is_clearance:
            active["end"] = event_time
            active["end_reason"] = "clearance" if is_clearance else event_type
            duration = max(0, active["end"] - active["start"])
            if duration > 40:
                active["stress"] = True
                active["stress_reasons"].add("UA")
            rows.append(active)
            active = None
            last_opponent_shot_time = None

    if active is not None:
        duration = max(0, active["end"] - active["start"])
        if duration > 40:
            active["stress"] = True
            active["stress_reasons"].add("UA")
        rows.append(active)

    sequence_rows = []
    for sequence in rows:
        duration = max(0, sequence["end"] - sequence["start"])
        sequence_rows.append({
            "Duration": round(duration, 2),
            "Period": sequence["period"],
            "Stress": sequence["stress"],
            "Stress_Reasons": ", ".join(sorted(sequence["stress_reasons"])) if sequence["stress_reasons"] else "",
            "Bin_Start": int(duration // 5) * 5
        })

    return pd.DataFrame(sequence_rows)


def get_team_sequence_histogram(team_abbrev, game_id, phase="All", include_average=False):
    """Build histogram rows and summary metrics for the team tab."""
    current = get_team_sequence_data(team_abbrev, game_id, phase)
    if current.empty:
        return current, pd.DataFrame(), {"Efficiency": 0.0, "Median": 0.0, "Total": 0}

    total = len(current)
    under_40 = len(current[current["Duration"] < 40])
    metrics = {
        "Efficiency": round((under_40 / total) * 100, 1) if total else 0.0,
        "Median": round(float(current["Duration"].median()), 1) if total else 0.0,
        "Total": total
    }

    average = pd.DataFrame()
    if include_average:
        team_games = get_team_games(team_abbrev)
        frames = [
            get_team_sequence_data(team_abbrev, game["Game_ID"], phase)
            for game in team_games
            if game["Game_ID"] != str(game_id)
        ]
        frames = [frame for frame in frames if not frame.empty]
        if frames:
            average = pd.concat(frames, ignore_index=True)

    return current, average, metrics


def get_pressure_sequence_metrics(game_id, goalie_id):
    """
    Estimate defensive stress sequences from play-by-play.

    NHL play-by-play does not expose defensive-zone entries, so this uses shot-pressure
    windows against the goalie: first shot/miss/goal against until stoppage, goal, or
    period boundary.
    """
    raw_game = _load_raw_game(game_id)
    if not raw_game:
        return {
            "Avg_Sequence_Duration": 0.0,
            "Red_Line_Shifts": 0,
            "xGA_Per_60": 0.0,
            "High_Danger_Per_60": 0.0
        }

    sequences = []
    high_danger_chances = 0
    total_xga = 0.0
    active_start = None
    active_period = None
    prev_event = None

    for event in _iter_non_shootout_plays(raw_game):
        event_type = event.get('typeDescKey', '').lower()
        period = event.get('periodDescriptor', {}).get('number', 1)
        event_time = get_total_seconds(event.get('timeInPeriod'), period)
        details = event.get('details', {})

        if active_start is not None and active_period != period:
            sequences.append(max(0, ((active_period or 1) * 1200) - active_start))
            active_start = None
            active_period = None

        is_goalie_shot_event = (
            event_type in {'shot-on-goal', 'missed-shot', 'goal'} and
            details.get('goalieInNetId') == goalie_id
        )

        if is_goalie_shot_event:
            if active_start is None:
                active_start = event_time
                active_period = period

            shot_xg = calculate_shot_probability(event, prev_event)
            total_xga += shot_xg
            if shot_xg >= 0.08:
                high_danger_chances += 1

            if event_type == 'goal':
                sequences.append(max(0, event_time - active_start))
                active_start = None
                active_period = None

        elif event_type == 'stoppage' and active_start is not None:
            sequences.append(max(0, event_time - active_start))
            active_start = None
            active_period = None

        prev_event = event

    if active_start is not None:
        final_second = max(
            [get_total_seconds(e.get('timeInPeriod'), e.get('periodDescriptor', {}).get('number', 1)) for e in _iter_non_shootout_plays(raw_game)] or [active_start]
        )
        sequences.append(max(0, final_second - active_start))

    game_minutes = _get_game_duration_minutes(raw_game)
    red_line_shifts = sum(1 for duration in sequences if duration > 40)
    avg_sequence_duration = sum(sequences) / len(sequences) if sequences else 0.0

    return {
        "Avg_Sequence_Duration": round(avg_sequence_duration, 2),
        "Red_Line_Shifts": red_line_shifts,
        "xGA_Per_60": round((total_xga / game_minutes) * 60.0, 3),
        "High_Danger_Per_60": round((high_danger_chances / game_minutes) * 60.0, 3)
    }


def get_goalie_matchup_label(game_id, goalie_id, schedule_info=None):
    """Return a goalie-relative matchup label such as 'vs BOS' or 'at TOR'."""
    raw_game = _load_raw_game(game_id)
    home_team = raw_game.get('homeTeam', {})
    away_team = raw_game.get('awayTeam', {})
    goalie_team_id = _get_goalie_team_id(raw_game, goalie_id)

    if goalie_team_id and home_team and away_team:
        if goalie_team_id == home_team.get('id'):
            return f"vs {away_team.get('abbrev', 'UNK')}"
        if goalie_team_id == away_team.get('id'):
            return f"at {home_team.get('abbrev', 'UNK')}"

    if schedule_info and schedule_info.get('matchup'):
        return schedule_info['matchup']

    return str(game_id)


def load_goalie_data(active_goalie):
    """Loads one goalie's game history from the master report."""
    master_map = _load_schedule_map()
    master_report = load_master_reports()
    all_games = []

    for data in master_report.get(active_goalie.lower(), {}).values():
        g_id = str(data.get('game_id'))
        goalie_id = data.get('goalie_id')
        g_info = master_map.get(g_id, {})
        raw_game = _load_raw_game(g_id)
        total_stats = data.get('total', {}).get('stats', {})
        shots = total_stats.get('S_saves', 0) + total_stats.get('S_goals', 0)
        service, tax = calculate_service_tax(total_stats, include_baseline=True)
        expected_saves = total_stats.get('xS', 0) or shots - total_stats.get('xG', 0)
        expected_sv = expected_saves / shots if shots > 0 else 0
        actual_sv = total_stats.get('S_saves', 0) / shots if shots > 0 else 0
        all_games.append({
            "Date": g_info.get('date', data.get('gameDate') or raw_game.get('gameDate', "0000-00-00")),
            "Matchup": get_goalie_matchup_label(g_id, goalie_id, g_info),
            "Game_ID": g_id,
            "SLC": current_report_score(data),
            "Sovereignty": round((actual_sv - expected_sv) * 100, 3) if shots > 0 else 0,
            "Net_Load": round(service - tax, 3),
            "Service": service,
            "Tax": round(tax, 3),
            "NPW": total_stats.get('NPW', 0),
            "UA": total_stats.get('UA', 0),
            "RP": total_stats.get('RP', 0)
        })

    if not all_games:
        return pd.DataFrame()

    return pd.DataFrame(all_games).sort_values("Date")


def _get_period_stats(report, period):
    return report.get('periods', {}).get(str(period), {}).get('stats', {})


def _period_tax(stats):
    return calculate_service_tax(stats, include_baseline=True)[1]


def _period_tax_base(stats):
    return max(_period_tax(stats) - 1, 0)


def _combine_stats(stats_list):
    combined = {}
    for stats in stats_list:
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                combined[key] = combined.get(key, 0) + value
    return combined


def _calculate_sovereignty(stats):
    shots = stats.get('S_saves', 0) + stats.get('S_goals', 0)
    if shots <= 0:
        return 0.0

    expected_saves = stats.get('xS', 0) or shots - stats.get('xG', 0)
    expected_sv = expected_saves / shots
    actual_sv = stats.get('S_saves', 0) / shots
    return round((actual_sv - expected_sv) * 100, 3)


def _build_single_loan_row(goalie_key, game_id, schedule_info=None):
    report = load_master_reports().get(goalie_key.lower(), {}).get(str(game_id), {})
    if not report:
        return {}

    raw_game = _load_raw_game(game_id)
    goalie_id = report.get('goalie_id')
    team_abbrev = _get_goalie_team_abbrev(raw_game, goalie_id)
    sequence_df = get_team_sequence_data(team_abbrev, game_id) if team_abbrev else pd.DataFrame()
    early_sequences = sequence_df[sequence_df["Period"].isin([1, 2])] if not sequence_df.empty else pd.DataFrame()

    p1_stats = _get_period_stats(report, 1)
    p2_stats = _get_period_stats(report, 2)
    p3_stats = _get_period_stats(report, 3)
    early_stats = _combine_stats([p1_stats, p2_stats])

    early_red_line = int((early_sequences["Duration"] > 40).sum()) if not early_sequences.empty else 0
    early_red_line_seconds = (
        float((early_sequences.loc[early_sequences["Duration"] > 40, "Duration"] - 40).sum())
        if not early_sequences.empty
        else 0.0
    )
    early_tax = _period_tax_base(early_stats)
    debt_score = early_tax + early_red_line
    early_service = calculate_service_tax(early_stats)[0]
    early_denominator = early_stats.get('UA', 0) + early_stats.get('RP', 0) + early_red_line
    early_reset_ratio = early_service / early_denominator if early_denominator else 0

    p3_ga = p3_stats.get('S_goals', 0)
    p3_sovereignty = _calculate_sovereignty(p3_stats)
    p3_tax = _period_tax_base(p3_stats)
    p3_slc = current_period_score(report, 3)

    date_value = report.get('gameDate') or raw_game.get('gameDate')
    matchup = get_goalie_matchup_label(game_id, goalie_id, schedule_info)
    if schedule_info:
        date_value = schedule_info.get('date', date_value)

    return {
        "Game_ID": str(game_id),
        "Date": date_value or "0000-00-00",
        "Matchup": matchup,
        "Goalie_ID": goalie_id,
        "Team": team_abbrev or "",
        "Early_Debt": round(debt_score, 3),
        "Early_Tax": round(early_tax, 3),
        "Early_UA": early_stats.get('UA', 0),
        "Early_RP": early_stats.get('RP', 0),
        "Early_iGvA": early_stats.get('iGvA', 0),
        "Early_iGvA_SH": early_stats.get('iGvA_SH', 0),
        "Early_Red_Line": early_red_line,
        "Early_Red_Line_Seconds": round(early_red_line_seconds, 1),
        "Early_Reset_Ratio": round(early_reset_ratio, 3),
        "Early_Service": early_service,
        "P3_Sovereignty": p3_sovereignty,
        "P3_GA": p3_ga,
        "P3_Tax": round(p3_tax, 3),
        "P3_SLC": round(p3_slc, 3),
        "Debt_Label": "High-Interest Loan" if debt_score >= 4 else "Managed Load"
    }


def build_high_interest_loan_data(goalie_key, selected_game_id):
    """Build selected-game and season context data for the 3rd-period debt view."""
    master_report = load_master_reports().get(goalie_key.lower(), {})
    if not master_report:
        return {}, pd.DataFrame(), pd.DataFrame()

    schedule_map = _load_schedule_map()
    rows = []
    for game_id in master_report:
        row = _build_single_loan_row(goalie_key, game_id, schedule_map.get(str(game_id), {}))
        if row:
            rows.append(row)

    season_df = pd.DataFrame(rows)
    if season_df.empty:
        return {}, season_df, pd.DataFrame()

    season_df = season_df.sort_values("Date")
    selected_rows = season_df[season_df["Game_ID"].astype(str) == str(selected_game_id)]
    selected_row = selected_rows.iloc[0].to_dict() if not selected_rows.empty else {}

    components_df = pd.DataFrame()
    if selected_row:
        components_df = pd.DataFrame([
            {"Component": "UA", "Value": selected_row["Early_UA"]},
            {"Component": "RP x2", "Value": selected_row["Early_RP"] * 2},
            {"Component": "Giveaways x2", "Value": selected_row["Early_iGvA"] * 2},
            {"Component": "SH Giveaways x3", "Value": selected_row["Early_iGvA_SH"] * 3},
            {"Component": "40s+ Sequences", "Value": selected_row["Early_Red_Line"]},
        ])

    return selected_row, season_df, components_df


def build_slc_hypothesis_data(goalie_key, goalie_name=None):
    """Build per-game data for the comparative SLC load hypothesis visual."""
    base_df = load_goalie_data(goalie_key)
    if base_df.empty:
        return base_df

    master_report = load_master_reports()
    rows = []

    for _, row in base_df.iterrows():
        report = master_report.get(goalie_key.lower(), {}).get(str(row["Game_ID"]), {})
        total_stats = report.get('total', {}).get('stats', {})
        goalie_id = report.get('goalie_id')
        sequences = total_stats.get('Sequences', 0)
        if sequences > 0:
            reset_ratio = total_stats.get('S_saves', 0) / sequences
        else:
            defensive_load = total_stats.get('UA', 0) + total_stats.get('RP', 0)
            reset_ratio = total_stats.get('NPW', 0) / defensive_load if defensive_load > 0 else 0
        sequence_metrics = get_pressure_sequence_metrics(row["Game_ID"], goalie_id)

        p3_stats = _get_period_stats(report, 3)
        early_scores = [
            current_period_score(report, period)
            for period in [1, 2]
            if report.get('periods', {}).get(str(period), {}).get('stats')
        ]
        early_slc = sum(early_scores) / len(early_scores) if early_scores else 0
        p3_hardware_failures = (
            p3_stats.get('UA', 0) +
            (p3_stats.get('RP', 0) * 2) +
            p3_stats.get('S_goals', 0)
        )

        enriched = row.to_dict()
        enriched.update({
            "Goalie": goalie_name or goalie_key.replace("_", " ").title(),
            "Rolling_SLC": 0.0,
            "Reset_Ratio": round(reset_ratio, 3),
            "Early_SLC": round(early_slc, 3),
            "P3_Hardware_Failures": p3_hardware_failures,
            "P3_Tax": round(_period_tax(p3_stats), 3),
            **sequence_metrics
        })
        rows.append(enriched)

    df = pd.DataFrame(rows).sort_values("Date")
    df["Rolling_SLC"] = df["SLC"].rolling(window=5, min_periods=1).mean().round(3)
    df["Pressure_Ratio"] = df.apply(
        lambda row: round(row["SLC"] / row["xGA_Per_60"], 3) if row["xGA_Per_60"] else 0,
        axis=1
    )
    return df


def build_game_slc_xga_data(goalie_key, game_id):
    """Build period-level SLC and xGA/60 for a selected goalie game."""
    report = load_master_reports().get(goalie_key.lower(), {}).get(str(game_id), {})
    if not report:
        return pd.DataFrame()

    raw_game = _load_raw_game(game_id)
    game_type = raw_game.get('gameType')
    rows = []

    for period, period_data in sorted(report.get('periods', {}).items(), key=lambda item: int(item[0])):
        stats = period_data.get('stats', {})
        period_number = int(period)
        period_minutes = 5 if period_number == 4 and game_type == 2 else 20
        xga_per_60 = (stats.get('xG', 0) / period_minutes) * 60 if period_minutes else 0
        rows.append({
            "Period": f"P{period}",
            "SLC": current_period_score(report, period),
            "xGA_Per_60": round(xga_per_60, 3)
        })

    return pd.DataFrame(rows)


def _late_xga(report):
    p3_stats = _get_period_stats(report, 3)
    ot_stats = _get_period_stats(report, 4)
    return float(p3_stats.get("xG", 0) or 0) + float(ot_stats.get("xG", 0) or 0)


def _proof_safe_div(numerator, denominator):
    return round(float(numerator) / float(denominator), 3) if denominator else 0.0


def _goalie_game_research_rows():
    """Build temporary goalie-game rows from master_report for proof analysis."""
    master_report = load_master_reports()
    rows = []

    for goalie_key, games in master_report.items():
        for game_id, report in games.items():
            stats = report.get("total", {}).get("stats", {})
            raw_game = _load_raw_game(game_id)
            date_value = report.get("gameDate") or raw_game.get("gameDate")
            sequences = stats.get("Sequences", 0)
            slc, scored_stats = calculate_slc_score(stats, team_profile=report_team_profile(report))
            service, tax = calculate_service_tax(stats, include_baseline=False)

            rows.append({
                "Goalie_Key": goalie_key,
                "Game_ID": str(game_id),
                "Date": pd.to_datetime(date_value, errors="coerce"),
                "Team": _get_goalie_team_abbrev(raw_game, report.get("goalie_id")) or "",
                "Volatility_Profile": report_team_profile(report) or "Unknown",
                "SLC": slc,
                "Sovereignty_Rate": float(scored_stats.get("Survival_Rate", 0) or 0),
                "Relief_Rate": float(scored_stats.get("Relief_Rate", 0) or 0),
                "Pressure_Cost_Rate": float(scored_stats.get("Pressure_Cost_Rate", 0) or 0),
                "Pressure_Efficiency": _proof_safe_div(slc, sequences),
                "Net_Load_per_Sequence": _proof_safe_div(service - tax, sequences),
                "Sequences": sequences,
                "Late_xGA": _late_xga(report),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["Goalie_Key", "Date", "Game_ID"]).reset_index(drop=True)


def _attach_team_recent_late_xga_baseline(df):
    if df.empty or "Team" not in df.columns:
        return df

    team_games = (
        df[df["Team"].astype(bool)]
        .groupby(["Team", "Game_ID", "Date"], as_index=False)
        .agg({"Late_xGA": "sum"})
        .sort_values(["Team", "Date", "Game_ID"])
    )
    if team_games.empty:
        df["Team_Late_xGA_Roll5"] = np.nan
        df["Team_Relative_Collapse"] = np.nan
        df["Team_Rest_Days"] = np.nan
        df["Team_Games_Last_7"] = np.nan
        return df

    frames = []
    for _, group in team_games.groupby("Team"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)
        group["Team_Late_xGA_Roll5"] = group["Late_xGA"].shift(1).rolling(5, min_periods=3).mean()
        group["Team_Rest_Days"] = group["Date"].diff().dt.days.clip(lower=0, upper=10)
        group["Team_Games_Last_7"] = [
            int((group.loc[: i - 1, "Date"] >= row["Date"] - pd.Timedelta(days=7)).sum())
            for i, row in group.iterrows()
        ]
        group["Team_Relative_Collapse"] = np.where(
            group["Team_Late_xGA_Roll5"].notna(),
            (group["Late_xGA"] >= group["Team_Late_xGA_Roll5"] + 0.35).astype(int),
            np.nan,
        )
        frames.append(group)

    team_context = pd.concat(frames, ignore_index=True)[[
        "Team",
        "Game_ID",
        "Team_Late_xGA_Roll5",
        "Team_Relative_Collapse",
        "Team_Rest_Days",
        "Team_Games_Last_7",
    ]]
    return df.merge(team_context, on=["Team", "Game_ID"], how="left")


def _attach_team_volatility_profiles(df):
    if df.empty or "Team" not in df.columns:
        return df

    team_games = (
        df[df["Team"].astype(bool)]
        .groupby(["Team", "Game_ID", "Date"], as_index=False)
        .agg({"Late_xGA": "sum"})
        .sort_values(["Team", "Date", "Game_ID"])
    )
    if team_games.empty:
        df["Volatility_Profile"] = "Unknown"
        return df

    team_rows = []
    for team, group in team_games.groupby("Team"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)
        rolling_avg = group["Late_xGA"].shift(1).rolling(5, min_periods=3).mean()
        rolling_fluctuation = group["Late_xGA"].shift(1).rolling(5, min_periods=3).std()
        season_fluctuation = float(group["Late_xGA"].std()) if len(group) > 1 else 0.0
        typical_fluctuation = rolling_fluctuation.fillna(season_fluctuation).fillna(0.0)
        spike_threshold = rolling_avg + typical_fluctuation
        usable = rolling_avg.notna()
        spike_rate = float((group.loc[usable, "Late_xGA"] > spike_threshold.loc[usable]).mean()) if usable.any() else 0.0
        team_rows.append({
            "Team": team,
            "Team_Fluctuation": season_fluctuation,
            "Team_Spike_Rate": spike_rate,
        })

    profiles = pd.DataFrame(team_rows)
    fluctuation_line = profiles["Team_Fluctuation"].median()
    spike_line = profiles["Team_Spike_Rate"].median()

    def label_profile(row):
        high_fluctuation = row["Team_Fluctuation"] >= fluctuation_line
        high_spike = row["Team_Spike_Rate"] >= spike_line
        if not high_fluctuation and not high_spike:
            return "Stable"
        if not high_fluctuation and high_spike:
            return "Brittle"
        if high_fluctuation and not high_spike:
            return "Controlled Chaos"
        return "Volatile"

    profiles["Volatility_Profile"] = profiles.apply(label_profile, axis=1)
    return df.merge(profiles, on="Team", how="left")


def build_team_context_by_game(master_report=None):
    """Build rolling team volatility context keyed by (team, game_id)."""
    master_report = master_report if master_report is not None else load_master_reports()
    rows = []
    for games in master_report.values():
        for game_id, report in games.items():
            raw_game = _load_raw_game(game_id)
            team = _get_goalie_team_abbrev(raw_game, report.get("goalie_id"))
            date_value = report.get("gameDate") or raw_game.get("gameDate")
            if not team:
                continue
            rows.append({
                "Team": team,
                "Game_ID": str(game_id),
                "Date": pd.to_datetime(date_value, errors="coerce"),
                "Late_xGA": _late_xga(report),
            })

    if not rows:
        return {}

    team_games = (
        pd.DataFrame(rows)
        .dropna(subset=["Date"])
        .groupby(["Team", "Game_ID", "Date"], as_index=False)
        .agg({"Late_xGA": "sum"})
        .sort_values(["Team", "Date", "Game_ID"])
    )
    if team_games.empty:
        return {}

    frames = []
    for _, group in team_games.groupby("Team"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)
        group["Rolling_Avg"] = group["Late_xGA"].shift(1).rolling(5, min_periods=3).mean()
        group["Typical_Fluctuation"] = group["Late_xGA"].shift(1).rolling(5, min_periods=3).std()
        group["Spike_Threshold"] = group["Rolling_Avg"] + group["Typical_Fluctuation"]
        group["Spike"] = np.where(
            group["Rolling_Avg"].notna(),
            group["Late_xGA"] > group["Spike_Threshold"],
            np.nan,
        )
        group["Spike_Rate"] = group["Spike"].shift(1).expanding(min_periods=3).mean()
        frames.append(group)

    context = pd.concat(frames, ignore_index=True)
    fluctuation_line = context["Typical_Fluctuation"].median()
    spike_line = context["Spike_Rate"].median()

    def label_profile(row):
        if pd.isna(row["Typical_Fluctuation"]) or pd.isna(row["Spike_Rate"]):
            return "Unknown"
        high_fluctuation = row["Typical_Fluctuation"] >= fluctuation_line
        high_spike = row["Spike_Rate"] >= spike_line
        if not high_fluctuation and not high_spike:
            return "Stable"
        if not high_fluctuation and high_spike:
            return "Brittle"
        if high_fluctuation and not high_spike:
            return "Controlled Chaos"
        return "Volatile"

    context["Volatility_Profile"] = context.apply(label_profile, axis=1)
    context_map = {}
    for _, row in context.iterrows():
        context_map[(row["Team"], str(row["Game_ID"]))] = {
            "team": row["Team"],
            "volatility_profile": row["Volatility_Profile"],
            "late_xga_rolling_avg": round(float(row["Rolling_Avg"]), 4) if pd.notna(row["Rolling_Avg"]) else None,
            "typical_fluctuation": round(float(row["Typical_Fluctuation"]), 4) if pd.notna(row["Typical_Fluctuation"]) else None,
            "spike_rate": round(float(row["Spike_Rate"]), 4) if pd.notna(row["Spike_Rate"]) else None,
        }
    return context_map


def attach_team_context_to_master(master_report=None):
    """Attach rolling team context to each stored report."""
    master_report = master_report if master_report is not None else load_master_reports()
    context_map = build_team_context_by_game(master_report)
    updated = 0
    for games in master_report.values():
        for game_id, report in games.items():
            raw_game = _load_raw_game(game_id)
            team = _get_goalie_team_abbrev(raw_game, report.get("goalie_id"))
            context = context_map.get((team, str(game_id))) if team else None
            if not context:
                continue
            if report.get("team_context") != context:
                report["team_context"] = context
                updated += 1
    return master_report, updated


def build_late_xga_collapse_proof():
    """
    Test whether low recent SLC is associated with more future late-xGA collapses.

    This is intentionally in-memory: master_report.json remains the source of truth.
    """
    df = _goalie_game_research_rows()
    if df.empty:
        return {
            "summary": {},
            "risk_by_schedule": pd.DataFrame(),
            "risk_by_feature": pd.DataFrame(),
        }

    df = _attach_team_recent_late_xga_baseline(df)
    if "Volatility_Profile" not in df.columns or df["Volatility_Profile"].eq("Unknown").all():
        df = _attach_team_volatility_profiles(df)
    frames = []
    for _, group in df.groupby("Goalie_Key"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)

        group["Rest_Days"] = group["Team_Rest_Days"]
        group["Games_Last_7"] = group["Team_Games_Last_7"]
        group["Schedule_Context"] = group.apply(
            lambda row: "Stressed"
            if row["Games_Last_7"] >= 3 or (pd.notna(row["Rest_Days"]) and row["Rest_Days"] <= 1)
            else "Normal",
            axis=1,
        )
        group["SLC_Roll3"] = group["SLC"].shift(1).rolling(3, min_periods=1).mean()
        group["SLC_Roll5"] = group["SLC"].shift(1).rolling(5, min_periods=1).mean()
        group["Sovereignty_Rate_Roll3"] = group["Sovereignty_Rate"].shift(1).rolling(3, min_periods=1).mean()
        group["Relief_Rate_Roll3"] = group["Relief_Rate"].shift(1).rolling(3, min_periods=1).mean()
        group["Pressure_Cost_Rate_Roll3"] = group["Pressure_Cost_Rate"].shift(1).rolling(3, min_periods=1).mean()
        group["PE_Roll3"] = group["Pressure_Efficiency"].shift(1).rolling(3, min_periods=1).mean()
        group["Late_xGA_Roll5"] = group["Team_Late_xGA_Roll5"]
        group["Sequences_Roll3"] = group["Sequences"].shift(1).rolling(3, min_periods=1).mean()

        for collapse_col in ["Team_Relative_Collapse"]:
            values = group[collapse_col].astype(float).to_numpy()
            for horizon in [1, 3, 5]:
                future = []
                for i in range(len(group)):
                    start = i + 1
                    end = i + 1 + horizon
                    window = values[start:end]
                    if end <= len(group) and len(window) and not np.all(np.isnan(window)):
                        future.append(float(np.nanmax(window)))
                    else:
                        future.append(np.nan)
                group[f"Future_{collapse_col}_H{horizon}"] = future

        frames.append(group)

    proof_df = pd.concat(frames, ignore_index=True)

    risk_rows = []
    for target in [
        "Future_Team_Relative_Collapse_H1",
        "Future_Team_Relative_Collapse_H3",
        "Future_Team_Relative_Collapse_H5",
    ]:
        for context, subset in proof_df.groupby("Schedule_Context"):
            use = subset[["SLC_Roll3", target]].dropna()
            if len(use) < 50:
                continue

            use = use.copy()
            use["SLC_Bucket"] = pd.qcut(use["SLC_Roll3"], 3, labels=["Low", "Mid", "High"], duplicates="drop")
            risks = use.groupby("SLC_Bucket", observed=True)[target].mean()
            counts = use.groupby("SLC_Bucket", observed=True)[target].count()
            if not {"Low", "High"}.issubset(set(risks.index)):
                continue

            risk_rows.append({
                "Target": target,
                "Schedule_Context": context,
                "Low_SLC_Risk": round(float(risks.loc["Low"]), 3),
                "High_SLC_Risk": round(float(risks.loc["High"]), 3),
                "Low_Minus_High": round(float(risks.loc["Low"] - risks.loc["High"]), 3),
                "Low_N": int(counts.loc["Low"]),
                "High_N": int(counts.loc["High"]),
                "N": int(len(use)),
            })

    feature_rows = []
    for feature in ["SLC_Roll3", "SLC_Roll5", "PE_Roll3"]:
        use = proof_df[[feature, "Future_Team_Relative_Collapse_H1"]].dropna()
        if len(use) < 50:
            continue
        corr = np.corrcoef(use[feature], use["Future_Team_Relative_Collapse_H1"])[0, 1]
        feature_rows.append({
            "Feature": feature,
            "Target": "Future_Team_Relative_Collapse_H1",
            "Pearson": round(float(corr), 3),
            "R2": round(float(corr * corr), 4),
            "N": int(len(use)),
        })

    risk_by_schedule = pd.DataFrame(risk_rows)
    risk_by_feature = pd.DataFrame(feature_rows)
    model_lift = _build_collapse_model_lift(proof_df)
    model_lift_by_profile = _build_collapse_model_lift_by_profile(proof_df)
    profile_weight_optimizer = _build_profile_weight_optimizer(proof_df)
    profile_weight_validation = _build_profile_weight_validation(proof_df)
    summary = {
        "Goalie_Games": int(len(proof_df)),
        "Usable_Next_Game_Rows": int(proof_df["Future_Team_Relative_Collapse_H1"].notna().sum()),
        "Team_Baseline_Window": 5,
        "Team_Baseline_Buffer": 0.35,
    }

    return {
        "summary": summary,
        "risk_by_schedule": risk_by_schedule,
        "risk_by_feature": risk_by_feature,
        "model_lift": model_lift,
        "model_lift_by_profile": model_lift_by_profile,
        "profile_weight_optimizer": profile_weight_optimizer,
        "profile_weight_validation": profile_weight_validation,
        "same_game": _build_same_game_heatsink_proof(),
    }


def _period_pair_stats(report, periods):
    return _combine_stats([
        _get_period_stats(report, period)
        for period in periods
    ])


def _build_same_game_heatsink_proof():
    rows = []
    master_report = load_master_reports()

    for goalie_key, games in master_report.items():
        for game_id, report in games.items():
            early_stats = _period_pair_stats(report, [1, 2])
            p3_stats = _get_period_stats(report, 3)
            early_shots = early_stats.get("S_saves", 0) + early_stats.get("S_goals", 0)
            p3_shots = p3_stats.get("S_saves", 0) + p3_stats.get("S_goals", 0)
            if early_shots <= 0 or p3_shots <= 0:
                continue

            early_slc, scored_early = calculate_slc_score(early_stats)
            early_sequences = early_stats.get("Sequences", 0)
            early_service, early_tax = calculate_service_tax(early_stats, include_baseline=False)
            early_relief = _proof_safe_div(early_service - early_tax, early_sequences)
            early_relief_rate = float(scored_early.get("Relief_Rate", 0) or 0)
            early_cost_rate = float(scored_early.get("Pressure_Cost_Rate", 0) or 0)
            early_pressure = float(early_stats.get("Pressure_Weight", 0) or early_shots)
            p3_xga = float(p3_stats.get("xG", 0) or 0)

            rows.append({
                "Goalie_Key": goalie_key,
                "Game_ID": str(game_id),
                "Date": pd.to_datetime(report.get("gameDate"), errors="coerce"),
                "Early_SLC": early_slc,
                "Early_Relief": early_relief,
                "Early_Relief_Rate": early_relief_rate,
                "Early_Cost_Rate": early_cost_rate,
                "Early_Net_Load": early_service - early_tax,
                "Early_Pressure": early_pressure,
                "Early_Sequences": early_sequences,
                "P3_xGA": p3_xga,
            })

    if not rows:
        return {
            "summary": {},
            "risk_by_early_slc": pd.DataFrame(),
            "lift": pd.DataFrame(),
        }

    df = pd.DataFrame(rows)
    p3_threshold = df["P3_xGA"].quantile(0.85)
    df["P3_Collapse"] = (df["P3_xGA"] >= p3_threshold).astype(int)
    df["Early_Pressure_Context"] = np.where(
        df["Early_Pressure"] >= df["Early_Pressure"].median(),
        "High early pressure",
        "Lower early pressure",
    )

    risk_rows = []
    for context, subset in df.groupby("Early_Pressure_Context"):
        use = subset[["Early_SLC", "Early_Relief_Rate", "P3_Collapse"]].dropna()
        if len(use) < 100:
            continue

        for feature, label in [
            ("Early_SLC", "Early SLC"),
            ("Early_Relief_Rate", "Early relief rate"),
        ]:
            feature_use = use[[feature, "P3_Collapse"]].copy()
            if feature_use[feature].nunique() < 5:
                continue
            feature_use["Bucket"] = pd.qcut(
                feature_use[feature],
                3,
                labels=["Low", "Middle", "High"],
                duplicates="drop",
            )
            rates = feature_use.groupby("Bucket", observed=True)["P3_Collapse"].mean()
            counts = feature_use.groupby("Bucket", observed=True)["P3_Collapse"].count()
            if not {"Low", "High"}.issubset(set(rates.index)):
                continue

            gap = rates.loc["Low"] - rates.loc["High"]
            risk_rows.append({
                "Context": context,
                "Signal": label,
                "Low_Risk": round(float(rates.loc["Low"]), 3),
                "High_Risk": round(float(rates.loc["High"]), 3),
                "Low_Minus_High": round(float(gap), 3),
                "Low_N": int(counts.loc["Low"]),
                "High_N": int(counts.loc["High"]),
                "N": int(len(feature_use)),
            })

    lift_rows = []
    split_date = df["Date"].quantile(0.70)
    train_df = df[df["Date"] <= split_date]
    test_df = df[df["Date"] > split_date]
    model_specs = {
        "Early Pressure Only": ["Early_Pressure", "Early_Sequences"],
        "Early Pressure + SLC": ["Early_Pressure", "Early_Sequences", "Early_SLC"],
        "Early Pressure + Relief Rate": ["Early_Pressure", "Early_Sequences", "Early_Relief_Rate"],
    }
    previous = None
    for model_name, features in model_specs.items():
        model = _fit_linear_risk_model(train_df, "P3_Collapse", features)
        preds = _predict_linear_risk(model, test_df)
        added_signal_direction = ""
        if model is not None and model_name == "Early Pressure + SLC":
            slc_coef = model["coef"][1 + features.index("Early_SLC")]
            added_signal_direction = "Higher early SLC raised predicted risk" if slc_coef > 0 else "Higher early SLC lowered predicted risk"
        if model is not None and model_name == "Early Pressure + Relief Rate":
            relief_coef = model["coef"][1 + features.index("Early_Relief_Rate")]
            added_signal_direction = "Higher early relief rate raised predicted risk" if relief_coef > 0 else "Higher early relief rate lowered predicted risk"
        use = pd.DataFrame({
            "P3_Collapse": test_df["P3_Collapse"],
            "Predicted_Risk": preds,
        }).dropna()
        if len(use) < 100 or use["Predicted_Risk"].nunique() < 3:
            continue
        use["Risk_Bucket"] = pd.qcut(use["Predicted_Risk"], 3, labels=False, duplicates="drop")
        rates = use.groupby("Risk_Bucket", observed=True)["P3_Collapse"].mean()
        counts = use.groupby("Risk_Bucket", observed=True)["P3_Collapse"].count()
        if len(rates) < 2:
            continue
        low_bucket = rates.index.min()
        high_bucket = rates.index.max()
        separation = float(rates.loc[high_bucket] - rates.loc[low_bucket])
        added_lift = np.nan if previous is None else separation - previous
        previous = separation
        lift_rows.append({
            "Model": model_name,
            "Low_Predicted_Risk_Rate": round(float(rates.loc[low_bucket]), 3),
            "High_Predicted_Risk_Rate": round(float(rates.loc[high_bucket]), 3),
            "Risk_Separation": round(separation, 3),
            "Added_Lift_vs_Previous": round(float(added_lift), 3) if pd.notna(added_lift) else np.nan,
            "Added_Signal_Direction": added_signal_direction,
            "Low_N": int(counts.loc[low_bucket]),
            "High_N": int(counts.loc[high_bucket]),
            "N": int(len(use)),
        })

    return {
        "summary": {
            "Games": int(len(df)),
            "P3_xGA_Top15_Threshold": round(float(p3_threshold), 3),
        },
        "risk_by_early_slc": pd.DataFrame(risk_rows),
        "lift": pd.DataFrame(lift_rows),
    }


def _rank_feature(series, ascending=True):
    ranked = series.rank(pct=True, ascending=ascending)
    return ranked.fillna(ranked.median()).fillna(0.5)


def _fit_linear_risk_model(train_df, target_col, feature_cols):
    train = train_df[[target_col] + feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < len(feature_cols) + 20 or train[target_col].nunique() < 2:
        return None

    means = train[feature_cols].mean()
    stds = train[feature_cols].std().replace(0, 1).fillna(1)
    x = (train[feature_cols] - means) / stds
    matrix = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    y = train[target_col].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    return {"coef": coef, "means": means, "stds": stds, "features": feature_cols}


def _predict_linear_risk(model, df):
    if model is None:
        return pd.Series(np.nan, index=df.index)

    features = model["features"]
    x = df[features].replace([np.inf, -np.inf], np.nan)
    valid = x.notna().all(axis=1)
    preds = pd.Series(np.nan, index=df.index)
    if not valid.any():
        return preds

    scaled = (x.loc[valid] - model["means"]) / model["stds"]
    matrix = np.column_stack([np.ones(len(scaled)), scaled.to_numpy(dtype=float)])
    preds.loc[valid] = np.clip(matrix @ model["coef"], 0, 1)
    return preds


def _build_collapse_model_lift(proof_df):
    rows = []
    targets = {
        "Future_Team_Relative_Collapse_H1": "Above team recent baseline, next game",
    }
    model_specs = {
        "Schedule Only": ["Rest_Days", "Games_Last_7"],
        "Schedule + Recent Load": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3"],
        "Schedule + Load + SLC": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "SLC_Roll3"],
        "Schedule + Load + Relief Rate": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "Relief_Rate_Roll3"],
        "Schedule + Load + SLC + Relief Rate": [
            "Rest_Days",
            "Games_Last_7",
            "Late_xGA_Roll5",
            "Sequences_Roll3",
            "SLC_Roll3",
            "Relief_Rate_Roll3",
        ],
        "Schedule + Load + SLC + Relief + Cost": [
            "Rest_Days",
            "Games_Last_7",
            "Late_xGA_Roll5",
            "Sequences_Roll3",
            "SLC_Roll3",
            "Relief_Rate_Roll3",
            "Pressure_Cost_Rate_Roll3",
        ],
    }

    proof_df = proof_df.sort_values("Date").copy()
    split_date = proof_df["Date"].quantile(0.70)
    train_df = proof_df[proof_df["Date"] <= split_date]
    test_df = proof_df[proof_df["Date"] > split_date]

    for schedule_context in ["All", "Normal", "Stressed"]:
        context_df = test_df if schedule_context == "All" else test_df[test_df["Schedule_Context"] == schedule_context]
        if context_df.empty:
            continue

        for target_col, target_label in targets.items():
            previous_separation = None
            load_baseline_separation = None
            for model_name, feature_cols in model_specs.items():
                model = _fit_linear_risk_model(train_df, target_col, feature_cols)
                predictions = _predict_linear_risk(model, context_df)
                model_use = pd.DataFrame({
                    target_col: context_df[target_col],
                    "Predicted_Risk": predictions,
                }).dropna()
                if len(model_use) < 100 or model_use["Predicted_Risk"].nunique() < 3:
                    continue

                model_use = model_use.copy()
                model_use["Risk_Bucket"] = pd.qcut(
                    model_use["Predicted_Risk"],
                    3,
                    labels=False,
                    duplicates="drop",
                )
                rates = model_use.groupby("Risk_Bucket", observed=True)[target_col].mean()
                counts = model_use.groupby("Risk_Bucket", observed=True)[target_col].count()
                if len(rates) < 2:
                    continue

                low_bucket = rates.index.min()
                high_bucket = rates.index.max()
                low_rate = float(rates.loc[low_bucket])
                high_rate = float(rates.loc[high_bucket])
                separation = high_rate - low_rate
                added_lift = np.nan if previous_separation is None else separation - previous_separation
                previous_separation = separation
                if model_name == "Schedule + Recent Load":
                    load_baseline_separation = separation
                lift_vs_load = (
                    separation - load_baseline_separation
                    if load_baseline_separation is not None and model_name != "Schedule + Recent Load"
                    else np.nan
                )

                rows.append({
                    "Target": target_label,
                    "Schedule_Context": schedule_context,
                    "Model": model_name,
                    "Low_Predicted_Risk_Rate": round(low_rate, 3),
                    "High_Predicted_Risk_Rate": round(high_rate, 3),
                    "Risk_Separation": round(separation, 3),
                    "Added_Lift_vs_Previous": round(float(added_lift), 3) if pd.notna(added_lift) else np.nan,
                    "Lift_vs_Load_Baseline": round(float(lift_vs_load), 3) if pd.notna(lift_vs_load) else np.nan,
                    "Low_N": int(counts.loc[low_bucket]),
                    "High_N": int(counts.loc[high_bucket]),
                    "N": int(len(model_use)),
                })

    return pd.DataFrame(rows)


def _build_collapse_model_lift_by_profile(proof_df):
    if "Volatility_Profile" not in proof_df.columns:
        return pd.DataFrame()

    rows = []
    target_col = "Future_Team_Relative_Collapse_H1"
    target_label = "Above team recent baseline, next game"
    model_specs = {
        "Schedule + Recent Load": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3"],
        "Schedule + Recent Load + SLC": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "SLC_Roll3"],
        "Schedule + Recent Load + Sovereignty": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "Sovereignty_Rate_Roll3"],
        "Schedule + Recent Load + Relief Rate": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "Relief_Rate_Roll3"],
        "Schedule + Recent Load + Pressure": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "Pressure_Cost_Rate_Roll3"],
    }

    proof_df = proof_df.sort_values("Date").copy()
    split_date = proof_df["Date"].quantile(0.70)
    train_df = proof_df[proof_df["Date"] <= split_date]
    test_df = proof_df[proof_df["Date"] > split_date]

    for profile, profile_test_df in test_df.groupby("Volatility_Profile"):
        if not profile or profile == "Unknown":
            continue

        profile_train_df = train_df[train_df["Volatility_Profile"] == profile]
        if profile_train_df.empty:
            continue

        load_baseline_separation = None
        for model_name, feature_cols in model_specs.items():
            model = _fit_linear_risk_model(profile_train_df, target_col, feature_cols)
            predictions = _predict_linear_risk(model, profile_test_df)
            model_use = pd.DataFrame({
                target_col: profile_test_df[target_col],
                "Predicted_Risk": predictions,
            }).dropna()
            if len(model_use) < 50 or model_use["Predicted_Risk"].nunique() < 3:
                continue

            model_use = model_use.copy()
            model_use["Risk_Bucket"] = pd.qcut(
                model_use["Predicted_Risk"],
                3,
                labels=False,
                duplicates="drop",
            )
            rates = model_use.groupby("Risk_Bucket", observed=True)[target_col].mean()
            counts = model_use.groupby("Risk_Bucket", observed=True)[target_col].count()
            if len(rates) < 2:
                continue

            low_bucket = rates.index.min()
            high_bucket = rates.index.max()
            low_rate = float(rates.loc[low_bucket])
            high_rate = float(rates.loc[high_bucket])
            separation = high_rate - low_rate
            if model_name == "Schedule + Recent Load":
                load_baseline_separation = separation
            lift_vs_load = (
                separation - load_baseline_separation
                if load_baseline_separation is not None and model_name != "Schedule + Recent Load"
                else np.nan
            )

            rows.append({
                "Volatility_Profile": profile,
                "Target": target_label,
                "Model": model_name,
                "Low_Predicted_Risk_Rate": round(low_rate, 3),
                "High_Predicted_Risk_Rate": round(high_rate, 3),
                "Risk_Separation": round(separation, 3),
                "Lift_vs_Load_Baseline": round(float(lift_vs_load), 3) if pd.notna(lift_vs_load) else np.nan,
                "Low_N": int(counts.loc[low_bucket]),
                "High_N": int(counts.loc[high_bucket]),
                "N": int(len(model_use)),
            })

    return pd.DataFrame(rows)


def _risk_separation_for_features(train_df, test_df, target_col, feature_cols):
    model = _fit_linear_risk_model(train_df, target_col, feature_cols)
    predictions = _predict_linear_risk(model, test_df)
    model_use = pd.DataFrame({
        target_col: test_df[target_col],
        "Predicted_Risk": predictions,
    }).dropna()
    if len(model_use) < 50 or model_use["Predicted_Risk"].nunique() < 3:
        return None

    model_use = model_use.copy()
    model_use["Risk_Bucket"] = pd.qcut(
        model_use["Predicted_Risk"],
        3,
        labels=False,
        duplicates="drop",
    )
    rates = model_use.groupby("Risk_Bucket", observed=True)[target_col].mean()
    counts = model_use.groupby("Risk_Bucket", observed=True)[target_col].count()
    if len(rates) < 2:
        return None

    low_bucket = rates.index.min()
    high_bucket = rates.index.max()
    return {
        "low_rate": float(rates.loc[low_bucket]),
        "high_rate": float(rates.loc[high_bucket]),
        "separation": float(rates.loc[high_bucket] - rates.loc[low_bucket]),
        "low_n": int(counts.loc[low_bucket]),
        "high_n": int(counts.loc[high_bucket]),
        "n": int(len(model_use)),
    }


def _build_profile_weight_optimizer(proof_df):
    if "Volatility_Profile" not in proof_df.columns:
        return pd.DataFrame()

    target_col = "Future_Team_Relative_Collapse_H1"
    base_features = ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3"]
    rows = []

    proof_df = proof_df.sort_values("Date").copy()
    split_date = proof_df["Date"].quantile(0.70)
    train_df = proof_df[proof_df["Date"] <= split_date]
    test_df = proof_df[proof_df["Date"] > split_date]

    for profile, profile_test_df in test_df.groupby("Volatility_Profile"):
        if not profile or profile == "Unknown":
            continue

        profile_train_df = train_df[train_df["Volatility_Profile"] == profile].copy()
        profile_test_df = profile_test_df.copy()
        if profile_train_df.empty or profile_test_df.empty:
            continue

        best = _best_profile_weights(
            profile_train_df,
            profile_test_df,
            target_col,
            base_features,
        )
        if not best:
            continue

        rows.append({
            "Volatility_Profile": profile,
            "Sovereignty_Weight": best["sovereignty_weight"],
            "Relief_Weight": best["relief_weight"],
            "Pressure_Weight": best["pressure_weight"],
            "Base_Separation": round(best["base_separation"], 3),
            "Optimized_SLC_Separation": round(best["optimized_separation"], 3),
            "Lift_vs_Load_Baseline": round(best["lift"], 3),
            "Final_Search_Step": best["final_step"],
            "Low_N": best["low_n"],
            "High_N": best["high_n"],
            "N": best["n"],
        })

    return pd.DataFrame(rows)


def _weight_candidates(center=None, step=0.1, radius=None):
    values = [round(i * step, 4) for i in range(int(round(1 / step)) + 1)]
    candidates = []
    for sovereignty_weight in values:
        for relief_weight in values:
            pressure_weight = round(1.0 - sovereignty_weight - relief_weight, 4)
            if pressure_weight < -0.0001 or pressure_weight > 1.0001:
                continue
            pressure_weight = round(max(0.0, min(1.0, pressure_weight)), 4)
            if center and radius is not None:
                if abs(sovereignty_weight - center["sovereignty_weight"]) > radius:
                    continue
                if abs(relief_weight - center["relief_weight"]) > radius:
                    continue
                if abs(pressure_weight - center["pressure_weight"]) > radius:
                    continue
            candidates.append((sovereignty_weight, relief_weight, pressure_weight))
    return candidates


def _weighted_slc_series(df, sovereignty_weight, relief_weight, pressure_weight):
    return (
        (df["Sovereignty_Rate_Roll3"] * sovereignty_weight) +
        (df["Relief_Rate_Roll3"] * relief_weight) -
        (df["Pressure_Cost_Rate_Roll3"] * pressure_weight)
    )


def _best_profile_weights(tune_train_df, tune_test_df, target_col, base_features):
    base_result = _risk_separation_for_features(
        tune_train_df,
        tune_test_df,
        target_col,
        base_features,
    )
    if not base_result:
        return None

    best = None
    search_plan = [
        {"step": 0.10, "radius": None},
        {"step": 0.05, "radius": 0.10},
        {"step": 0.01, "radius": 0.05},
    ]

    for search in search_plan:
        pass_best = None
        for sovereignty_weight, relief_weight, pressure_weight in _weight_candidates(
            center=best,
            step=search["step"],
            radius=search["radius"],
        ):
            train_use = tune_train_df.copy()
            test_use = tune_test_df.copy()
            train_use["Optimized_SLC_Roll3"] = _weighted_slc_series(
                train_use,
                sovereignty_weight,
                relief_weight,
                pressure_weight,
            )
            test_use["Optimized_SLC_Roll3"] = _weighted_slc_series(
                test_use,
                sovereignty_weight,
                relief_weight,
                pressure_weight,
            )
            result = _risk_separation_for_features(
                train_use,
                test_use,
                target_col,
                base_features + ["Optimized_SLC_Roll3"],
            )
            if not result:
                continue

            lift = result["separation"] - base_result["separation"]
            candidate = {
                "sovereignty_weight": sovereignty_weight,
                "relief_weight": relief_weight,
                "pressure_weight": pressure_weight,
                "base_separation": base_result["separation"],
                "optimized_separation": result["separation"],
                "lift": lift,
                "final_step": search["step"],
                "low_n": result["low_n"],
                "high_n": result["high_n"],
                "n": result["n"],
            }
            if (
                pass_best is None or
                candidate["lift"] > pass_best["lift"] or
                (
                    candidate["lift"] == pass_best["lift"] and
                    candidate["optimized_separation"] > pass_best["optimized_separation"]
                )
            ):
                pass_best = candidate

        if pass_best:
            best = pass_best

    return best


def _build_profile_weight_validation(proof_df):
    if "Volatility_Profile" not in proof_df.columns:
        return pd.DataFrame()

    target_col = "Future_Team_Relative_Collapse_H1"
    base_features = ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3"]
    rows = []

    proof_df = proof_df.sort_values("Date").copy()
    outer_split = proof_df["Date"].quantile(0.70)
    train_df = proof_df[proof_df["Date"] <= outer_split]
    holdout_df = proof_df[proof_df["Date"] > outer_split]

    for profile, profile_holdout_df in holdout_df.groupby("Volatility_Profile"):
        if not profile or profile == "Unknown":
            continue

        profile_train_df = train_df[train_df["Volatility_Profile"] == profile].sort_values("Date").copy()
        profile_holdout_df = profile_holdout_df.sort_values("Date").copy()
        if len(profile_train_df) < 100 or len(profile_holdout_df) < 50:
            continue

        inner_split = profile_train_df["Date"].quantile(0.70)
        tune_train_df = profile_train_df[profile_train_df["Date"] <= inner_split]
        tune_test_df = profile_train_df[profile_train_df["Date"] > inner_split]
        if len(tune_train_df) < 50 or len(tune_test_df) < 50:
            continue

        best = _best_profile_weights(
            tune_train_df,
            tune_test_df,
            target_col,
            base_features,
        )
        if not best:
            continue

        base_holdout = _risk_separation_for_features(
            profile_train_df,
            profile_holdout_df,
            target_col,
            base_features,
        )
        if not base_holdout:
            continue

        train_use = profile_train_df.copy()
        holdout_use = profile_holdout_df.copy()
        train_use["Validated_SLC_Roll3"] = _weighted_slc_series(
            train_use,
            best["sovereignty_weight"],
            best["relief_weight"],
            best["pressure_weight"],
        )
        holdout_use["Validated_SLC_Roll3"] = _weighted_slc_series(
            holdout_use,
            best["sovereignty_weight"],
            best["relief_weight"],
            best["pressure_weight"],
        )
        holdout_result = _risk_separation_for_features(
            train_use,
            holdout_use,
            target_col,
            base_features + ["Validated_SLC_Roll3"],
        )
        if not holdout_result:
            continue

        rows.append({
            "Volatility_Profile": profile,
            "Sovereignty_Weight": best["sovereignty_weight"],
            "Relief_Weight": best["relief_weight"],
            "Pressure_Weight": best["pressure_weight"],
            "Tuning_Base_Separation": round(best["base_separation"], 3),
            "Tuning_SLC_Separation": round(best["optimized_separation"], 3),
            "Tuning_Lift": round(best["lift"], 3),
            "Holdout_Base_Separation": round(base_holdout["separation"], 3),
            "Holdout_SLC_Separation": round(holdout_result["separation"], 3),
            "Holdout_Lift": round(holdout_result["separation"] - base_holdout["separation"], 3),
            "N": holdout_result["n"],
        })

    return pd.DataFrame(rows)


def get_filtered_stats(goalie_key, start_date, end_date):
    """Filters goalie performance across a date window."""
    df = load_goalie_data(goalie_key)
    if df.empty:
        return df

    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
    return df.loc[mask]

BASELINE_VAULT = "./data/league_baselines.json"

def gain_local_baseline(game_date):
    """Retrieves a pre-calculated baseline from the local ledger."""
    if os.path.exists(BASELINE_VAULT):
        with open(BASELINE_VAULT, 'r') as f:
            vault = json.load(f)
            return vault.get(game_date)
    return None

def dump_local_baseline(game_date, value):
    """Saves a new baseline calculation to the local ledger."""
    vault = {}
    if os.path.exists(BASELINE_VAULT):
        with open(BASELINE_VAULT, 'r') as f:
            vault = json.load(f)
    
    vault[game_date] = value
    with open(BASELINE_VAULT, 'w') as f:
        json.dump(vault, f, indent=4)

def _normalize_shot_type(raw_type):
    if not raw_type:
        return "Wrist"

    normalized = raw_type.strip().replace("-", " ").title()
    return normalized

def _parse_time_in_period(time_str):
    if not time_str:
        return 0.0
    try:
        minutes, seconds = time_str.split(":")
        return float(minutes) * 60.0 + float(seconds)
    except Exception:
        return 0.0

def calculate_shot_probability(event, prev_event):
    """
    Calculates xG for a single raw shot event from a play-by-play file.
    """
    details = event.get('details', {})
    if not isinstance(details, dict):
        details = {}
    shot_x = abs(details.get('xCoord', 0))
    shot_y = details.get('yCoord', 0)
    distance = np.sqrt((89 - shot_x) ** 2 + shot_y ** 2)
    angle = np.abs(np.degrees(np.arctan2(shot_y, 89 - shot_x)))

    shot_types = {
        "Deflected": 0.22,
        "Tip In": 0.19,
        "Backhand": 0.14,
        "Snap": 0.11,
        "Wrist": 0.09,
        "Slap": 0.07,
        "Wrap Around": 0.15
    }
    shot_type = _normalize_shot_type(details.get('shotType', 'Wrist'))
    base_xg = shot_types.get(shot_type, 0.10)

    lateral_bonus = 1.0
    if prev_event and prev_event.get('typeDescKey') == 'pass':
        prev_details = prev_event.get('details', {})
        if not isinstance(prev_details, dict):
            prev_details = {}
        prev_y = prev_details.get('yCoord', 0)
        if (prev_y * shot_y < 0) or (abs(prev_y - shot_y) > 15):
            time_delta = _parse_time_in_period(event.get('timeInPeriod', '0:00')) - _parse_time_in_period(prev_event.get('timeInPeriod', '0:00'))
            if 0 <= time_delta < 2.0:
                lateral_bonus = 1.75

    rebound_multiplier = 1.0
    if prev_event and prev_event.get('typeDescKey') in {'shot-on-goal', 'goal'}:
        current_goalie = details.get('goalieInNetId')
        prev_details = prev_event.get('details', {})
        if not isinstance(prev_details, dict):
            prev_details = {}
        prev_goalie = prev_details.get('goalieInNetId')
        time_delta = _parse_time_in_period(event.get('timeInPeriod', '0:00')) - _parse_time_in_period(prev_event.get('timeInPeriod', '0:00'))
        if current_goalie and prev_goalie and current_goalie == prev_goalie and 0 <= time_delta <= 3.0:
            rebound_multiplier = 2.25

    distance_weight = np.exp(-0.022 * distance)  # Softened decay
    angle_weight = np.cos(np.radians(angle * 0.4)) # Widened danger zone

    xg = base_xg * distance_weight * angle_weight * lateral_bonus * rebound_multiplier
    return float(min(round(float(xg), 4), 0.999))


def apply_xg_to_report(report, raw_game_data, goalie_id=None):
    """Attach period and total xG to a single goalie report."""
    if not report or not raw_game_data:
        return report

    target_goalie_id = goalie_id or report.get('goalie_id')
    if not target_goalie_id:
        return report

    period_totals = {}
    prev_event = None

    for period, period_data in list(report.get('periods', {}).items()):
        stats = period_data.get('stats', {})
        stats.pop('xG', None)
        if set(period_data.keys()) == {'stats'} and not stats:
            report['periods'].pop(period, None)

    for event in raw_game_data.get('plays', []):
        period_desc = event.get('periodDescriptor', {})
        period_type = period_desc.get('periodType') or period_desc.get('type')
        period_number = period_desc.get('number', 1)
        type_key = event.get('typeDescKey', '').lower()

        if period_type == "SO" or period_number > 4:
            prev_event = event
            continue

        if type_key not in {'shot-on-goal', 'missed-shot', 'goal'}:
            prev_event = event
            continue

        details = event.get('details', {})
        if details.get('goalieInNetId') != target_goalie_id:
            prev_event = event
            continue

        if details.get('xCoord') is None or details.get('yCoord') is None:
            prev_event = event
            continue

        period = str(period_number)
        period_totals[period] = period_totals.get(period, 0.0) + calculate_shot_probability(event, prev_event)
        prev_event = event

    total_game_xg = 0.0
    report.setdefault('periods', {})
    for period, xg_value in period_totals.items():
        report['periods'].setdefault(period, {'stats': {}})
        report['periods'][period].setdefault('stats', {})
        report['periods'][period]['stats']['xG'] = round(xg_value, 4)
        total_game_xg += xg_value

    report.setdefault('total', {'stats': {}})
    report['total'].setdefault('stats', {})
    report['total']['stats']['xG'] = round(total_game_xg, 4)
    return recalculate_report_scores(report)


def recalculate_report_scores(report):
    """Re-score a report after its stats, including xG, are present."""
    if not report:
        return report

    total_stats = {}
    calibration_values = []

    for period_data in report.get('periods', {}).values():
        stats = period_data.get('stats', {})
        if not stats:
            continue

        stats['xG'] = round(float(stats.get('xG', 0)), 4)
        stats['xS'] = round(float(stats.get('xS', 0)), 4)
        stats['SLC_event'] = round(float(stats.get('SLC_event', 0)), 4)
        shots = stats.get('S_saves', 0) + stats.get('S_goals', 0)
        if shots > 0:
            period_data['score'], _ = calculate_slc_score(stats)
            period_data['service'], period_data['tax'] = calculate_service_tax(stats)

        if stats.get('xG_Calibration'):
            calibration_values.append(float(stats['xG_Calibration']))

        for key, value in stats.items():
            if key == 'xG_Calibration':
                continue
            if isinstance(value, (int, float)):
                total_stats[key] = total_stats.get(key, 0) + value

    total_stats['xG'] = round(float(total_stats.get('xG', 0)), 4)
    total_stats['xS'] = round(float(total_stats.get('xS', 0)), 4)
    total_stats['SLC_event'] = round(float(total_stats.get('SLC_event', 0)), 4)
    total_stats['Sequence_Fatigue'] = round(float(total_stats.get('Sequence_Fatigue', 0)), 4)
    total_stats['xG_Calibration'] = round(sum(calibration_values) / len(calibration_values), 4) if calibration_values else 1.0
    report['total'] = {
        'score': calculate_slc_score(total_stats)[0],
        'stats': total_stats
    }
    return report


def run_xg_scrubber():
    raw_dir = './data/raw_games'
    report_path = './data/processedGames/master_report.json'

    with open(report_path, 'r', encoding='utf-8') as f:
        master_report = json.load(f)

    for goalie_games in master_report.values():
        for game_id, game_entry in goalie_games.items():
            raw_path = os.path.join(raw_dir, f'{game_id}.json')
            if not os.path.exists(raw_path):
                continue

            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_game = json.load(f)

            apply_xg_to_report(game_entry, raw_game)
            goalie_games[game_id] = prepare_report_for_master(game_entry)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(master_report, f, indent=4)

    print('Master Report updated with xG stats.')

# if __name__ == '__main__':
#     run_xg_scrubber()

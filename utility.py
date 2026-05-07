import requests # type: ignore
import os
import numpy as np
import json
import pandas as pd
from pathlib import Path

# NEW VAULT PATH
RAW_DATA_DIR = "./data/raw_games/"
SCHEDULE_FILE = Path("./data/schedule/master_schedule.json")


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
        service = (stats.get('NPW', 0) * 1.5) + stats.get('iTkA', 0)
        tax = (
            (stats.get('UA', 0) * 2.0) +
            (stats.get('RP', 0) * 3.0) +
            (stats.get('iGvA', 0) * 2.0) +
            (stats.get('iGvA_SH', 0) * 4.5)
        )
    else:
        npw_sh = stats.get('NPW_SH', 0)
        ua_sh = stats.get('UA_SH', 0)
        rp_sh = stats.get('RP_SH', 0)
        npw_ev = max(stats.get('NPW', 0) - npw_sh, 0)
        ua_ev = max(stats.get('UA', 0) - ua_sh, 0)
        rp_ev = max(stats.get('RP', 0) - rp_sh, 0)
        service = npw_ev + (npw_sh * 1.5) + stats.get('iTkA', 0)
        tax = (
            (ua_ev * 1.0) +
            (ua_sh * 2.0) +
            (rp_ev * 2.0) +
            (rp_sh * 3.0) +
            (stats.get('iGvA', 0) * 2.0) +
            (stats.get('iGvA_SH', 0) * 4.5)
        )

    if include_baseline:
        tax += 1
    return round(float(service), 3), round(float(tax), 3)


def calculate_slc_score(stats, is_sh=False):
    """Core SLC math using event-level environmental tax contributions."""
    total_shots = stats['S_saves'] + stats['S_goals']
    if total_shots == 0: return 0, stats

    service, tax = calculate_service_tax(stats, is_sh=is_sh)
    stats['Service_Weighted'] = service
    stats['Tax_Weighted'] = tax
    reset_component = service / (tax + 1)
    stats['Reset_Component'] = round(float(reset_component), 4)

    if 'SLC_event' in stats:
        slc = stats.get('SLC_event', 0) + reset_component
    else:
        expected_saves = total_shots - stats.get('xG', 0)
        slc = (stats.get('S_saves', 0) - expected_saves) + reset_component
    return round(float(slc), 3), stats

def calculate_cumulative_slc(report_list):
    """Sums the total SLC scores across a list of game reports."""
    return round(sum(report['total']['score'] for report in report_list), 3)

def calculate_average_slc(report_list):
    """Calculates the mean SLC score across a list of game reports."""
    if not report_list: return 0
    total = sum(report['total']['score'] for report in report_list)
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

    master[goalie_key][str(report['game_id'])] = report
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
            "SLC": float(data.get('total', {}).get('score', 0)),
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
    p3_slc = float(report.get('periods', {}).get('3', {}).get('score', 0))

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

        p1 = report.get('periods', {}).get('1', {})
        p2 = report.get('periods', {}).get('2', {})
        p3_stats = _get_period_stats(report, 3)
        early_scores = [p.get('score') for p in [p1, p2] if p.get('score') is not None]
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
            "SLC": float(period_data.get('score', 0)),
            "xGA_Per_60": round(xga_per_60, 3)
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

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(master_report, f, indent=4)

    print('Master Report updated with xG stats.')

# if __name__ == '__main__':
#     run_xg_scrubber()

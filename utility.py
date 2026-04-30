import requests # type: ignore
import os
import json

# NEW VAULT PATH
RAW_DATA_DIR = "./data/raw_games/"

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
            return json.load(f)
    
    # 2. Dump: Fetch and Save
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    try:
        response = requests.get(url)
        data = response.json()
        with open(file_path, 'w') as f:
            json.dump(data, f)
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

def calculate_slc_score(stats, xS_basline):
    """Core SLC Math Logic[cite: 3]."""
    total_shots = stats['S_saves'] + stats['S_goals']
    if total_shots == 0: return 0, stats

    # S (Sovereignty): Actual SV% - xSV%[cite: 3]
    actual_sv = stats['S_saves'] / total_shots
    S = actual_sv - xS_basline
    S_contribution = S * 100

    # Numerator (Service) / Denominator (Physical Tax)[cite: 3]
    numerator = stats['NPW'] + stats['iTkA']
    denominator = (stats['UA'] * 1.0) + (stats['RP'] * 2.0) + (stats['iGvA'] * 2.0) + (stats['iGvA_SH'] * 3.0) + 1

    slc = S_contribution + (numerator / denominator)
    return round(slc, 3), stats

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
        'total_service': sum(r['total']['stats']['NPW'] + r['total']['stats']['iTkA'] for r in report_list),
        'total_tax': sum(
            (r['total']['stats']['UA'] * 1.0) + 
            (r['total']['stats']['RP'] * 2.0) + 
            (r['total']['stats']['iGvA'] * 2.0) + 
            (r['total']['stats']['iGvA_SH'] * 3.0) 
            for r in report_list
        )
    }
    return seasonal_totals

BASE_DATA_DIR = "./data/processedGames/"

def get_goalie_dir(goalie_name):
    # Sanitize the name for folder compatibility
    folder_name = goalie_name.replace(" ", "_").lower()
    path = os.path.join(BASE_DATA_DIR, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_report_locally(report, goalie_name):
    """Only dumps the report if the goalie actually saw shots."""
    total_shots = report['total']['stats']['S_saves'] + report['total']['stats']['S_goals']
    if total_shots <= 0:
        return None # Logic: Do not vault empty performances

    goalie_dir = get_goalie_dir(goalie_name)
    file_path = os.path.join(goalie_dir, f"{report['game_id']}.json")
    with open(file_path, 'w') as f:
        json.dump(report, f, indent=4)
    return file_path

def load_local_report(game_id, goalie_name):
    """Checks the goalie's specific folder for a game result."""
    goalie_dir = get_goalie_dir(goalie_name)
    file_path = os.path.join(goalie_dir, f"{game_id}.json")
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

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
import requests
import json
import urllib.parse
from utility import gain_local_baseline, dump_local_baseline

TEAM_MAP = {
    1: "NJD", 2: "NYI", 3: "NYR", 4: "PHI", 5: "PIT", 6: "BOS", 
    7: "BUF", 8: "MTL", 9: "OTT", 10: "TOR", 12: "CAR", 13: "FLA", 
    14: "TBL", 15: "WSH", 16: "CHI", 17: "DET", 18: "NSH", 19: "STL", 
    20: "CGY", 21: "COL", 22: "EDM", 23: "VAN", 24: "ANA", 25: "DAL", 
    26: "LAK", 28: "SJS", 29: "CBJ", 30: "MIN", 52: "WPG", 53: "ARI", 54: "VGK", 
    55: "SEA", 68: "UTA"
}

def resolve_goalie_id(goalie_name):
    """
    Uses the Stats API to find a specific goalie ID.
    """
    # Encoding the filter to handle spaces/special characters
    # Logic: lastName='Bussi' and firstName='Brandon'
    first_name = goalie_name.split(" ")[0]
    last_name = goalie_name.split(" ")[1]
    expression = f"lastName='{last_name}' and firstName='{first_name}'"
    encoded_exp = urllib.parse.quote(expression)
    
    url = f"https://api.nhle.com/stats/rest/en/players?cayenneExp={encoded_exp}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data and 'data' in data and len(data['data']) > 0:
            # The ID in this API is usually 'id' or 'playerId'
            # Let's grab the first match
            return data['data'][0]['id']
    except Exception as e:
        print(f"Error resolving Goalie {first_name} {last_name}: {e}")
    return None

import requests
import urllib.parse

def resolve_goalie_and_team(goalie_name):
    # Standard name split
    parts = goalie_name.split(" ")
    first_name, last_name = parts[0], parts[1]
    expression = f"lastName='{last_name}' and firstName='{first_name}'"
    encoded_exp = urllib.parse.quote(expression)
    url = f"https://api.nhle.com/stats/rest/en/players?cayenneExp={encoded_exp}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data and 'data' in data:
            # Identity Check: Get the goalie ID[cite: 4]
            players = data['data']
            valid_goalies = [p for p in players if p.get('positionCode') == 'G']
            
            if not valid_goalies:
                print(f"!! No goalie found for {goalie_name}")
                return None, []

            g_id = valid_goalies[0]['id']
            
            # TRADE LOGIC: Use the game-log endpoint to find all team abbreviations[cite: 5]
            # Format: /v1/player/{player_id}/game-log/{season}/{game-type}
            log_url = f"https://api-web.nhle.com/v1/player/{g_id}/game-log/20252026/2"
            log_data = requests.get(log_url).json()
            
            # Extract unique team abbreviations from the game log
            teams = set()
            for game in log_data.get('gameLog', []):
                team_tri = game.get('teamAbbrev')
                if team_tri:
                    teams.add(team_tri)
            
            # Fallback to current team if no games played yet[cite: 4]
            if not teams:
                current_team = TEAM_MAP.get(valid_goalies[0].get('currentTeamId'))
                if current_team: teams.add(current_team)

            print(f"-> Resolved: {goalie_name} (ID: {g_id}) | History: {list(teams)}")
            return g_id, list(teams)
            
    except Exception as e:
        print(f"Resolution Error: {e}")
    return None, []

def resolve_game_id(team_tri, date_str):
    """
    Translates Team (e.g., 'BOS') and Date ('YYYY-MM-DD') into Game ID
    using the highly targeted Weekly Schedule endpoint.
    """
    url = f"https://api-web.nhle.com/v1/club-schedule/{team_tri}/week/{date_str}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # This endpoint returns a 'games' list for that specific week
        for game in data.get('games', []):
            if game.get('gameDate') == date_str:
                return game.get('id')
                
        print(f"!! No game found for {team_tri} on {date_str} in that week's schedule.")
    except Exception as e:
        print(f"Error resolving Game via Weekly Schedule: {e}")
    return None

def get_baseline_for_date(game_date, season_id="20252026"):
    """
    Fetches league-wide GOALIE stats. 
    Stops the entire system if live data cannot be resolved.
    """
    clean_date = game_date.split(" ")[0]
    
    # 1. GAIN: Check the vault first
    cached_val = gain_local_baseline(clean_date)
    if cached_val:
        return cached_val

    # 2. DUMP: Only fetch if missing
    url = f"https://api.nhle.com/stats/rest/en/goalie/summary?limit=-1&cayenneExp=seasonId={season_id} and gameDate<='{clean_date}'"
    response = requests.get(url)
    response.raise_for_status() 
    data = response.json()
    
    # 2. Check for Data Content
    if not data or 'data' not in data or len(data['data']) == 0:
        # CRASH EXPLICITLY: Do not allow the script to continue with a guess
        raise ValueError(f"CRITICAL DATA GAP: No goalie stats found for {clean_date} in season {season_id}.")

    total_saves = 0
    total_shots = 0
    
    for goalie in data['data']:
        saves = goalie.get('saves', 0)
        shots = goalie.get('shotsAgainst', 0)
        total_saves += saves
        total_shots += shots
    
    if total_shots == 0:
        raise ValueError(f"CRITICAL MATH ERROR: API returned data for {clean_date}, but total shots are 0.")

    rolling_avg = round(total_saves / total_shots, 3)
    
    # Save for future use
    dump_local_baseline(clean_date, rolling_avg)
    return rolling_avg

def sync_season_schedule():
    # Fetching the entire season range to avoid "Unknown" games
    url = "https://statsapi.web.nhl.com/api/v1/schedule?season=20252026"
    master_map = {}
    
    try:
        response = requests.get(url)
        data = response.json()
        # The NHL API organizes by day/week
        for week in data.get('gameWeek', []):
            for game in week.get('games', []):
                g_id = str(game.get('id'))
                away = game.get('awayTeam', {}).get('abbrev')
                home = game.get('homeTeam', {}).get('abbrev')
                master_map[g_id] = f"{away} @ {home}"
        
        with open("./data/schedule/master_schedule.json", "w") as f:
            json.dump(master_map, f, indent=4)
        print("Success: master_schedule.json created.")
    except Exception as e:
        print(f"Sync failed: {e}")

sync_season_schedule()
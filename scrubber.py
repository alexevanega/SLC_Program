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
    first_name = goalie_name.split(" ")[0].title()
    last_name = goalie_name.split(" ")[1].title()
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
    first_name, last_name = parts[0].title(), parts[1].title()
    expression = f"lastName='{last_name}' and firstName='{first_name}'"
    encoded_exp = urllib.parse.quote(expression)
    print(encoded_exp)
    print(expression)
    print (first_name, last_name)
    url = f"https://api.nhle.com/stats/rest/en/players?cayenneExp={encoded_exp}"
    
    try:
        response = requests.get(url)
        data = response.json()
        print(str(data))
        
        if data and 'data' in data:
            # Identity Check: Get the goalie ID
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
    Translates Team(s) (e.g., 'BOS' or ['BOS', 'CBJ']) and Date ('YYYY-MM-DD') into Game ID.
    If the goalie has multiple team abbreviations, this will try each candidate until it
    finds a matching game for the requested date.
    """
    if not team_tri:
        return None

    candidate_teams = team_tri if isinstance(team_tri, (list, tuple)) else [team_tri]
    for team in candidate_teams:
        url = f"https://api-web.nhle.com/v1/club-schedule/{team}/week/{date_str}"
        print(f"Checking schedule for {team}: {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            for game in data.get('games', []):
                if game.get('gameDate') == date_str:
                    return game.get('id')

            print(f"!! No game found for {team} on {date_str} in that week's schedule.")
        except Exception as e:
            print(f"Warning: schedule lookup failed for {team} on {date_str}: {e}")
            continue

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
    """
    Fetches full season schedule using nhlpy client.
    Uses team_season_schedule for each team, deduplicates games by ID.
    Much more reliable than the old statsapi endpoint.
    """
    from nhlpy import NHLClient
    
    # All NHL teams (2025-2026 season)
    TEAMS = [
        "NJD", "NYI", "NYR", "PHI", "PIT", "BOS", "BUF", "MTL", "OTT", "TOR", 
        "CAR", "FLA", "TBL", "WSH", "CHI", "DET", "NSH", "STL", "CGY", "COL", 
        "EDM", "VAN", "ANA", "DAL", "LAK", "SJS", "CBJ", "MIN", "WPG", "ARI", 
        "VGK", "SEA", "UTA"
    ]
    
    master_map = {}
    client = NHLClient()
    season = "20252026"
    
    try:
        for team in TEAMS:
            try:
                result = client.schedule.team_season_schedule(team, season)
                if 'games' in result:
                    for game in result['games']:
                        g_id = str(game.get('id'))
                        away = game.get('awayTeam', {}).get('abbrev')
                        home = game.get('homeTeam', {}).get('abbrev')
                        game_date = game.get('gameDate')
                        
                        # Deduplicate: each game appears twice (once per team), keep first occurrence
                        if g_id not in master_map:
                            master_map[g_id] = {
                                "matchup": f"{away} @ {home}",
                                "date": game_date
                            }
                
                print(f"  ✓ {team}")
            except Exception as e:
                print(f"  ✗ {team}: {e}")
                continue
        
        with open("./data/schedule/master_schedule.json", "w") as f:
            json.dump(master_map, f, indent=4)
        print(f"\nSuccess: master_schedule.json created with {len(master_map)} unique games.")
    except Exception as e:
        print(f"Sync failed: {e}")
        raise


# sync_season_schedule()
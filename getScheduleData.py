from nhlpy import NHLClient
from pathlib import Path
import json
import os

TEAM_MAP = [
    "NJD", "NYI", "NYR", "PHI", "PIT", "BOS", 
    "BUF", "MTL", "OTT", "TOR", "CAR", "FLA", 
    "TBL", "WSH", "CHI", "DET", "NSH", "STL", 
    "CGY", "COL", "EDM", "VAN", "ANA", "DAL", 
    "LAK", "SJS", "CBJ", "MIN", "WPG", "ARI", "VGK", 
    "SEA", "UTA"
]

def fill_schedule_data():
    client = NHLClient()
    base = Path("./data/schedule")

    for team in TEAM_MAP:
        # This returns a dictionary containing 'games', 'clubTimezone', etc.
        raw_data = client.schedule.team_season_schedule(team, 20252026)
        
        # Extract the actual list of games
        schedule_games = raw_data.get('games', [])
        
        out = base / team / "team_schedule.json"
        out.parent.mkdir(parents=True, exist_ok=True)
    
        # Convert the Game objects to dictionaries so the keys are REACHABLE
        serializable_data = [game.__dict__ if hasattr(game, '__dict__') else game for game in schedule_games]

        with out.open("w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4)

# Relative Paths# Force these to Path objects from the start to stop the backslash injection
SCHEDULE_DIR = Path("./data/schedule")
MASTER_PATH = SCHEDULE_DIR / "master_schedule.json"

def aggregate_schedules():
    master_map = {}
    if not SCHEDULE_DIR.exists(): return

    for file_path in SCHEDULE_DIR.rglob("team_schedule.json"):
        # as_posix() forces forward slashes to avoid backslash errors
        print(f"Processing: {file_path.as_posix()}") 
        
        try:
            # Inside the try block of aggregate_schedules:
            with file_path.open('r', encoding='utf-8') as f:
                content = json.load(f)
    
            # If the file is the full dictionary, get the games list. 
            # If it's already just the list (from the fix above), use it directly.
            games_list = content.get('games', []) if isinstance(content, dict) else content
    
            for game in games_list:
                if isinstance(game, dict):
                    g_id = str(game.get('id'))
                    away = game.get('awayTeam', {}).get('abbrev')
                    home = game.get('homeTeam', {}).get('abbrev')
            
                if g_id and away and home:
                    master_map[g_id] = f"{away} @ {home}"
        except Exception as e:
            print(f"Skipping {file_path.as_posix()}: {e}")

    with MASTER_PATH.open('w', encoding='utf-8') as f:
        json.dump(master_map, f, indent=4)

def verify_master_integrity():
    if not MASTER_PATH.exists():
        print("Error: Master schedule not found.")
        return

    with MASTER_PATH.open('r', encoding='utf-8') as f:
        master_data = json.load(f)

    game_ids = list(master_data.keys())
    total_entries = len(game_ids)
    unique_entries = len(set(game_ids))

    print(f"--- Master Schedule Audit ---")
    print(f"Total Entries:  {total_entries}")
    print(f"Unique IDs:     {unique_entries}")
    
    if total_entries != unique_entries:
        print(f"!! CRITICAL: Found {total_entries - unique_entries} duplicate keys.")
    else:
        print("Success: No duplicate Game IDs found in the master file.")

if __name__ == "__main__":
    verify_master_integrity()
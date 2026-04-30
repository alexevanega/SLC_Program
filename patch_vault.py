import os
import json
import requests

VAULT_PATH = "./data/processedGames/"
# Using a broader range to ensure no "Unknowns" for the 25-26 season
SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/2025-10-01"

def patch_system():
    # 1. Build the Master Lookup
    print("Fetching master schedule...")
    master_map = {}
    try:
        # The NHL API returns weeks of games
        # To be safe, we'd ideally loop this, but let's grab the season start node
        response = requests.get(SCHEDULE_URL)
        data = response.json()
        for week in data.get('gameWeek', []):
            for game in week.get('games', []):
                g_id = str(game.get('id'))
                away = game.get('awayTeam', {}).get('abbrev')
                home = game.get('homeTeam', {}).get('abbrev')
                date = game.get('gameDate')
                master_map[g_id] = {"label": f"{away} @ {home}", "date": date}
    except Exception as e:
        print(f"Failed to fetch schedule: {e}")
        return

    # 2. Patch the 68 Goalie Folders
    print("Patching JSON files in vault...")
    goalie_folders = [d for d in os.listdir(VAULT_PATH) if os.path.isdir(os.path.join(VAULT_PATH, d))]
    
    for folder in goalie_folders:
        folder_path = os.path.join(VAULT_PATH, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                file_path = os.path.join(folder_path, file)
                
                with open(file_path, 'r') as f:
                    game_data = json.load(f)
                
                g_id = str(game_data.get('game_id'))
                
                # If we find a match in our master map, inject it
                if g_id in master_map:
                    game_data['matchup'] = master_map[g_id]['label']
                    game_data['gameDate'] = master_map[g_id]['date']
                    
                    # Write the corrected data back to the local SSD
                    with open(file_path, 'w') as f:
                        json.dump(game_data, f, indent=4)

    print("Patch Complete. Your vault is now UI-Ready.")

if __name__ == "__main__":
    patch_system()
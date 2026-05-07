import os
import json
import requests

VAULT_PATH = "./data/processedGames/"
MASTER_REPORT_FILE = os.path.join(VAULT_PATH, "master_report.json")
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

    # 2. Patch the master report only.
    print("Patching master report...")
    if not os.path.exists(MASTER_REPORT_FILE):
        print("No master_report.json found.")
        return

    with open(MASTER_REPORT_FILE, 'r', encoding='utf-8') as f:
        master_report = json.load(f)

    patched = 0
    for goalie_games in master_report.values():
        for game_data in goalie_games.values():
            g_id = str(game_data.get('game_id'))
            if g_id not in master_map:
                continue

            game_data['matchup'] = master_map[g_id]['label']
            game_data['gameDate'] = master_map[g_id]['date']
            patched += 1

    with open(MASTER_REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(master_report, f, indent=4)

    print(f"Patch Complete. Updated {patched} master report entries.")

if __name__ == "__main__":
    patch_system()

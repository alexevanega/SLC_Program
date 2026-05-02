from nhlpy import NHLClient
import json

client = NHLClient()

# List of all NHL teams
TEAMS = [
    "NJD", "NYI", "NYR", "PHI", "PIT", "BOS", "BUF", "MTL", "OTT", "TOR", 
    "CAR", "FLA", "TBL", "WSH", "CHI", "DET", "NSH", "STL", "CGY", "COL", 
    "EDM", "VAN", "ANA", "DAL", "LAK", "SJS", "CBJ", "MIN", "WPG", "ARI", 
    "VGK", "SEA", "UTA"
]

master_map = {}
duplicate_count = 0

print(f"Fetching season schedule from {len(TEAMS)} teams...")

for team in TEAMS[:5]:  # Test with first 5 teams
    try:
        result = client.schedule.team_season_schedule(team, "20252026")
        if 'games' in result:
            for game in result['games']:
                g_id = str(game.get('id'))
                away = game.get('awayTeam', {}).get('abbrev')
                home = game.get('homeTeam', {}).get('abbrev')
                game_date = game.get('gameDate')
                
                if g_id not in master_map:
                    master_map[g_id] = {
                        "matchup": f"{away} @ {home}",
                        "date": game_date
                    }
                else:
                    duplicate_count += 1
        
        print(f"  {team}: {len(result.get('games', []))} games")
    except Exception as e:
        print(f"  {team}: FAILED - {e}")

print(f"\nTotal unique games: {len(master_map)}")
print(f"Duplicate entries skipped: {duplicate_count}")
print(f"Sample games:")
for i, (g_id, info) in enumerate(list(master_map.items())[:3]):
    print(f"  {g_id}: {info['matchup']} on {info['date']}")



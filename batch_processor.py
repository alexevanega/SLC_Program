import requests # type: ignore 
from engine import run_slc_workflow
from datetime import datetime, timedelta
from utility import current_report_score

def harvest_full_season(team_list, goalieNameAndId):
    # Ensure team_list is handled as a list
    if isinstance(team_list, str):
        team_list = [team_list]
        
    all_season_reports = []
    
    for team_tri in team_list:
        print(f"Scanning schedule for {team_tri}...")
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_tri}/now"
        try:
            response = requests.get(url)
            data = response.json()
            # Regular season games only[cite: 1]
            games = [g for g in data.get('games', []) if g.get('gameType') == 2]
            
            for game in games:
                game_date = game.get('gameDate')
                # Run existing engine workflow[cite: 2]
                report = run_slc_workflow(goalieNameAndId, team_tri, game_date)
                
                # Filter for games where goalie actually had S_saves or S_goals[cite: 1, 2]
                if report and (report['total']['stats']['S_saves'] + report['total']['stats']['S_goals']) > 0:
                    all_season_reports.append(report)
        except Exception as e:
            print(f"System Error on {team_tri}: {e}")
            
    return all_season_reports

def harvest_date_range(team_tri, goalieNameAndId, start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    reports = []
    
    print(f"Starting Harvest for {goalieNameAndId['name']} from {start_date} to {end_date}...")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        report = run_slc_workflow(goalieNameAndId, team_tri, date_str)
        
        # --- THE FILTER ---
        # Only add the report if Goalie actually played
        if report and (report['total']['stats']['S_saves'] + report['total']['stats']['S_goals']) > 0:
            reports.append(report)
            
        current += timedelta(days=1)
    return reports

def generate_goalie_profile(reports):
    if not reports:
        print("\n" + "="*40)
        print("SEASONAL PROFILE: 0 Games Found")
        print("="*40)
        return

    total_slc = 0
    total_saves = 0
    total_goals = 0
    game_count = len(reports)
    
    for r in reports:
        total_slc += current_report_score(r)
        total_saves += r['total']['stats']['S_saves']
        total_goals += r['total']['stats']['S_goals']
    
    avg_slc = total_slc / game_count
    
    print("\n" + "="*40)
    print(f"SEASONAL PROFILE: {game_count} Active Games") # Explicitly tracking active starts
    print(f"Cumulative SLC: {total_slc:.3f}")
    print(f"Average SLC:    {avg_slc:.3f}")
    
    # Preventing division by zero for the SV% calculation
    total_shots = total_saves + total_goals
    season_sv = total_saves / total_shots if total_shots > 0 else 0
    print(f"Season SV%:     {season_sv:.3f}")
    print("="*40)

from batch_processor import harvest_full_season, harvest_date_range, generate_goalie_profile
from datetime import datetime, timedelta
from engine import run_slc_workflow
from scrubber import resolve_goalie_and_team

# --- CONFIGURATION ---
GOALIE = "Frederik Andersen"
g_id, team_tri = resolve_goalie_and_team(GOALIE)
goalieNameAndId = {'name': GOALIE, 'id': g_id}
# 1. THE SINGLE GAME RUNNER
def run_single_game_report(game_date):
    """Analyzes one specific game on a specific date."""
    print(f"--- STARTING SINGLE GAME RUN: {game_date} ---")
    # This calls the engine directly for a one-off analysis
    report = run_slc_workflow(goalieNameAndId, team_tri, game_date)
    
    if report:
        # We wrap it in a list so generate_goalie_profile can still process it
        generate_goalie_profile([report])
    return report

# 2. THE MONTHLY RUNNER
def run_monthly_report(month_str): # e.g., "2026-03"
    start = f"{month_str}-01"
    # Logic to handle month end could be added, or just use a 31-day window
    end = f"{month_str}-31" 
    results = harvest_date_range(team_tri, goalieNameAndId, start, end)
    generate_goalie_profile(results)

# 3.  THE WEEKLY RUNNER
def run_weekly_report(start_date): # e.g., "2026-01-19"
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=6)
    results = harvest_date_range(team_tri, goalieNameAndId, start_date, end_dt.strftime("%Y-%m-%d"))
    generate_goalie_profile(results)

# 4. THE SEASONAL RUNNER
def run_seasonal_report():
    """Analyzes every game in the team_tri's full season schedule"""
    print(f"--- STARTING FULL SEASON RUN FOR {goalieNameAndId['name']} ---")
    reports = harvest_full_season(team_tri, goalieNameAndId)
    generate_goalie_profile(reports)

# 5. THE CUSTOM RANGE RUNNER (The "Formula for each need")
def run_custom_range_report(start_date, end_date):
    """Analyzes a specific custom window of time."""
    print(f"--- STARTING CUSTOM RANGE: {start_date} to {end_date} ---")
    reports = harvest_date_range(team_tri, goalieNameAndId, start_date, end_date)
    generate_goalie_profile(reports)

if __name__ == "__main__":
    
    run_custom_range_report("2026-04-26", "2026-05-05")
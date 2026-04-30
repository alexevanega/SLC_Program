# Import specialized arms
from scrubber import resolve_game_id, get_baseline_for_date
from slc_by_period_progression import analyze_progression_from_raw
from utility import fetch_and_vault_raw_data, load_local_report,save_report_locally

def run_slc_workflow(goalieNameAndId, team_tri, game_date):
    """
    The Logical Workflow with Persistence:
    1. Resolve Strings to IDs
    2. Check Local Vault for existing data
    3. Run Period Analysis
    4. Save and Output Results
    """
    print(f"\nInitializing SLC Analysis for {goalieNameAndId['name']}...")
    
    # Arm 1: The Scrubber
    m_id = resolve_game_id(team_tri, game_date)
    
    if not m_id:
        print("Workflow Aborted: Could not resolve Game ID.")
        return
    
    # --- THE GATEKEEPER ---
    # Check if we have already crunched the numbers for this Game ID
    existing_report = load_local_report(m_id, goalieNameAndId['name'])
    if existing_report:
        # Return immediately if found - total bypass of API and Pandas
        return existing_report
    
    # DUMP AND GAIN: Get the raw mass of data for the game
    raw_game_data = fetch_and_vault_raw_data(m_id)
    
    # ANALYZE: Pass the raw data and goalie ID to the progression logic
    # This logic now extracts ONLY what it needs for this specific goalie
    xS_baseline = get_baseline_for_date(game_date)
    report_data = analyze_progression_from_raw(raw_game_data, goalieNameAndId['id'], xS_baseline)

    # Only print the report if there is active data to show
    total_activity = report_data['total']['stats']['S_saves'] + report_data['total']['stats']['S_goals']
    
    # 3. THE PRINT LAYER: Identical output regardless of data source
    if total_activity == 0:
        pass
    else:
        print(f"\nSLC PROGRESSION REPORT | Game ID: {report_data['game_id']} | Goalie: {goalieNameAndId['name']}")
        print("-" * 80)
        print(f"{'Period':<8} | {'SLC Score':<10} | {'Saves/Goals':<12} | {'Service':<8} | {'Phys. Tax'}")
        print("-" * 80)

        # Note: Using .items() because the data structure is a dictionary
        for p_num, p_data in report_data['periods'].items():
            s = p_data['stats']
            print(f"P{p_num:<7} | {p_data['score']:<10.3f} | {s['S_saves']}/{s['S_goals']:<9} | {p_data['service']:<8} | {p_data['tax']:<10}")

        print("-" * 80)
        print(f"{'TOTAL':<8} | {report_data['total']['score']:<10.3f} | {report_data['total']['stats']['S_saves']}/{report_data['total']['stats']['S_goals']:<9}")
        print(f"{'Breakdown':<8} | {report_data['total']['stats']}")
        print("-" * 80)

    # STEP 4: DUMP - Save the new result in the goalie-specific folder
    if report_data:
        save_report_locally(report_data, goalieNameAndId['name'])
    
    return report_data
import pandas as pd
import math
from utility import get_total_seconds, calculate_slc_score

def analyze_progression_from_raw(raw_game_data, goalie_id, xS_baseline):
    """
    Optimized with Pandas to eliminate loop-based time complexity.
    """
    if not raw_game_data or 'plays' not in raw_game_data:
        return None
    
    game_id = raw_game_data.get('id')
    # Load all plays into a DataFrame
    df = pd.DataFrame(raw_game_data['plays'])

    # --- VECTORIZED PRE-PROCESSING ---
    # Extract period and period type
    df['period'] = df['periodDescriptor'].apply(lambda x: x.get('number'))
    df['periodType'] = df['periodDescriptor'].apply(lambda x: x.get('periodType'))
    
    # Filter out Shootouts and irrelevant periods
    df = df[(df['periodType'] != 'SO') & (df['period'] <= 4)].copy()
    
    # Extract total seconds for time-based logic[cite: 11, 12]
    df['time_sec'] = df.apply(lambda row: get_total_seconds(row['timeInPeriod'], row['period']), axis=1)

    # --- GOALIE EVENT FILTERING ---
    # We identify plays where the goalie is the primary actor (giveaway/takeaway) 
    # or the one in net (shots/goals)
    def is_goalie_involved(row):
        details = row.get('details')
        # This check prevents the 'float' object error by ensuring details is a dictionary
        if not isinstance(details, dict): 
            return False
        return details.get('playerId') == goalie_id or details.get('goalieInNetId') == goalie_id

    df['is_goalie_action'] = df.apply(is_goalie_involved, axis=1)
    
    # Initialize the stats buckets for P1-P4
    progression = {p: {'S_saves': 0, 'S_goals': 0, 'NPW': 0, 'iTkA': 0, 'iGvA': 0, 'UA': 0, 'RP': 0, 'iGvA_SH': 0} 
                   for p in range(1, 5)}

    # --- TIME-SENSITIVE REBOUND LOGIC ---
    # Logic: Pandas finds shots-on-goal for our goalie and checks the 'next' relevant event
    # We still iterate through a significantly smaller 'filtered' set for rebound windows
    goalie_df = df[(df['is_goalie_action']) | (df['typeDescKey'] == 'stoppage')].copy()
    
    last_save_time = -999
    active_window = False

    for _, row in goalie_df.iterrows():
        p_num = row['period']
        stats = progression[p_num]
        event = row['typeDescKey']
        details = row.get('details', {})
        sit_code = str(row.get('situationCode', '1551'))
        is_sh = sit_code[1] != '5' or sit_code[2] != '5'

        # Service & Tax logic
        if event == 'giveaway' and details.get('playerId') == goalie_id:
            if is_sh: stats['iGvA_SH'] += 1
            else: stats['iGvA'] += 1
        
        elif event == 'takeaway' and details.get('playerId') == goalie_id:
            stats['iTkA'] += 1

        elif event == 'stoppage' and active_window:
            if (row['time_sec'] - last_save_time) <= 3:
                stats['NPW'] += 1
            active_window = False

        elif event == 'goal' and details.get('goalieInNetId') == goalie_id:
            stats['S_goals'] += 1
            active_window = False

        elif event == 'shot-on-goal' and details.get('goalieInNetId') == goalie_id:
            # Rebound Detection
            if active_window and (row['time_sec'] - last_save_time) <= 3:
                x, y = details.get('xCoord', 0), details.get('yCoord', 0)
                dist = math.sqrt((x - (89 if x > 0 else -89))**2 + (y - 0)**2)
                if dist < 25: stats['RP'] += 1
                else: stats['UA'] += 1
            
            stats['S_saves'] += 1
            last_save_time = row['time_sec']
            active_window = True

        # UA Timeout
        if active_window and (row['time_sec'] - last_save_time) > 5:
            stats['UA'] += 1
            active_window = False

    return compile_final_report(game_id, goalie_id, progression, xS_baseline)

def compile_final_report(game_id, goalie_id, progression, xS_baseline):
    """Summarizes stats into SLC format."""
    total_stats = {'S_saves': 0, 'S_goals': 0, 'NPW': 0, 'iTkA': 0, 'iGvA': 0, 'UA': 0, 'RP': 0, 'iGvA_SH': 0}
    summaries = {}

    for p_num, p_stats in progression.items():
        if (p_stats['S_saves'] + p_stats['S_goals']) == 0: continue
        
        score, _ = calculate_slc_score(p_stats, xS_baseline)
        summaries[p_num] = {
            'score': score,
            'stats': p_stats,
            'tax': (p_stats['UA'] * 1.0) + (p_stats['RP'] * 2.0) + (p_stats['iGvA'] * 2.0) + (p_stats['iGvA_SH'] * 3.0),
            'service': p_stats['NPW'] + p_stats['iTkA']
        }
        for key in total_stats: total_stats[key] += p_stats[key]

    final_score, _ = calculate_slc_score(total_stats, xS_baseline)
    return {
        'game_id': game_id, 'goalie_id': goalie_id, 
        'periods': summaries, 'total': {'score': final_score, 'stats': total_stats}
    }
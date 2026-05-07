import json
import pandas as pd
from pathlib import Path
from utility import calculate_service_tax

MASTER_REPORT_PATH = Path("./data/processedGames/master_report.json")


def categorize_goalie(avg_slc, sovereignty, slc_baseline=0, sovereignty_baseline=0):
    slc_above_baseline = avg_slc >= slc_baseline
    sovereignty_above_baseline = sovereignty >= sovereignty_baseline

    if slc_above_baseline and sovereignty_above_baseline:
        return "Stabilizer"
    if slc_above_baseline and not sovereignty_above_baseline:
        return "Systems Man"
    if not slc_above_baseline and sovereignty_above_baseline:
        return "Functional Drain"
    return "Systemic Drain"


def get_league_leaderboard(min_gp=1):
    league_data = []
    
    if not MASTER_REPORT_PATH.exists():
        return pd.DataFrame()

    with MASTER_REPORT_PATH.open('r', encoding='utf-8') as f:
        master_report = json.load(f)

    for goalie_key, games in master_report.items():
        if not games:
            continue

        first_game = next(iter(games.values()))
        goalie_name = goalie_key.replace("_", " ").title()
        cumulative_stats = {
            "Goalie": goalie_name, "ID": first_game.get('goalie_id'), "GP": 0,
            "Total_SLC": 0.0, "Avg_SLC": 0.0,
            "S_saves": 0, "S_goals": 0, "NPW": 0, "RP": 0, "UA": 0,
            "NPW_SH": 0, "RP_SH": 0, "UA_SH": 0,
            "iTkA": 0, "iGvA": 0, "iGvA_SH": 0, "Tax": 0.0,
            "Service": 0.0, "xG": 0.0, "xS": 0.0, "Sequences": 0
        }
        
        for data in games.values():
            total = data.get('total', {})
            stats = total.get('stats', {})
            
            cumulative_stats["GP"] += 1
            cumulative_stats["Total_SLC"] += total.get('score', 0)
            cumulative_stats["S_saves"] += stats.get('S_saves', 0)
            cumulative_stats["S_goals"] += stats.get('S_goals', 0)
            cumulative_stats["NPW"] += stats.get('NPW', 0)
            cumulative_stats["NPW_SH"] += stats.get('NPW_SH', 0)
            cumulative_stats["RP"] += stats.get('RP', 0)
            cumulative_stats["RP_SH"] += stats.get('RP_SH', 0)
            cumulative_stats["UA"] += stats.get('UA', 0)
            cumulative_stats["UA_SH"] += stats.get('UA_SH', 0)
            cumulative_stats["iTkA"] += stats.get('iTkA', 0)
            cumulative_stats["iGvA"] += stats.get('iGvA', 0)
            cumulative_stats["iGvA_SH"] += stats.get('iGvA_SH', 0)
            cumulative_stats["xG"] += stats.get('xG', 0)
            cumulative_stats["xS"] += stats.get('xS', 0)
            cumulative_stats["Sequences"] += stats.get('Sequences', 0)
        
        shots = cumulative_stats["S_saves"] + cumulative_stats["S_goals"]
        actual_sv = cumulative_stats["S_saves"] / shots if shots > 0 else 0
        cumulative_stats["SV%"] = round(actual_sv, 3) if shots > 0 else 0
        expected_saves = cumulative_stats["xS"] if cumulative_stats["xS"] else shots - cumulative_stats["xG"]
        expected_sv = expected_saves / shots if shots > 0 else 0
        cumulative_stats["Sovereignty"] = round((actual_sv - expected_sv) * 100, 3) if shots > 0 else 0
        cumulative_stats["GSAx"] = round(cumulative_stats["xG"] - cumulative_stats["S_goals"], 3)
        cumulative_stats["Avg_SLC"] = round(cumulative_stats["Total_SLC"] / cumulative_stats["GP"], 3) if cumulative_stats["GP"] > 0 else 0
        cumulative_stats["Service"], cumulative_stats["Tax"] = calculate_service_tax(cumulative_stats, include_baseline=True)
        cumulative_stats["Net_Load"] = round(cumulative_stats["Service"] - cumulative_stats["Tax"], 3)
        
        if cumulative_stats["Sequences"] > 0:
            cumulative_stats["Reset_Ratio"] = round(cumulative_stats["S_saves"] / cumulative_stats["Sequences"], 2)
        else:
            defensive_load = cumulative_stats["UA"] + cumulative_stats["RP"]
            cumulative_stats["Reset_Ratio"] = round(cumulative_stats["NPW"] / defensive_load, 2) if defensive_load > 0 else 0
        league_data.append(cumulative_stats)

    df = pd.DataFrame(league_data)
    if df.empty:
        return df

    df = df[df['GP'] >= min_gp].copy()
    if df.empty:
        return df

    slc_baseline = df["Avg_SLC"].mean()
    sovereignty_baseline = df["Sovereignty"].mean()
    df["Identity"] = df.apply(
        lambda row: categorize_goalie(
            row["Avg_SLC"],
            row["Sovereignty"],
            slc_baseline,
            sovereignty_baseline
        ),
        axis=1
    )
    df["Category"] = df["Identity"]
    df["Type"] = df["Identity"]
    df["SLC_Baseline"] = round(float(slc_baseline), 3)
    df["Sovereignty_Baseline"] = round(float(sovereignty_baseline), 3)

    return df.sort_values("Total_SLC", ascending=False)

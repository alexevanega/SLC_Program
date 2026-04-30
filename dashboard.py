import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px
from pathlib import Path
from scrubber import resolve_goalie_and_team

VAULT_PATH = Path("./data/processedGames/")
MASTER_PATH = Path("./data/schedule/master_schedule.json")

def load_goalie_data(active_goalie):
    # 1. Load the Master Schedule[cite: 4]
    master_map = {}
    if MASTER_PATH.exists():
        with MASTER_PATH.open('r', encoding='utf-8') as f:
            master_map = json.load(f)

    # 2. Hard-fix the goalie name for the resolution
    # This specifically fixes the IndexError you just hit
    api_name = active_goalie.replace("_", " ").title()
    _, goalie_teams = resolve_goalie_and_team(api_name)
    
    # 3. Use the sanitized folder name for the path
    folder_path = VAULT_PATH / active_goalie.lower()
    all_games = []
    
    if not folder_path.exists():
        return pd.DataFrame()

    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with (folder_path / file).open('r', encoding='utf-8') as f:
                data = json.load(f)
                
                # USE THE CORRECT KEYS FROM UTILITY_9.PY AND PATCH_VAULT.PY[cite: 5, 10]
                g_id = str(data.get('game_id'))
                # Your patch_vault.py specifically uses 'gameDate'
                date_val = data.get('gameDate', "0000-00-00")
                
                matchup_str = master_map.get(g_id, f"ID: {g_id}")
                clean_matchup = matchup_str

                # REFINED STRIPPING LOGIC[cite: 2, 7]
                if goalie_teams and matchup_str:
                    for t_abbrev in goalie_teams:
                        t_abbrev = t_abbrev.upper()
                        if t_abbrev in matchup_str:
                            # If goalie is VGK: "VGK @ WPG" -> "at WPG" | "WPG @ VGK" -> "vs WPG"
                            clean_matchup = matchup_str.replace(f"{t_abbrev} @", "at").replace(f"@ {t_abbrev}", "vs")
                            break

                all_games.append({
                    "Date": date_val,
                    "Matchup": clean_matchup,
                    "SLC": float(data.get('total', {}).get('score', 0))
                })
                
    return pd.DataFrame(all_games).sort_values("Date")

st.title("Systemic Goalie Audit")
active_goalie = None

if os.path.exists(VAULT_PATH):
    goalies = sorted([d for d in os.listdir(VAULT_PATH) if os.path.isdir(os.path.join(VAULT_PATH, d))])
    active_goalie = st.sidebar.selectbox("Select Goalie", goalies)

if active_goalie:
    df = load_goalie_data(active_goalie)
    if not df.empty:
        st.metric(f"Season Average: {active_goalie}", f"{df['SLC'].mean():.3f}")
        fig = px.line(df, x="Matchup", y="SLC", markers=True, hover_data=["Date"])
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
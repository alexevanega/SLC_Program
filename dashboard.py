import streamlit as st
import contextlib
import io
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
from scrubber import get_player_metadata
from leaderboard_engine import categorize_goalie
from report_exporter import (
    comparison_summary,
    dataframe_csv,
    goalie_audit_summary,
    rankings_summary,
    team_redline_summary,
)
from update_local_data import update_local_data
from utility import (
    build_game_slc_xga_data,
    build_high_interest_loan_data,
    build_slc_hypothesis_data,
    calculate_service_tax,
    current_report_score,
    get_available_teams,
    get_team_games,
    get_team_sequence_data,
    get_team_sequence_histogram,
    load_master_reports,
    load_goalie_data
)

PLAYOFF_CUTOFF = pd.Timestamp("2026-04-16")
RAW_GAMES_PATH = Path("./data/raw_games")
MASTER_REPORT_PATH = Path("./data/processedGames/master_report.json")

# --- 2. THE STANDARD GOALIE HEADER (GLOBAL COMPONENT) ---

def category_color(category):
    return {
        "Stabilizer": "green",
        "Systems Man": "blue",
        "Functional Drain": "orange",
        "Systemic Drain": "red"
    }.get(category, "gray")


def render_goalie_header(goalie_name, leaderboard_df, key_suffix=""):
    """Standardized Bio-Header for Single and Comparative Views."""
    goalie_rows = leaderboard_df[leaderboard_df["Goalie"] == goalie_name] if not leaderboard_df.empty else pd.DataFrame()
    if goalie_rows.empty:
        st.info(f"No {goalie_name} leaderboard profile is available in this game set.")
        return

    g_row = goalie_rows.iloc[0]
    meta = get_player_metadata(g_row["ID"])
    
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 3, 1, 1])
        with c1:
            if meta["image"]: st.image(meta["image"], width=100)
        with c2:
            st.markdown(f"### {goalie_name}")
            st.caption(f"{meta['team']} | {meta['height']} | {meta['weight']} | {meta['age']}")
            
            category = g_row.get("Category", g_row["Identity"])
            color = category_color(category)
            st.markdown(f":{color}[**{category}**] (Reset: {g_row['Reset_Ratio']})")
            st.caption("Category is based on full-season SLC and Sovereignty, including playoffs.")
        with c3:
            st.metric("Avg SLC", f"{g_row['Avg_SLC']:.2f}")
        with c4:
            st.metric("GP", int(g_row["GP"]))


def filter_hypothesis_window(df, start_date, end_date):
    if df.empty:
        return df

    filtered = df.copy()
    filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
    filtered = filtered.dropna(subset=["Date"])
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    mask = (filtered["Date"] >= start_ts) & (filtered["Date"] <= end_ts)
    return filtered.loc[mask]


def get_comparison_window(t_window, date_value):
    if t_window == "Date Range" and len(date_value) == 2:
        start_d, end_d = date_value
    elif t_window == "Single Game" and date_value:
        start_d = end_d = date_value
    else:
        start_d, end_d = datetime(2000, 1, 1), datetime.now()

    if pd.Timestamp(start_d) > pd.Timestamp(end_d):
        start_d, end_d = end_d, start_d
    return start_d, end_d


def _is_playoff_date(date_value):
    parsed_date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed_date):
        return False
    return parsed_date > PLAYOFF_CUTOFF


def _game_type_from_id(game_id):
    game_id = str(game_id or "")
    if len(game_id) >= 6 and game_id[4:6].isdigit():
        return int(game_id[4:6])
    return None


def _matches_game_phase(date_value, game_phase, game_id=None):
    parsed_date = pd.to_datetime(date_value, errors="coerce")
    game_type = _game_type_from_id(game_id)

    if game_phase == "Playoffs":
        if pd.notna(parsed_date):
            return parsed_date > PLAYOFF_CUTOFF
        return game_type == 3

    if game_type == 1:
        return False
    if pd.notna(parsed_date):
        return parsed_date <= PLAYOFF_CUTOFF and game_type != 3
    return game_type == 2


def filter_by_game_phase(df, game_phase, date_col="Date"):
    if df.empty or date_col not in df.columns:
        return df

    filtered = df.copy()
    game_ids = filtered["Game_ID"] if "Game_ID" in filtered.columns else pd.Series([None] * len(filtered), index=filtered.index)
    mask = [
        _matches_game_phase(date_value, game_phase, game_id)
        for date_value, game_id in zip(filtered[date_col], game_ids)
    ]
    return filtered.loc[mask]


def filter_team_games_by_phase(games, game_phase):
    return [
        game for game in games
        if _matches_game_phase(game.get("Date"), game_phase, game.get("Game_ID"))
    ]


def get_master_report_version():
    return MASTER_REPORT_PATH.stat().st_mtime_ns if MASTER_REPORT_PATH.exists() else 0


@st.cache_data
def _load_raw_game_date(game_id):
    raw_path = RAW_GAMES_PATH / f"{game_id}.json"
    if not raw_path.exists():
        return None

    with raw_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("gameDate")


@st.cache_data(show_spinner=False)
def _load_raw_game_for_dashboard(game_id):
    raw_path = RAW_GAMES_PATH / f"{game_id}.json"
    if not raw_path.exists():
        return {}

    with raw_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_report_date(game_id, report):
    return report.get("gameDate") or _load_raw_game_date(game_id)


def _goalie_team_abbrev_from_raw(raw_game, goalie_id):
    goalie_team_id = None
    for player in raw_game.get("rosterSpots", []):
        if player.get("playerId") == goalie_id:
            goalie_team_id = player.get("teamId")
            break

    if not goalie_team_id:
        return None

    for team_key in ["homeTeam", "awayTeam"]:
        team = raw_game.get(team_key, {})
        if team.get("id") == goalie_team_id:
            return team.get("abbrev")
    return None


def _empty_leaderboard_stats(goalie_name, goalie_id):
    return {
        "Goalie": goalie_name, "ID": goalie_id, "GP": 0,
        "Total_SLC": 0.0, "Avg_SLC": 0.0,
        "S_saves": 0, "S_goals": 0, "NPW": 0, "RP": 0, "UA": 0,
        "NPW_SH": 0, "RP_SH": 0, "UA_SH": 0,
        "iTkA": 0, "iGvA": 0, "iGvA_SH": 0, "Tax": 0.0,
        "Service": 0.0, "xG": 0.0, "xS": 0.0, "Sequences": 0
    }


def _add_report_to_leaderboard_stats(cumulative_stats, data):
    total = data.get("total", {})
    stats = total.get("stats", {})

    cumulative_stats["GP"] += 1
    cumulative_stats["Total_SLC"] += current_report_score(data)
    cumulative_stats["S_saves"] += stats.get("S_saves", 0)
    cumulative_stats["S_goals"] += stats.get("S_goals", 0)
    cumulative_stats["NPW"] += stats.get("NPW", 0)
    cumulative_stats["NPW_SH"] += stats.get("NPW_SH", 0)
    cumulative_stats["RP"] += stats.get("RP", 0)
    cumulative_stats["RP_SH"] += stats.get("RP_SH", 0)
    cumulative_stats["UA"] += stats.get("UA", 0)
    cumulative_stats["UA_SH"] += stats.get("UA_SH", 0)
    cumulative_stats["iTkA"] += stats.get("iTkA", 0)
    cumulative_stats["iGvA"] += stats.get("iGvA", 0)
    cumulative_stats["iGvA_SH"] += stats.get("iGvA_SH", 0)
    cumulative_stats["xG"] += stats.get("xG", 0)
    cumulative_stats["xS"] += stats.get("xS", 0)
    cumulative_stats["Sequences"] += stats.get("Sequences", 0)


def _finalize_leaderboard_stats(cumulative_stats, category_override=None, slc_baseline=0, sovereignty_baseline=0):
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

    category = category_override or categorize_goalie(
        cumulative_stats["Avg_SLC"],
        cumulative_stats["Sovereignty"],
        slc_baseline,
        sovereignty_baseline
    )
    cumulative_stats["Identity"] = category
    cumulative_stats["Category"] = category
    cumulative_stats["Type"] = category
    cumulative_stats["SLC_Baseline"] = round(float(slc_baseline), 3)
    cumulative_stats["Sovereignty_Baseline"] = round(float(sovereignty_baseline), 3)
    return cumulative_stats


def build_full_season_category_map(master_report, baseline_min_gp=20):
    season_rows = []
    for goalie_key, games in master_report.items():
        if not games:
            continue

        first_game = next(iter(games.values()))
        goalie_name = goalie_key.replace("_", " ").title()
        season_stats = _empty_leaderboard_stats(goalie_name, first_game.get("goalie_id"))
        for data in games.values():
            _add_report_to_leaderboard_stats(season_stats, data)
        if season_stats["GP"] > 0:
            season_stats["Goalie_Key"] = goalie_key
            season_rows.append(_finalize_leaderboard_stats(season_stats))

    season_df = pd.DataFrame(season_rows)
    if season_df.empty:
        return {}

    baseline_df = season_df[season_df["GP"] >= baseline_min_gp]
    if baseline_df.empty:
        baseline_df = season_df

    slc_baseline = baseline_df["Avg_SLC"].mean()
    sovereignty_baseline = baseline_df["Sovereignty"].mean()
    categories = {}
    for _, row in season_df.iterrows():
        categories[row["Goalie_Key"]] = categorize_goalie(
            row["Avg_SLC"],
            row["Sovereignty"],
            slc_baseline,
            sovereignty_baseline
        )
    return categories


def build_phase_leaderboard(game_phase, min_gp=1):
    league_data = []
    master_report = load_master_reports()
    full_season_categories = build_full_season_category_map(master_report)

    for goalie_key, games in master_report.items():
        if not games:
            continue

        first_game = next(iter(games.values()))
        goalie_name = goalie_key.replace("_", " ").title()
        cumulative_stats = _empty_leaderboard_stats(goalie_name, first_game.get("goalie_id"))

        for game_id, data in games.items():
            if not _matches_game_phase(_get_report_date(game_id, data), game_phase, game_id):
                continue

            _add_report_to_leaderboard_stats(cumulative_stats, data)

        if cumulative_stats["GP"] < min_gp:
            continue

        _finalize_leaderboard_stats(cumulative_stats, full_season_categories.get(goalie_key))

        league_data.append(cumulative_stats)

    df = pd.DataFrame(league_data)
    return df.sort_values("Total_SLC", ascending=False) if not df.empty else df


def add_hypothesis_traces(fig, df, goalie_name, color, row_offset=0):
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Pressure_Ratio"],
            mode="lines+markers",
            name=goalie_name,
            showlegend=False,
            line=dict(color=color, width=3),
            customdata=df[["SLC", "xGA_Per_60"]],
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Pressure Ratio: %{y:.3f}<br>"
                "SLC: %{customdata[0]:.3f}<br>"
                "xGA/60: %{customdata[1]:.3f}<extra></extra>"
            )
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["Reset_Ratio"],
            y=df["Red_Line_Shifts"],
            mode="markers",
            name=f"{goalie_name} Reset vs Red-Line",
            showlegend=False,
            marker=dict(color=color, size=10, opacity=0.75),
            text=df["Matchup"],
            customdata=df[["Date", "Avg_Sequence_Duration", "Sovereignty"]],
            hovertemplate=(
                "%{customdata[0]|%Y-%m-%d} %{text}<br>"
                "Reset Ratio: %{x:.3f}<br>"
                "Red-Line Sequences: %{y}<br>"
                "Avg Sequence: %{customdata[1]:.1f}s<br>"
                "Sovereignty: %{customdata[2]:.3f}<extra></extra>"
            )
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["P3_Hardware_Failures"],
            name=f"{goalie_name} P3 Failures",
            showlegend=False,
            marker_color=color,
            opacity=0.45,
            offsetgroup=row_offset,
            hovertemplate="%{x|%Y-%m-%d}<br>P3 Failures: %{y}<extra></extra>"
        ),
        row=3, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Early_SLC"],
            mode="lines",
            name=f"{goalie_name} P1-P2 SLC",
            showlegend=False,
            line=dict(color=color, dash="dash", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>P1-P2 SLC: %{y:.3f}<extra></extra>"
        ),
        row=3, col=1, secondary_y=True
    )


def build_rankings_scatter(leaderboard_df):
    slc_mid = leaderboard_df["Avg_SLC"].mean()
    sovereignty_mid = leaderboard_df["Sovereignty"].mean()
    x_min = min(leaderboard_df["Avg_SLC"].min(), slc_mid) - 0.15
    x_max = max(leaderboard_df["Avg_SLC"].max(), slc_mid) + 0.15
    y_min = min(leaderboard_df["Sovereignty"].min(), sovereignty_mid) - 0.5
    y_max = max(leaderboard_df["Sovereignty"].max(), sovereignty_mid) + 0.5

    color_map = {
        "Stabilizer": "#2ca02c",
        "Systems Man": "#1f77b4",
        "Functional Drain": "#ff7f0e",
        "Systemic Drain": "#d62728"
    }
    fig = go.Figure()
    for identity, identity_df in leaderboard_df.groupby("Identity"):
        fig.add_trace(
            go.Scatter(
                x=identity_df["Avg_SLC"],
                y=identity_df["Sovereignty"],
                mode="markers",
                name=identity,
                marker=dict(
                    color=color_map.get(identity, "#ff7f0e"),
                    size=9,
                    opacity=0.75,
                    line=dict(width=1, color="rgba(40,40,40,0.45)")
                ),
                customdata=identity_df[["Goalie", "GP", "Avg_SLC", "Identity"]],
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "Avg SLC: %{x:.3f}<br>"
                    "Sovereignty: %{y:.3f}<br>"
                    "GP: %{customdata[1]}<br>"
                    "Category: %{customdata[3]}<extra></extra>"
                )
            )
        )

    fig.add_vline(x=slc_mid, line_dash="dash", line_color="gray")
    fig.add_hline(y=sovereignty_mid, line_dash="dash", line_color="gray")
    fig.add_annotation(x=(slc_mid + x_max) / 2, y=(sovereignty_mid + y_max) / 2, text="Effective & Efficient", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(x_min + slc_mid) / 2, y=(sovereignty_mid + y_max) / 2, text="Accountable but Taxing", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(slc_mid + x_max) / 2, y=(y_min + sovereignty_mid) / 2, text="Active but Leaking", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(x_min + slc_mid) / 2, y=(y_min + sovereignty_mid) / 2, text="Drain on the Team", showarrow=False, font=dict(size=14))
    fig.update_layout(
        title="Who is helping and who is hurting?",
        height=550,
        xaxis_title="Avg SLC (Sequence Management)",
        yaxis_title="Sovereignty (Shot Quality Control)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=40, t=80, b=55)
    )
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])
    return fig


def render_chart_readout(kind, **kwargs):
    if kind == "selected_pressure":
        period_df = kwargs["period_df"]
        goalie_name = kwargs["goalie_name"]
        if period_df.empty:
            return
        slc_delta = period_df["SLC"].iloc[-1] - period_df["SLC"].iloc[0]
        xga_delta = period_df["xGA_Per_60"].iloc[-1] - period_df["xGA_Per_60"].iloc[0]
        if slc_delta < 0 and xga_delta > 0:
            st.warning(f"{goalie_name}'s game tilted toward stress: SLC fell by {slc_delta:.3f} while xGA/60 rose by {xga_delta:.3f}.")
        else:
            st.info(f"{goalie_name}'s period profile shows SLC movement of {slc_delta:.3f} against xGA/60 movement of {xga_delta:.3f}.")

    elif kind == "single_quadrant":
        goalie_df = kwargs["goalie_df"]
        selected_game_id = str(kwargs["selected_game_id"])
        selected = goalie_df[goalie_df["Game_ID"].astype(str) == selected_game_id]
        if selected.empty:
            return
        row = selected.iloc[0]
        reset_mid = goalie_df["Reset_Ratio"].median()
        redline_mid = goalie_df["Red_Line_Shifts"].median()
        if row["Reset_Ratio"] >= reset_mid and row["Red_Line_Shifts"] <= redline_mid:
            st.success("This game sits in the cleaner part of the context map: stronger reset work with fewer red-line sequences than this goalie's median.")
        else:
            st.info(f"This game produced a {row['Reset_Ratio']:.3f} reset ratio with {int(row['Red_Line_Shifts'])} red-line sequences, so read it against the season medians on the dashed lines.")

    elif kind == "comparison_quadrant":
        hyp_a = kwargs["hyp_a"]
        hyp_b = kwargs["hyp_b"]
        goalie_a = kwargs["goalie_a"]
        goalie_b = kwargs["goalie_b"]
        a_clean = (hyp_a["Reset_Ratio"].mean() - hyp_a["Red_Line_Shifts"].mean())
        b_clean = (hyp_b["Reset_Ratio"].mean() - hyp_b["Red_Line_Shifts"].mean())
        leader = goalie_a if a_clean >= b_clean else goalie_b
        st.info(f"The quadrant is separating reset control from red-line exposure. In this window, {leader} has the cleaner combined reset-versus-wall profile.")

    elif kind == "hypothesis":
        hyp_a = kwargs["hyp_a"]
        hyp_b = kwargs["hyp_b"]
        goalie_a = kwargs["goalie_a"]
        goalie_b = kwargs["goalie_b"]
        pressure_delta = hyp_a["Pressure_Ratio"].mean() - hyp_b["Pressure_Ratio"].mean()
        redline_delta = hyp_a["Red_Line_Shifts"].mean() - hyp_b["Red_Line_Shifts"].mean()
        if pressure_delta > 0 and redline_delta < 0:
            st.success(f"{goalie_a} is showing the cleaner pressure profile here: higher Pressure Ratio and fewer red-line sequences than {goalie_b}.")
        elif pressure_delta < 0 and redline_delta > 0:
            st.warning(f"{goalie_b} is carrying the cleaner pressure profile here: {goalie_a} is lower on Pressure Ratio and higher on red-line sequences.")
        else:
            st.info("The pressure profile is mixed: use the top line for efficiency, the middle plot for reset control, and the bottom plot for late-game payback signals.")

    elif kind == "trend":
        df_a = kwargs["df_a"]
        df_b = kwargs["df_b"]
        goalie_a = kwargs["goalie_a"]
        goalie_b = kwargs["goalie_b"]
        total_delta = df_a["Cumulative_SLC"].iloc[-1] - df_b["Cumulative_SLC"].iloc[-1]
        leader = goalie_a if total_delta >= 0 else goalie_b
        st.info(f"The momentum trend is cumulative SLC, so it rewards repeated stabilization over the window. {leader} finishes ahead by {abs(total_delta):.3f} cumulative SLC.")

    elif kind == "load_scatter":
        scatter_df = kwargs["scatter_df"]
        if scatter_df.empty:
            return
        best = scatter_df.sort_values(["Reset_Ratio", "Sovereignty"], ascending=False).iloc[0]
        st.info(f"The load-positioning view uses Reset Ratio as relief work and Sovereignty as shot-quality control. {best['Goalie']} is highest on that combined read among the selected goalies.")

    elif kind == "team_wall":
        metrics = kwargs["metrics"]
        team = kwargs["team"]
        if metrics["Efficiency"] >= 80:
            st.success(f"{team} stayed above the target lung-capacity line: {metrics['Efficiency']:.1f}% of sequences ended before the 40-second wall.")
        else:
            st.warning(f"{team} fell below the 80% target: only {metrics['Efficiency']:.1f}% of sequences ended before the 40-second wall.")


def build_quadrant_chart(df_a, df_b, goalie_a, goalie_b):
    quadrant_df = pd.concat([df_a, df_b], ignore_index=True)
    reset_mid = quadrant_df["Reset_Ratio"].median()
    redline_mid = quadrant_df["Red_Line_Shifts"].median()
    x_max = max(quadrant_df["Reset_Ratio"].max() * 1.1, reset_mid + 0.25, 1)
    y_max = max(quadrant_df["Red_Line_Shifts"].max() + 1, redline_mid + 1, 1)

    fig = go.Figure()
    for goalie_name, color in [(goalie_a, "green"), (goalie_b, "orange")]:
        goalie_df = quadrant_df[quadrant_df["Goalie"] == goalie_name]
        fig.add_trace(
            go.Scatter(
                x=goalie_df["Reset_Ratio"],
                y=goalie_df["Red_Line_Shifts"],
                mode="markers",
                name=goalie_name,
                marker=dict(
                    color=color,
                    size=goalie_df["xGA_Per_60"].clip(lower=0.2) * 8,
                    sizemode="diameter",
                    opacity=0.72,
                    line=dict(
                        width=goalie_df["P3_Hardware_Failures"].clip(upper=6) + 1,
                        color="rgba(30,30,30,0.55)"
                    )
                ),
                text=goalie_df["Matchup"],
                customdata=goalie_df[["Date", "SLC", "Sovereignty", "xGA_Per_60", "P3_Hardware_Failures", "Avg_Sequence_Duration"]],
                hovertemplate=(
                    "%{customdata[0]|%Y-%m-%d} %{text}<br>"
                    "Reset Ratio: %{x:.3f}<br>"
                    "Red-Line Sequences: %{y}<br>"
                    "SLC: %{customdata[1]:.3f}<br>"
                    "Sovereignty: %{customdata[2]:.3f}<br>"
                    "xGA/60: %{customdata[3]:.3f}<br>"
                    "P3 Failures: %{customdata[4]}<br>"
                    "Avg Sequence: %{customdata[5]:.1f}s<extra></extra>"
                )
            )
        )

    fig.add_vline(x=reset_mid, line_dash="dash", line_color="gray")
    fig.add_hline(y=redline_mid, line_dash="dash", line_color="gray")
    fig.add_annotation(x=reset_mid / 2, y=(redline_mid + y_max) / 2, text="Chaos Debt", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(reset_mid + x_max) / 2, y=(redline_mid + y_max) / 2, text="Busy Stabilizer", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=reset_mid / 2, y=redline_mid / 2, text="Hidden Risk", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(reset_mid + x_max) / 2, y=redline_mid / 2, text="Clean Stabilizer", showarrow=False, font=dict(size=14))

    fig.update_layout(
        height=560,
        xaxis_title="Reset Ratio: Saves / Sequences",
        yaxis_title="Red-Line Sequences >40s",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=40, t=70, b=50)
    )
    fig.update_xaxes(range=[0, x_max])
    fig.update_yaxes(range=[0, y_max])
    return fig


def build_selected_game_pressure_chart(period_df, goalie_name, matchup):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=period_df["Period"],
            y=period_df["SLC"],
            mode="lines+markers",
            name="SLC",
            line=dict(color="green", width=3),
            hovertemplate="%{x}<br>SLC: %{y:.3f}<extra></extra>"
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=period_df["Period"],
            y=period_df["xGA_Per_60"],
            mode="lines+markers",
            name="xGA/60",
            line=dict(color="orange", dash="dot", width=2),
            hovertemplate="%{x}<br>xGA/60: %{y:.3f}<extra></extra>"
        ),
        secondary_y=True
    )
    fig.update_layout(
        title=f"{goalie_name} vs Shot Quality: {matchup}",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=50, t=80, b=45)
    )
    fig.update_yaxes(title_text="SLC", secondary_y=False)
    fig.update_yaxes(title_text="xGA/60", secondary_y=True)
    fig.update_xaxes(title_text="Period")
    return fig


def build_single_goalie_quadrant_chart(goalie_df, selected_game_id, goalie_name, view_mode="Selected Game"):
    reset_mid = goalie_df["Reset_Ratio"].median()
    redline_mid = goalie_df["Red_Line_Shifts"].median()
    x_max = max(goalie_df["Reset_Ratio"].max() * 1.1, reset_mid + 0.25, 1)
    y_max = max(goalie_df["Red_Line_Shifts"].max() + 1, redline_mid + 1, 1)
    selected = goalie_df[goalie_df["Game_ID"].astype(str) == str(selected_game_id)]
    background = goalie_df[goalie_df["Game_ID"].astype(str) != str(selected_game_id)]

    fig = go.Figure()
    if view_mode == "Full Season":
        fig.add_trace(
            go.Scatter(
                x=goalie_df["Reset_Ratio"],
                y=goalie_df["Red_Line_Shifts"],
                mode="markers",
                name=f"{goalie_name} Full Season",
                marker=dict(
                    color="green",
                    size=goalie_df["xGA_Per_60"].clip(lower=0.2) * 7,
                    sizemode="diameter",
                    opacity=0.75,
                    line=dict(
                        width=goalie_df["P3_Hardware_Failures"].clip(upper=5) + 1,
                        color="rgba(30,30,30,0.55)"
                    )
                ),
                text=goalie_df["Matchup"],
                customdata=goalie_df[["Date", "SLC", "xGA_Per_60", "P3_Hardware_Failures"]],
                hovertemplate=(
                    "%{customdata[0]|%Y-%m-%d} %{text}<br>"
                    "Reset Ratio: %{x:.3f}<br>"
                    "Red-Line Sequences: %{y}<br>"
                    "SLC: %{customdata[1]:.3f}<br>"
                    "xGA/60: %{customdata[2]:.3f}<br>"
                    "P3 Failures: %{customdata[3]}<extra></extra>"
                )
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=background["Reset_Ratio"],
                y=background["Red_Line_Shifts"],
                mode="markers",
                name=f"{goalie_name} Games",
                marker=dict(color="rgba(120,120,120,0.45)", size=9),
                text=background["Matchup"],
                customdata=background[["Date", "SLC", "xGA_Per_60", "P3_Hardware_Failures"]],
                hovertemplate=(
                    "%{customdata[0]|%Y-%m-%d} %{text}<br>"
                    "Reset Ratio: %{x:.3f}<br>"
                    "Red-Line Sequences: %{y}<br>"
                    "SLC: %{customdata[1]:.3f}<br>"
                    "xGA/60: %{customdata[2]:.3f}<br>"
                    "P3 Failures: %{customdata[3]}<extra></extra>"
                )
            )
        )
        if not selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected["Reset_Ratio"],
                    y=selected["Red_Line_Shifts"],
                    mode="markers",
                    name="Selected Game",
                    marker=dict(color="green", size=18, line=dict(width=3, color="black")),
                    text=selected["Matchup"],
                    customdata=selected[["Date", "SLC", "xGA_Per_60", "P3_Hardware_Failures"]],
                    hovertemplate=(
                        "%{customdata[0]|%Y-%m-%d} %{text}<br>"
                        "Reset Ratio: %{x:.3f}<br>"
                        "Red-Line Sequences: %{y}<br>"
                        "SLC: %{customdata[1]:.3f}<br>"
                        "xGA/60: %{customdata[2]:.3f}<br>"
                        "P3 Failures: %{customdata[3]}<extra></extra>"
                    )
                )
            )

    fig.add_vline(x=reset_mid, line_dash="dash", line_color="gray")
    fig.add_hline(y=redline_mid, line_dash="dash", line_color="gray")
    fig.add_annotation(x=reset_mid / 2, y=(redline_mid + y_max) / 2, text="Chaos Debt", showarrow=False)
    fig.add_annotation(x=(reset_mid + x_max) / 2, y=(redline_mid + y_max) / 2, text="Busy Stabilizer", showarrow=False)
    fig.add_annotation(x=reset_mid / 2, y=redline_mid / 2, text="Hidden Risk", showarrow=False)
    fig.add_annotation(x=(reset_mid + x_max) / 2, y=redline_mid / 2, text="Clean Stabilizer", showarrow=False)
    fig.update_layout(
        title=f"{goalie_name} {'Full Season' if view_mode == 'Full Season' else 'Game Context'} Quadrant",
        height=480,
        xaxis_title="Reset Ratio: Saves / Sequences",
        yaxis_title="Red-Line Sequences >40s",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=40, t=80, b=50)
    )
    fig.update_xaxes(range=[0, x_max])
    fig.update_yaxes(range=[0, y_max])
    return fig


def build_team_wall_histogram(sequence_df, average_df, metrics, team_abbrev, game_label):
    max_duration = max(
        sequence_df["Duration"].max() if not sequence_df.empty else 0,
        average_df["Duration"].max() if not average_df.empty else 0,
        45
    )
    bin_edges = list(range(0, int(max_duration // 5 * 5) + 10, 5))
    current_counts = sequence_df["Bin_Start"].value_counts().reindex(bin_edges, fill_value=0).sort_index()
    labels = [f"{bin_start}-{bin_start + 5}s" for bin_start in current_counts.index]
    colors = [
        "#2ca02c" if bin_start < 25 else "#f2c94c" if bin_start < 40 else "#d62728"
        for bin_start in current_counts.index
    ]

    fig = go.Figure()
    if not average_df.empty:
        avg_counts = average_df["Bin_Start"].value_counts().reindex(bin_edges, fill_value=0).sort_index()
        avg_counts = avg_counts / max(len(get_team_games(team_abbrev)) - 1, 1)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=avg_counts,
                name="Season Avg Ghost",
                marker_color="rgba(120,120,120,0.28)",
                hovertemplate="%{x}<br>Avg sequences: %{y:.2f}<extra></extra>"
            )
        )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=current_counts.values,
            name="Current Game",
            marker_color=colors,
            hovertemplate="%{x}<br>Sequences: %{y}<extra></extra>"
        )
    )

    stress_counts = sequence_df[sequence_df["Stress"]]["Bin_Start"].value_counts().reindex(bin_edges, fill_value=0).sort_index()
    stress_x = [f"{bin_start}-{bin_start + 5}s" for bin_start, count in stress_counts.items() if count > 0]
    stress_y = [current_counts.loc[bin_start] + 0.35 for bin_start, count in stress_counts.items() if count > 0]
    stress_text = [f"Stress x{count}" for count in stress_counts if count > 0]
    if stress_x:
        fig.add_trace(
            go.Scatter(
                x=stress_x,
                y=stress_y,
                mode="text",
                text=["!" for _ in stress_x],
                name="Stress Marker",
                textfont=dict(color="#d62728", size=18),
                hovertext=stress_text,
                hovertemplate="%{x}<br>%{hovertext}<extra></extra>"
            )
        )

    median_label = f"{int(metrics['Median'] // 5) * 5}-{int(metrics['Median'] // 5) * 5 + 5}s"
    if median_label in labels:
        fig.add_vline(x=labels.index(median_label), line_dash="dash", line_color="black")

    fig.add_annotation(
        x=0.01,
        y=1.08,
        xref="paper",
        yref="paper",
        text=f"Efficiency Under 40s: {metrics['Efficiency']:.1f}% | Median: {metrics['Median']:.1f}s | Sequences: {metrics['Total']}",
        showarrow=False,
        align="left",
        font=dict(size=14)
    )
    fig.update_layout(
        title=f"{team_abbrev} 40-Second Wall: {game_label}",
        height=560,
        xaxis_title="Sequence Duration",
        yaxis_title="Frequency",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=40, t=100, b=70)
    )
    return fig


@st.cache_data(show_spinner=False)
def build_team_season_redline_data(team_abbrev, game_signature):
    rows = []
    for game_id, game_date, matchup, label in game_signature:
        sequence_df = get_team_sequence_data(team_abbrev, game_id, phase="All")
        if sequence_df.empty:
            continue

        total_sequences = len(sequence_df)
        redline_sequences = int((sequence_df["Duration"] > 40).sum())
        under_40 = total_sequences - redline_sequences
        rows.append({
            "Game_ID": str(game_id),
            "Date": game_date,
            "Matchup": matchup,
            "Label": label,
            "Red_Line_Shifts": redline_sequences,
            "Efficiency": round((under_40 / total_sequences) * 100, 1) if total_sequences else 0.0,
            "Median_Sequence": round(float(sequence_df["Duration"].median()), 1) if total_sequences else 0.0,
            "Total_Sequences": total_sequences,
            "Max_Sequence": round(float(sequence_df["Duration"].max()), 1) if total_sequences else 0.0
        })

    if not rows:
        return pd.DataFrame()

    trend_df = pd.DataFrame(rows)
    trend_df["Date"] = pd.to_datetime(trend_df["Date"], errors="coerce")
    return trend_df.sort_values("Date")


def build_monthly_team_redline_data(trend_df):
    if trend_df.empty:
        return trend_df

    monthly_df = trend_df.copy()
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"], errors="coerce")
    monthly_df = monthly_df.dropna(subset=["Date"])
    if monthly_df.empty:
        return monthly_df

    monthly_df["Month"] = monthly_df["Date"].dt.to_period("M").dt.to_timestamp()
    grouped = monthly_df.groupby("Month", as_index=False).agg(
        Games=("Game_ID", "count"),
        Red_Line_Shifts=("Red_Line_Shifts", "mean"),
        Red_Line_Total=("Red_Line_Shifts", "sum"),
        Efficiency=("Efficiency", "mean"),
        Median_Sequence=("Median_Sequence", "mean"),
        Total_Sequences=("Total_Sequences", "mean"),
        Max_Sequence=("Max_Sequence", "max")
    )
    grouped["Month_Label"] = grouped["Month"].dt.strftime("%b %Y")
    grouped["Red_Line_Shifts"] = grouped["Red_Line_Shifts"].round(2)
    grouped["Efficiency"] = grouped["Efficiency"].round(1)
    grouped["Median_Sequence"] = grouped["Median_Sequence"].round(1)
    grouped["Total_Sequences"] = grouped["Total_Sequences"].round(1)
    grouped["Max_Sequence"] = grouped["Max_Sequence"].round(1)
    return grouped.sort_values("Month")


def build_team_redline_trend_chart(monthly_df, team_abbrev, game_phase):
    avg_redline = monthly_df["Red_Line_Shifts"].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly_df["Month"],
            y=monthly_df["Red_Line_Shifts"],
            mode="lines+markers",
            name="Monthly Avg 40s+ Shifts",
            line=dict(color="#d62728", width=3),
            marker=dict(size=11),
            text=monthly_df["Month_Label"],
            customdata=monthly_df[["Games", "Red_Line_Total", "Efficiency", "Median_Sequence", "Total_Sequences", "Max_Sequence"]],
            hovertemplate=(
                "%{text}<br>"
                "Games: %{customdata[0]}<br>"
                "Avg 40s+ Shifts/Game: %{y:.2f}<br>"
                "Total 40s+ Shifts: %{customdata[1]}<br>"
                "Avg Efficiency: %{customdata[2]:.1f}%<br>"
                "Avg Median Sequence: %{customdata[3]:.1f}s<br>"
                "Avg Total Sequences: %{customdata[4]:.1f}<br>"
                "Longest Sequence: %{customdata[5]:.1f}s<extra></extra>"
            )
        )
    )
    fig.add_hline(
        y=avg_redline,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Monthly Avg: {avg_redline:.1f}",
        annotation_position="top left"
    )
    fig.update_layout(
        title=f"{team_abbrev} Monthly 40-Second Wall Trend: {game_phase}",
        height=430,
        xaxis_title="Month",
        yaxis_title="Avg Shifts Over 40 Seconds per Game",
        margin=dict(l=50, r=40, t=80, b=50)
    )
    return fig


@st.cache_data(show_spinner=False)
def build_team_summary_table(game_phase, data_version):
    teams = get_available_teams()
    master_report = load_master_reports()
    team_stats = {
        team: {
            "Team": team,
            "GP": 0,
            "SLC": 0.0,
            "S_saves": 0,
            "S_goals": 0,
            "xS": 0.0,
            "xG": 0.0,
            "Service": 0.0,
            "Tax": 0.0,
            "RedLine_Shifts": 0,
            "RedLine_Games": 0
        }
        for team in teams
    }

    for games in master_report.values():
        for game_id, report in games.items():
            if not _matches_game_phase(_get_report_date(game_id, report), game_phase, game_id):
                continue

            raw_game = _load_raw_game_for_dashboard(game_id)
            team = _goalie_team_abbrev_from_raw(raw_game, report.get("goalie_id"))
            if team not in team_stats:
                continue

            total = report.get("total", {})
            stats = total.get("stats", {})
            service, tax = calculate_service_tax(stats, include_baseline=True)

            team_stats[team]["GP"] += 1
            team_stats[team]["SLC"] += current_report_score(report)
            team_stats[team]["S_saves"] += stats.get("S_saves", 0)
            team_stats[team]["S_goals"] += stats.get("S_goals", 0)
            team_stats[team]["xS"] += stats.get("xS", 0)
            team_stats[team]["xG"] += stats.get("xG", 0)
            team_stats[team]["Service"] += service
            team_stats[team]["Tax"] += tax

    for team in teams:
        team_games = filter_team_games_by_phase(get_team_games(team), game_phase)
        for game in team_games:
            sequence_df = get_team_sequence_data(team, game["Game_ID"], phase="All")
            if sequence_df.empty:
                continue
            team_stats[team]["RedLine_Shifts"] += int((sequence_df["Duration"] > 40).sum())
            team_stats[team]["RedLine_Games"] += 1

    rows = []
    for stats in team_stats.values():
        if stats["GP"] == 0 and stats["RedLine_Games"] == 0:
            continue

        shots = stats["S_saves"] + stats["S_goals"]
        actual_sv = stats["S_saves"] / shots if shots else 0
        expected_saves = stats["xS"] if stats["xS"] else shots - stats["xG"]
        expected_sv = expected_saves / shots if shots else 0
        rows.append({
            "Team": stats["Team"],
            "GP": int(stats["GP"]),
            "SLC": round(stats["SLC"], 3),
            "SLC Avg": round(stats["SLC"] / stats["GP"], 3) if stats["GP"] else 0,
            "Sovereignty": round((actual_sv - expected_sv) * 100, 3) if shots else 0,
            "Net Load": round(stats["Service"] - stats["Tax"], 3),
            "RedLine Shifts": int(stats["RedLine_Shifts"]),
            "RedLine Shifts Avg": round(stats["RedLine_Shifts"] / stats["RedLine_Games"], 2) if stats["RedLine_Games"] else 0
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("SLC", ascending=False)


def build_team_integrity_matrix(team_summary):
    if team_summary.empty:
        return go.Figure()

    chart_df = team_summary.copy()
    redline_avg = chart_df["RedLine Shifts Avg"].mean()
    slc_avg = chart_df["SLC Avg"].mean()

    def classify(row):
        if row["RedLine Shifts Avg"] < redline_avg and row["SLC Avg"] > slc_avg:
            return "Stabilizers"
        if row["RedLine Shifts Avg"] > redline_avg and row["SLC Avg"] > slc_avg:
            return "High-Load Heroes"
        if row["RedLine Shifts Avg"] < redline_avg and row["SLC Avg"] < slc_avg:
            return "Passive Drift"
        return "Systemic Collapse"

    chart_df["Integrity"] = chart_df.apply(classify, axis=1)
    colors = {
        "Stabilizers": "#2ca02c",
        "High-Load Heroes": "#1f77b4",
        "Passive Drift": "#ffbf00",
        "Systemic Collapse": "#d62728"
    }

    x_min = min(chart_df["RedLine Shifts Avg"].min(), redline_avg) - 0.75
    x_max = max(chart_df["RedLine Shifts Avg"].max(), redline_avg) + 0.75
    y_min = min(chart_df["SLC Avg"].min(), slc_avg) - 0.15
    y_max = max(chart_df["SLC Avg"].max(), slc_avg) + 0.15

    fig = go.Figure()
    for integrity, group in chart_df.groupby("Integrity"):
        fig.add_trace(
            go.Scatter(
                x=group["RedLine Shifts Avg"],
                y=group["SLC Avg"],
                mode="markers+text",
                text=group["Team"],
                textposition="top center",
                name=integrity,
                marker=dict(
                    color=colors.get(integrity, "#7f7f7f"),
                    size=13,
                    opacity=0.82,
                    line=dict(width=1, color="rgba(35,35,35,0.55)")
                ),
                customdata=group[["Team", "SLC", "Sovereignty", "Net Load", "RedLine Shifts"]],
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "RedLine Shifts Avg: %{x:.2f}<br>"
                    "SLC Avg: %{y:.3f}<br>"
                    "Total SLC: %{customdata[1]:.3f}<br>"
                    "Sovereignty: %{customdata[2]:.3f}<br>"
                    "Net Load: %{customdata[3]:.1f}<br>"
                    "Cumulative RedLine Shifts: %{customdata[4]}<extra></extra>"
                )
            )
        )

    fig.add_vline(x=redline_avg, line_dash="dash", line_color="gray")
    fig.add_hline(y=slc_avg, line_dash="dash", line_color="gray")
    fig.add_annotation(x=(x_min + redline_avg) / 2, y=(slc_avg + y_max) / 2, text="Stabilizers", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(redline_avg + x_max) / 2, y=(slc_avg + y_max) / 2, text="High-Load Heroes", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(x_min + redline_avg) / 2, y=(y_min + slc_avg) / 2, text="Passive Drift", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=(redline_avg + x_max) / 2, y=(y_min + slc_avg) / 2, text="Systemic Collapse", showarrow=False, font=dict(size=14))
    fig.update_layout(
        title="Defensive Wall Integrity Matrix",
        height=560,
        xaxis_title="RedLine Shifts Avg (lower is better)",
        yaxis_title="SLC Avg",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=55, r=45, t=90, b=60)
    )
    fig.update_xaxes(range=[x_max, x_min])
    fig.update_yaxes(range=[y_min, y_max])
    return fig


def build_high_interest_loan_chart(selected_row, season_df, components_df, goalie_name):
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.36, 0.64],
        subplot_titles=("P1-P2 Biological Tax Ledger", "Season Context: Early Debt vs P3 Sovereignty"),
        specs=[[{}, {}]]
    )

    fig.add_trace(
        go.Bar(
            x=components_df["Component"],
            y=components_df["Value"],
            name="Debt Components",
            marker_color=["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#1f77b4"],
            hovertemplate="%{x}<br>Debt Units: %{y:.1f}<extra></extra>"
        ),
        row=1,
        col=1
    )

    other_games = season_df[season_df["Game_ID"].astype(str) != str(selected_row["Game_ID"])]
    if not other_games.empty:
        fig.add_trace(
            go.Scatter(
                x=other_games["Early_Debt"],
                y=other_games["P3_Sovereignty"],
                mode="markers",
                name="Season Games",
                marker=dict(
                    color="rgba(120,120,120,0.45)",
                    size=other_games["P3_GA"].clip(lower=0) * 5 + 8,
                    line=dict(width=1, color="rgba(60,60,60,0.35)")
                ),
                text=other_games["Matchup"],
                customdata=other_games[["Date", "P3_GA", "Early_Red_Line", "P3_SLC"]],
                hovertemplate=(
                    "%{customdata[0]} %{text}<br>"
                    "Early Debt: %{x:.1f}<br>"
                    "P3 Sovereignty: %{y:.3f}<br>"
                    "P3 GA: %{customdata[1]}<br>"
                    "40s+ Early Sequences: %{customdata[2]}<br>"
                    "P3 SLC: %{customdata[3]:.3f}<extra></extra>"
                )
            ),
            row=1,
            col=2
        )

    fig.add_trace(
        go.Scatter(
            x=[selected_row["Early_Debt"]],
            y=[selected_row["P3_Sovereignty"]],
            mode="markers+text",
            name="Selected Game",
            text=["Selected"],
            textposition="top center",
            marker=dict(color="#d62728", size=18, line=dict(width=3, color="black")),
            hovertemplate=(
                f"{selected_row['Date']} {selected_row['Matchup']}<br>"
                "Early Debt: %{x:.1f}<br>"
                "P3 Sovereignty: %{y:.3f}<extra></extra>"
            )
        ),
        row=1,
        col=2
    )

    if len(season_df) > 1:
        debt_mid = season_df["Early_Debt"].median()
        sov_mid = season_df["P3_Sovereignty"].median()
        fig.add_vline(x=debt_mid, line_dash="dash", line_color="gray", row=1, col=2)
        fig.add_hline(y=sov_mid, line_dash="dash", line_color="gray", row=1, col=2)

    fig.update_layout(
        title=f"{goalie_name} High-Interest Loan Game Log",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=40, t=90, b=60)
    )
    fig.update_yaxes(title_text="Debt Units", row=1, col=1)
    fig.update_xaxes(title_text="Component", row=1, col=1)
    fig.update_xaxes(title_text="P1-P2 Biological Debt", row=1, col=2)
    fig.update_yaxes(title_text="3rd Period Sovereignty", row=1, col=2)
    return fig


def render_team_goalie_verdict(selected_game, metrics):
    master = load_master_reports()
    game_goalie_name = None
    game_goalie_report = None
    for goalie_key, games in master.items():
        if str(selected_game["Game_ID"]) in games:
            game_goalie_name = goalie_key.replace("_", " ").title()
            game_goalie_report = games[str(selected_game["Game_ID"])]
            break

    if game_goalie_name and game_goalie_report:
        total_stats = game_goalie_report.get("total", {}).get("stats", {})
        npw = total_stats.get("NPW", 0)
        ua = total_stats.get("UA", 0)
        rp = total_stats.get("RP", 0)
        igva = total_stats.get("iGvA", 0) + total_stats.get("iGvA_SH", 0)
        defensive_load = ua + rp

        if metrics["Efficiency"] >= 75 and npw >= defensive_load:
            st.success(
                f"**{game_goalie_name} helped the team tonight.** "
                f"The team cleared {metrics['Efficiency']:.0f}% of sequences before the 40-second wall. "
                f"{game_goalie_name} generated {npw} whistles and kept rebound situations controlled."
            )
        elif metrics["Efficiency"] < 60 and (rp + igva) > npw:
            st.error(
                f"**{game_goalie_name} added to the team's burden tonight.** "
                f"Only {metrics['Efficiency']:.0f}% of sequences ended before the wall. "
                f"{game_goalie_name} gave up {rp} rebound situations and {igva} giveaways, extending shifts that should have ended."
            )
        else:
            st.warning(
                f"**{game_goalie_name} had a mixed night.** "
                f"{metrics['Efficiency']:.0f}% of sequences stayed under the wall. "
                f"Some reset work ({npw} whistles) but also {rp} rebounds and {igva} giveaways putting the defense back to work."
            )
    else:
        st.info("No goalie SLC data found for this game. Run the goalie through the data pipeline first.")


def render_download_buttons(base_name, summary_text, csv_df=None, key_prefix="download"):
    c_txt, c_csv = st.columns(2)
    with c_txt:
        st.download_button(
            "Download printable summary",
            data=summary_text,
            file_name=f"{base_name}.txt",
            mime="text/plain",
            key=f"{key_prefix}_txt"
        )
    if csv_df is not None and not csv_df.empty:
        with c_csv:
            st.download_button(
                "Download data CSV",
                data=dataframe_csv(csv_df),
                file_name=f"{base_name}.csv",
                mime="text/csv",
                key=f"{key_prefix}_csv"
            )


def build_comparison_export_table(goalie_a, goalie_b, hyp_a, hyp_b):
    frames = []
    export_columns = [
        "Goalie", "Date", "Matchup", "Game_ID", "SLC", "Sovereignty",
        "Service", "Tax", "Reset_Ratio", "Pressure_Ratio", "xGA_Per_60",
        "Red_Line_Shifts", "Avg_Sequence_Duration", "Early_SLC",
        "P3_Hardware_Failures", "P3_Tax"
    ]

    for goalie_name, df in [(goalie_a, hyp_a), (goalie_b, hyp_b)]:
        if df.empty:
            continue
        export_df = df.copy()
        export_df["Goalie"] = goalie_name
        for column in export_columns:
            if column not in export_df.columns:
                export_df[column] = None
        frames.append(export_df[export_columns])

    if not frames:
        return pd.DataFrame(columns=export_columns)

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return combined.sort_values(["Date", "Goalie"])


def run_dashboard_data_update(game_type):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        summary = update_local_data(season="20252026", game_type=game_type)
    return summary or {}, buffer.getvalue()

# --- 4. MAIN UI EXECUTION ---

st.set_page_config(layout="wide", page_title="SLC Comparative Engine")
st.title("🏒 SLC Comparative Analysis Engine")

# Global State
top_phase, top_update_scope, top_update_button = st.columns([2, 2, 1])
with top_phase:
    game_phase = st.radio("Game Set", ["Regular Season", "Playoffs"], horizontal=True)
with top_update_scope:
    update_scope = st.selectbox("Update Scope", ["All Completed", "Regular Season", "Playoffs"])
with top_update_button:
    st.write("")
    st.write("")
    update_requested = st.button("Update Local Data", type="primary")

if update_requested:
    game_type = {"All Completed": "all", "Regular Season": 2, "Playoffs": 3}[update_scope]
    with st.spinner("Fetching missing games and updating local reports. This can take a few minutes."):
        try:
            update_summary, update_log = run_dashboard_data_update(game_type)
            st.cache_data.clear()
            st.success(
                "Local data update complete. "
                f"Fetched {update_summary.get('raw_games_fetched', 0)} raw games and wrote "
                f"{update_summary.get('processed_reports', 0)} new goalie reports."
            )
            with st.expander("Update log"):
                st.text(update_log)
        except Exception as exc:
            st.error(f"Local data update failed: {exc}")

leaderboard = build_phase_leaderboard(game_phase, min_gp=1)
goalie_list = sorted(leaderboard["Goalie"].tolist()) if not leaderboard.empty else ["No goalies available"]

tab_rank, tab_audit, tab_compare, tab_team = st.tabs(["Rankings", "Individual Audit", "Dual-Goalie Comparison", "Team Redline"])

with tab_rank:
    st.header(f"League Leaderboard: {game_phase}")
    leaderboard_min_gp = 20 if game_phase == "Regular Season" else 1
    rank_leaderboard = build_phase_leaderboard(game_phase, min_gp=leaderboard_min_gp)

    if rank_leaderboard.empty:
        st.warning("Leaderboard data is not available.")
    else:
        if game_phase == "Regular Season":
            st.caption("Regular-season leaderboard requires at least 20 games played. Quadrants are centered on the visible league averages.")
        else:
            st.caption("Playoff leaderboard does not use a 20-game minimum. Quadrants are centered on the visible league averages.")

        st.plotly_chart(build_rankings_scatter(rank_leaderboard), width="stretch")
        st.markdown("### Top Goalies by Total SLC")
        leaderboard_display = rank_leaderboard[
            ["Goalie", "GP", "Total_SLC", "Sovereignty", "Net_Load", "Service", "Tax", "Reset_Ratio", "Category"]
        ].rename(columns={
            "Total_SLC": "Total SLC",
            "Net_Load": "Net Load",
            "Reset_Ratio": "Reset Ratio"
        })
        leaderboard_display.index = range(1, len(leaderboard_display) + 1)
        leaderboard_display.index.name = "Place"

        st.dataframe(
            leaderboard_display,
            width="stretch",
            column_config={
                "_index": st.column_config.NumberColumn("Place")
            }
        )
        st.markdown("---")
        st.markdown("#### Team Summary")
        team_summary = build_team_summary_table(game_phase, get_master_report_version())
        if team_summary.empty:
            st.info("No team summary data is available for this game set.")
        else:
            st.plotly_chart(build_team_integrity_matrix(team_summary), width="stretch")
            st.caption(
                "Right side = fewer 40-second wall breaches. Top = stronger goalie SLC response. "
                "Dashed lines are league averages for the selected game set."
            )
            team_summary_display = team_summary.copy()
            team_summary_display.index = range(1, len(team_summary_display) + 1)
            team_summary_display.index.name = "Place"
            st.dataframe(
                team_summary_display,
                width="stretch",
                column_config={
                    "_index": st.column_config.NumberColumn("Place")
                }
            )
            st.download_button(
                "Download team summary CSV",
                data=dataframe_csv(team_summary),
                file_name=f"slc_team_summary_{game_phase.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"team_summary_{game_phase}"
            )
        render_download_buttons(
            f"slc_rankings_{game_phase.lower().replace(' ', '_')}",
            rankings_summary(rank_leaderboard, game_phase),
            rank_leaderboard,
            key_prefix=f"rankings_{game_phase}"
        )

with tab_audit:
    st.header(f"Individual Goalie Audit: {game_phase}")
    if leaderboard.empty:
        st.warning(f"No goalie metadata is available for {game_phase.lower()} games.")
    else:
        selected_goalie = st.selectbox("Select a goalie", goalie_list)
        goalie_key = selected_goalie.lower().replace(" ", "_")
        goalie_df = filter_by_game_phase(load_goalie_data(goalie_key), game_phase)

        if goalie_df.empty:
            st.warning(f"No game history found for {selected_goalie}.")
        else:
            render_goalie_header(selected_goalie, leaderboard, "audit")
            goalie_row = leaderboard[leaderboard["Goalie"] == selected_goalie].iloc[0]
            goalie_category = goalie_row.get("Category", goalie_row["Identity"])
            if goalie_category == "Stabilizer":
                st.success(f"{selected_goalie} is a Stabilizer: effective and efficient across the full season. They control shot quality and reduce defensive load.")
            elif goalie_category == "Systems Man":
                st.info(f"{selected_goalie} is a Systems Man: accountable but taxing across the full season. They manage sequences, but shot-quality results can leak.")
            elif goalie_category == "Functional Drain":
                st.warning(f"{selected_goalie} is a Functional Drain: active but leaking across the full season. The talent is visible, but the workflow keeps pressure alive.")
            elif goalie_category == "Systemic Drain":
                st.error(f"{selected_goalie} is a Systemic Drain: inefficient and ineffective across the full season. The profile shows both sequence strain and shot-quality leakage.")

            st.markdown("---")
            st.subheader("Game History")
            game_history_columns = ["Date", "Matchup", "SLC", "Sovereignty", "Net_Load", "Service", "Tax", "NPW", "UA", "RP"]
            st.dataframe(goalie_df.sort_values("Date", ascending=False)[game_history_columns], width="stretch")

            valid_game_dates = pd.to_datetime(goalie_df["Date"], errors="coerce").dropna()
            if valid_game_dates.empty:
                st.info("Date filtering is unavailable because these game reports do not have valid dates yet.")
            else:
                selected_date = st.date_input(
                    "Filter games by date:",
                    value=None,
                    min_value=valid_game_dates.min().date(),
                    max_value=valid_game_dates.max().date(),
                    help="Pick a date to filter the game history table."
                )
                if selected_date:
                    date_str = selected_date.strftime("%Y-%m-%d")
                    filtered = goalie_df[goalie_df['Date'].astype(str).str.startswith(date_str)]
                    if filtered.empty:
                        st.info(f"No games found for {selected_goalie} on {date_str}.")
                    else:
                        st.subheader(f"Games on {date_str}")
                        st.dataframe(filtered[game_history_columns], width="stretch")
                        st.success(f"Showing {len(filtered)} game(s) for {selected_goalie}.")

            st.markdown("---")
            st.subheader("Selected Game Pressure View")
            game_options = goalie_df.sort_values("Date", ascending=False).copy()
            game_options["Game_Label"] = game_options["Date"].astype(str) + " " + game_options["Matchup"].astype(str)
            game_select_col, quadrant_mode_col = st.columns([2, 1])
            with game_select_col:
                selected_game_label = st.selectbox("Select a game", game_options["Game_Label"].tolist())
            with quadrant_mode_col:
                quadrant_view_mode = st.radio("Quadrant View", ["Selected Game", "Full Season"], horizontal=True)
            selected_game = game_options[game_options["Game_Label"] == selected_game_label].iloc[0]
            selected_game_id = str(selected_game["Game_ID"])
            selected_matchup = selected_game["Matchup"]
            period_df = build_game_slc_xga_data(goalie_key, selected_game_id)
            goalie_hypothesis_df = filter_by_game_phase(
                build_slc_hypothesis_data(goalie_key, selected_goalie),
                game_phase
            )
            loan_row, loan_season_df, loan_components_df = build_high_interest_loan_data(goalie_key, selected_game_id)
            loan_season_df = filter_by_game_phase(loan_season_df, game_phase)

            if not period_df.empty:
                st.plotly_chart(
                    build_selected_game_pressure_chart(period_df, selected_goalie, selected_matchup),
                    width="stretch"
                )
                st.caption("Green line rising = goalie getting more effective as game goes on. Orange line rising = goalie facing harder shots. If orange rises and green falls, the goalie is being broken down.")
                render_chart_readout("selected_pressure", period_df=period_df, goalie_name=selected_goalie)
            else:
                st.info("Period-level pressure data is not available for that game.")

            if not goalie_hypothesis_df.empty:
                st.plotly_chart(
                    build_single_goalie_quadrant_chart(
                        goalie_hypothesis_df,
                        selected_game_id,
                        selected_goalie,
                        quadrant_view_mode
                    ),
                    width="stretch"
                )
                st.caption("Each dot is one game. Bottom-right is good: high reset work, few red-line sequences. Top-left is bad: the goalie is letting sequences run long.")
                render_chart_readout(
                    "single_quadrant",
                    goalie_df=goalie_hypothesis_df,
                    selected_game_id=selected_game_id
                )

            st.subheader("High-Interest Loan Game Log")
            if loan_row and not loan_season_df.empty and not loan_components_df.empty:
                l1, l2, l3, l4 = st.columns(4)
                l1.metric("P1-P2 Biological Debt", f"{loan_row['Early_Debt']:.1f}", loan_row["Debt_Label"])
                l2.metric("P1-P2 40s+ Sequences", int(loan_row["Early_Red_Line"]))
                l3.metric("3rd Period Sovereignty", f"{loan_row['P3_Sovereignty']:.3f}")
                l4.metric("3rd Period GA", int(loan_row["P3_GA"]), delta_color="inverse")

                st.plotly_chart(
                    build_high_interest_loan_chart(
                        loan_row,
                        loan_season_df,
                        loan_components_df,
                        selected_goalie
                    ),
                    width="stretch"
                )
                st.caption("Left panel: what created the debt in the first two periods. Right panel: did that debt get paid in the third? High early debt with negative third-period Sovereignty means the team paid for it.")

                if loan_row["Early_Debt"] >= 4 and (loan_row["P3_Sovereignty"] < 0 or loan_row["P3_GA"] > 0):
                    st.warning(
                        "This team did not simply collapse in the 3rd. The first 40 minutes show a high-interest fatigue loan: "
                        f"{loan_row['Early_Debt']:.1f} early debt units, {loan_row['Early_Red_Line']} red-line sequences, "
                        f"then {loan_row['P3_GA']} GA with {loan_row['P3_Sovereignty']:.3f} 3rd-period Sovereignty."
                    )
                else:
                    st.success(
                        "The early load stayed comparatively managed, so the 3rd-period result is less clearly tied to fatigue debt."
                    )
            else:
                st.info("High-interest loan data is not available for that game.")

            st.markdown("---")
            render_download_buttons(
                f"slc_audit_{goalie_key}_{game_phase.lower().replace(' ', '_')}",
                goalie_audit_summary(
                    selected_goalie,
                    game_phase,
                    goalie_row,
                    goalie_df,
                    selected_game,
                    period_df,
                    loan_row
                ),
                goalie_df,
                key_prefix=f"audit_{goalie_key}_{game_phase}"
            )

with tab_compare:
    # 1. Selection & Temporal Window
    c_sel1, c_sel2, c_sel3 = st.columns([1, 1, 2])
    with c_sel1: goalie_a = st.selectbox("Player A", goalie_list, index=0)
    with c_sel2: goalie_b = st.selectbox("Player B", goalie_list, index=min(1, len(goalie_list)-1))
    with c_sel3:
        t_window = st.radio("Temporal Window", ["Season-to-Date", "Date Range", "Single Game"], horizontal=True)
        if t_window == "Date Range":
            window_value = st.date_input("Select Range", value=(), key="compare_date_range")
        elif t_window == "Single Game":
            window_value = st.date_input("Select Game Date", key="compare_single_date")
        else:
            window_value = None
        start_d, end_d = get_comparison_window(t_window, window_value)

    # 2. Standard Bio-Headers
    h_a, h_b = st.columns(2)
    with h_a: render_goalie_header(goalie_a, leaderboard, "A")
    with h_b: render_goalie_header(goalie_b, leaderboard, "B")

    goalie_a_key = goalie_a.lower().replace(" ", "_")
    goalie_b_key = goalie_b.lower().replace(" ", "_")
    base_hyp_a = filter_by_game_phase(build_slc_hypothesis_data(goalie_a_key, goalie_a), game_phase)
    base_hyp_b = filter_by_game_phase(build_slc_hypothesis_data(goalie_b_key, goalie_b), game_phase)
    hyp_a = filter_hypothesis_window(base_hyp_a, start_d, end_d)
    hyp_b = filter_hypothesis_window(base_hyp_b, start_d, end_d)

    if t_window == "Date Range" and len(window_value) != 2:
        st.info("Pick both a start date and an end date to narrow the comparison window. Until then, the comparison uses the full selected game set.")

    st.caption(
        f"Comparison window: {pd.Timestamp(start_d).date()} to {pd.Timestamp(end_d).date()} | "
        f"{goalie_a}: {len(hyp_a)} of {len(base_hyp_a)} {game_phase.lower()} games | "
        f"{goalie_b}: {len(hyp_b)} of {len(base_hyp_b)} {game_phase.lower()} games"
    )

    comparison_table = build_comparison_export_table(goalie_a, goalie_b, hyp_a, hyp_b)
    with st.expander("Comparison Summary Table", expanded=False):
        if comparison_table.empty:
            st.info("No comparison rows are available for the selected filters.")
        else:
            st.dataframe(comparison_table, width="stretch")
            st.download_button(
                "Download full comparison table",
                data=dataframe_csv(comparison_table),
                file_name=(
                    f"slc_comparison_table_{goalie_a_key}_vs_{goalie_b_key}_"
                    f"{game_phase.lower().replace(' ', '_')}.csv"
                ),
                mime="text/csv",
                key=f"compare_table_{goalie_a_key}_{goalie_b_key}_{game_phase}"
            )

    if not hyp_a.empty and not hyp_b.empty:
        a_pressure = hyp_a["Pressure_Ratio"].mean()
        b_pressure = hyp_b["Pressure_Ratio"].mean()
        a_reset = hyp_a["Reset_Ratio"].mean()
        b_reset = hyp_b["Reset_Ratio"].mean()
        a_redline = hyp_a["Red_Line_Shifts"].mean()
        b_redline = hyp_b["Red_Line_Shifts"].mean()

        a_score = (a_pressure + a_reset) - a_redline
        b_score = (b_pressure + b_reset) - b_redline

        if a_score > b_score:
            st.success(f"In this window, **{goalie_a}** is the more effective and less taxing goalie. Higher pressure efficiency, better reset work, fewer red-line sequences.")
        elif b_score > a_score:
            st.success(f"In this window, **{goalie_b}** is the more effective and less taxing goalie. Higher pressure efficiency, better reset work, fewer red-line sequences.")
        else:
            st.info("In this window, both goalies are producing comparable load profiles.")

    st.subheader("SLC Load Hypothesis Map")

    if not hyp_a.empty and not hyp_b.empty:
        st.plotly_chart(
            build_quadrant_chart(hyp_a, hyp_b, goalie_a, goalie_b),
            width="stretch"
        )
        st.caption("Each dot is one game. Right side = more reset work. Bottom = fewer red-line sequences. Bottom-right corner is what you want to see.")
        render_chart_readout(
            "comparison_quadrant",
            hyp_a=hyp_a,
            hyp_b=hyp_b,
            goalie_a=goalie_a,
            goalie_b=goalie_b
        )

        fig_hypothesis = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.11,
            subplot_titles=(
                "Pressure Ratio: SLC / xGA per 60",
                "Reset Ratio vs Red-Line Sequences",
                "Early SLC Debt vs 3rd Period Failures"
            ),
            specs=[
                [{}],
                [{}],
                [{"secondary_y": True}]
            ]
        )

        add_hypothesis_traces(fig_hypothesis, hyp_a, goalie_a, "green", 0)
        add_hypothesis_traces(fig_hypothesis, hyp_b, goalie_b, "orange", 1)

        fig_hypothesis.update_yaxes(title_text="Pressure Ratio", row=1, col=1)
        fig_hypothesis.update_xaxes(title_text="Game Date", row=1, col=1)
        fig_hypothesis.update_xaxes(title_text="Reset Ratio", row=2, col=1)
        fig_hypothesis.update_yaxes(title_text="Red-Line Sequences >40s", row=2, col=1)
        fig_hypothesis.update_yaxes(title_text="P3 Failures", row=3, col=1, secondary_y=False)
        fig_hypothesis.update_yaxes(title_text="P1-P2 SLC", row=3, col=1, secondary_y=True)
        fig_hypothesis.update_layout(
            height=900,
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=50),
            barmode="group"
        )
        st.plotly_chart(fig_hypothesis, width="stretch")
        st.caption("Top: efficiency under pressure. Middle: reset control vs. sequences that ran too long. Bottom: did early-game load cause late-game failures?")
        render_chart_readout("hypothesis", hyp_a=hyp_a, hyp_b=hyp_b, goalie_a=goalie_a, goalie_b=goalie_b)
    else:
        missing = []
        if hyp_a.empty:
            missing.append(goalie_a)
        if hyp_b.empty:
            missing.append(goalie_b)
        st.info(
            "Not enough data is available for the SLC load hypothesis chart. "
            f"No games survived the selected {game_phase.lower()} date window for: {', '.join(missing)}."
        )

    # 3. Dual Trend Line
    st.subheader("Dual SLC Momentum Trend")
    df_a = filter_hypothesis_window(
        filter_by_game_phase(load_goalie_data(goalie_a_key), game_phase),
        start_d,
        end_d
    )
    df_b = filter_hypothesis_window(
        filter_by_game_phase(load_goalie_data(goalie_b_key), game_phase),
        start_d,
        end_d
    )
    
    if not df_a.empty and not df_b.empty:
        df_a = df_a.copy()
        df_b = df_b.copy()
        df_a['Cumulative_SLC'] = df_a['SLC'].cumsum()
        df_b['Cumulative_SLC'] = df_b['SLC'].cumsum()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df_a['Date'], y=df_a['Cumulative_SLC'], name=goalie_a, line=dict(color='green')))
        fig_trend.add_trace(go.Scatter(x=df_b['Date'], y=df_b['Cumulative_SLC'], name=goalie_b, line=dict(color='orange')))
        st.plotly_chart(fig_trend, width="stretch")
        st.caption("Cumulative SLC over time. A steeper climb means consistently stabilizing performances. A flat or declining line means inconsistency or drain.")
        render_chart_readout("trend", df_a=df_a, df_b=df_b, goalie_a=goalie_a, goalie_b=goalie_b)

        # 4. The Delta Pane ($\Delta$)
        st.divider()
        st.subheader("Hypothesis Delta Summary")
        _, loan_a_season, _ = build_high_interest_loan_data(goalie_a_key, str(df_a.iloc[0]["Game_ID"]))
        _, loan_b_season, _ = build_high_interest_loan_data(goalie_b_key, str(df_b.iloc[0]["Game_ID"]))
        if not loan_a_season.empty:
            loan_a_season = filter_hypothesis_window(filter_by_game_phase(loan_a_season, game_phase), start_d, end_d)
        if not loan_b_season.empty:
            loan_b_season = filter_hypothesis_window(filter_by_game_phase(loan_b_season, game_phase), start_d, end_d)

        pressure_delta = hyp_a["Pressure_Ratio"].mean() - hyp_b["Pressure_Ratio"].mean() if not hyp_a.empty and not hyp_b.empty else 0
        reset_delta = hyp_a["Reset_Ratio"].mean() - hyp_b["Reset_Ratio"].mean() if not hyp_a.empty and not hyp_b.empty else 0
        redline_delta = hyp_a["Red_Line_Shifts"].mean() - hyp_b["Red_Line_Shifts"].mean() if not hyp_a.empty and not hyp_b.empty else 0
        p3_failure_delta = hyp_a["P3_Hardware_Failures"].mean() - hyp_b["P3_Hardware_Failures"].mean() if not hyp_a.empty and not hyp_b.empty else 0
        early_debt_delta = (
            loan_a_season["Early_Debt"].mean() - loan_b_season["Early_Debt"].mean()
            if not loan_a_season.empty and not loan_b_season.empty
            else 0
        )

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Pressure Ratio", f"{pressure_delta:.3f}", delta=f"{pressure_delta:.3f}")
        d2.metric("Reset Ratio", f"{reset_delta:.3f}", delta=f"{reset_delta:.3f}")
        d3.metric("Red-Line Seq.", f"{redline_delta:.2f}", delta=f"{redline_delta:.2f}", delta_color="inverse")
        d4.metric("P3 Failures", f"{p3_failure_delta:.2f}", delta=f"{p3_failure_delta:.2f}", delta_color="inverse")
        d5.metric("Early Debt", f"{early_debt_delta:.2f}", delta=f"{early_debt_delta:.2f}", delta_color="inverse")

        if pressure_delta > 0 and reset_delta > 0 and redline_delta < 0:
            st.success(f"{goalie_a} has the stronger stabilizer profile in this window: better pressure efficiency, more resets, and fewer red-line sequences.")
        elif pressure_delta < 0 and reset_delta < 0 and redline_delta > 0:
            st.warning(f"{goalie_b} has the stronger stabilizer profile in this window. {goalie_a} is trailing on pressure efficiency and reset work while carrying more red-line exposure.")
        else:
            st.info("The delta summary is mixed. Pressure Ratio shows efficiency, Reset Ratio shows service, and Red-Line/Early Debt show the biological tax side.")

        # 5. Load vs Relief Scatter
        st.subheader("Load vs. Relief Positioning")
        scatter_df = pd.concat([
            leaderboard[leaderboard["Goalie"] == goalie_a],
            leaderboard[leaderboard["Goalie"] == goalie_b]
        ])
        reset_min, reset_max = scatter_df["Reset_Ratio"].min(), scatter_df["Reset_Ratio"].max()
        sov_min, sov_max = scatter_df["Sovereignty"].min(), scatter_df["Sovereignty"].max()
        reset_pad = max((reset_max - reset_min) * 0.35, 0.15)
        sov_pad = max((sov_max - sov_min) * 0.35, 1.0)

        fig_scatter = px.scatter(
            scatter_df,
            x="Reset_Ratio",
            y="Sovereignty",
            text="Goalie",
            color="Identity",
            size_max=60,
            range_x=[reset_min - reset_pad, reset_max + reset_pad],
            range_y=[sov_min - sov_pad, sov_max + sov_pad]
        )
        fig_scatter.add_vline(x=0.90, line_dash="dash", line_color="gray")
        fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
        fig_scatter.update_layout(
            height=460,
            xaxis_title="Reset Ratio",
            yaxis_title="Sovereignty",
            margin=dict(l=50, r=40, t=60, b=50)
        )
        st.plotly_chart(fig_scatter, width="stretch")
        render_chart_readout("load_scatter", scatter_df=scatter_df)
        render_download_buttons(
            f"slc_comparison_{goalie_a_key}_vs_{goalie_b_key}_{game_phase.lower().replace(' ', '_')}",
            comparison_summary(goalie_a, goalie_b, game_phase, start_d, end_d, hyp_a, hyp_b, df_a, df_b),
            comparison_table,
            key_prefix=f"compare_{goalie_a_key}_{goalie_b_key}_{game_phase}"
        )

with tab_team:
    st.header(f"Team Redline: {game_phase}")
    teams = get_available_teams()
    if not teams:
        st.warning("No team schedule data is available.")
    else:
        selected_team = st.selectbox("Team", teams)

        team_games = filter_team_games_by_phase(get_team_games(selected_team), game_phase)
        if not team_games:
            st.warning(f"No local raw {game_phase.lower()} games are available for {selected_team}.")
        else:
            game_signature = tuple(
                (game["Game_ID"], game["Date"], game["Matchup"], game["Label"])
                for game in team_games
            )
            trend_df = build_team_season_redline_data(selected_team, game_signature)
            st.subheader("Seasonal 40-Second Wall Trend")
            if trend_df.empty:
                st.info("No seasonal sequence trend data is available for this team.")
            else:
                monthly_trend_df = build_monthly_team_redline_data(trend_df)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Avg 40s+ Shifts", f"{trend_df['Red_Line_Shifts'].mean():.1f}")
                s2.metric("Avg Efficiency", f"{trend_df['Efficiency'].mean():.1f}%")
                s3.metric("Avg Median Sequence", f"{trend_df['Median_Sequence'].mean():.1f}s")
                s4.metric("Avg Total Sequences", f"{trend_df['Total_Sequences'].mean():.1f}")
                if monthly_trend_df.empty:
                    st.info("No monthly trend data is available for this team.")
                else:
                    st.plotly_chart(
                        build_team_redline_trend_chart(monthly_trend_df, selected_team, game_phase),
                        width="stretch"
                    )
                    st.caption("Each point is one month. The y-axis is the average number of defensive sequences per game that crossed the 40-second wall during that month.")
                    st.download_button(
                        "Download monthly trend CSV",
                        data=dataframe_csv(monthly_trend_df),
                        file_name=f"slc_team_redline_monthly_trend_{selected_team}_{game_phase.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"team_trend_{selected_team}_{game_phase}"
                    )

            st.markdown("---")
            st.subheader("Single Game Sequence Distribution")
            c_game, c_phase, c_overlay = st.columns([2, 1, 1])
            with c_game:
                game_labels = [game["Label"] for game in team_games]
                selected_game_label = st.selectbox("Game", game_labels)
            selected_game = next(game for game in team_games if game["Label"] == selected_game_label)
            with c_phase:
                phase = st.radio("Phase", ["All", "3rd Period"])
            with c_overlay:
                include_average = st.toggle("Season Avg", value=True)

            sequence_df, average_df, metrics = get_team_sequence_histogram(
                selected_team,
                selected_game["Game_ID"],
                phase=phase,
                include_average=include_average
            )

            if sequence_df.empty:
                st.info("No defensive pressure sequences were found for that game.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Efficiency Rating", f"{metrics['Efficiency']:.1f}%")
                m2.metric("Median Sequence", f"{metrics['Median']:.1f}s")
                m3.metric("Total Sequences", int(metrics["Total"]))
                render_team_goalie_verdict(selected_game, metrics)
                st.plotly_chart(
                    build_team_wall_histogram(sequence_df, average_df, metrics, selected_team, selected_game_label),
                    width="stretch"
                )
                st.caption("Green bars = sequences that ended quickly (good). Yellow = getting long. Red = over the wall (40+ seconds, defense is gassing). The ghost bars show the season average for comparison.")
                render_chart_readout("team_wall", metrics=metrics, team=selected_team)
                render_download_buttons(
                    f"slc_team_redline_{selected_team}_{selected_game['Game_ID']}_{phase.lower().replace(' ', '_')}",
                    team_redline_summary(selected_team, game_phase, selected_game, phase, metrics, sequence_df),
                    sequence_df,
                    key_prefix=f"team_{selected_team}_{selected_game['Game_ID']}_{phase}"
                )

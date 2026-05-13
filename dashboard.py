import contextlib
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from audit_engine import (
    build_goalie_audit,
    build_league_leaderboard,
    build_selected_game_periods,
    build_team_goalie_decision,
)
from report_exporter import dataframe_csv
from scrubber import get_player_metadata
from update_local_data import update_local_data
from utility import clear_data_caches


MASTER_REPORT_PATH = Path("./data/processedGames/master_report.json")


def get_master_report_version():
    return MASTER_REPORT_PATH.stat().st_mtime_ns if MASTER_REPORT_PATH.exists() else 0


@st.cache_data(show_spinner=False)
def cached_goalie_audit(goalie_key, game_phase, master_version, audit_schema_version=2):
    return build_goalie_audit(goalie_key, game_phase)


@st.cache_data(show_spinner=False)
def cached_league_leaderboard(game_phase, min_gp, master_version, league_schema_version=1):
    return build_league_leaderboard(game_phase, min_gp)


@st.cache_data(show_spinner=False)
def cached_team_goalie_decision(team_abbrev, game_phase, master_version, decision_schema_version=1):
    return build_team_goalie_decision(team_abbrev, game_phase)


def run_dashboard_data_update(game_type):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        summary = update_local_data(season="20252026", game_type=game_type)
    return summary or {}, buffer.getvalue()


def format_date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d")


def render_goalie_header(summary):
    meta = get_player_metadata(summary["ID"])
    with st.container(border=True):
        c1, c2, c3, c4, c5, c6 = st.columns([1.25, 3.25, 1, 1, 1, 1])
        with c1:
            if meta.get("image"):
                st.image(meta["image"], width=180)
        with c2:
            st.markdown(f"### {summary['Goalie']}")
            st.caption(f"{meta['team']} | {meta['height']} | {meta['weight']} | {meta['age']}")
            st.caption(f"{summary['SLC_Profile']} | SLC converts pressure survival, pressure relief, and battery fit into one goalie efficiency grade.")
        with c3:
            st.metric("SLC Grade", f"{summary['SLC_Grade']:.0f}")
        with c4:
            st.metric("Survival", f"{summary['Pressure_Survival_Grade']:.0f}")
        with c5:
            st.metric("Relief", f"{summary['Pressure_Relief_Grade']:.0f}")
        with c6:
            st.metric("Battery Fit", f"{summary['Battery_Fit_Grade']:.0f}")


def render_downloads(goalie_key, game_phase, game_log, evidence):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download game log CSV",
            data=dataframe_csv(game_log),
            file_name=f"slc_audit_game_log_{goalie_key}_{game_phase.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            disabled=game_log.empty,
        )
    with c2:
        st.download_button(
            "Download evidence CSV",
            data=dataframe_csv(evidence),
            file_name=f"slc_audit_evidence_{goalie_key}_{game_phase.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            disabled=evidence.empty,
        )


def build_efficiency_workload_chart(game_log, summary, verdict):
    chart_df = game_log.copy()
    chart_df["Date_Label"] = chart_df["Date"].apply(format_date)
    season_survival = float(summary["Pressure_Survival_per_Game"])
    season_relief = float(summary["Relief_Efficiency_per_Game"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["Relief_Efficiency"],
            y=chart_df["Pressure_Survival"],
            mode="markers",
            marker=dict(
                size=10,
                color=chart_df["SLC_per_Sequence"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="SLC / Seq"),
                line=dict(width=1, color="rgba(40,40,40,0.45)")
            ),
            customdata=chart_df[[
                "Date_Label", "Matchup", "SLC", "Net_Load_per_Sequence",
                "Relief_Capture_Rate", "SLC_per_Sequence", "Sequences", "Shots"
            ]],
            hovertemplate=(
                "%{customdata[0]} %{customdata[1]}<br>"
                "Relief Efficiency: %{x:.3f}<br>"
                "Pressure Survival: %{y:.3f}<br>"
                "Game SLC: %{customdata[2]:.3f}<br>"
                "Net Load / Sequence: %{customdata[3]:.3f}<br>"
                "Relief Capture Rate: %{customdata[4]:.3f}<br>"
                "SLC / Sequence: %{customdata[5]:.3f}<br>"
                "Sequences: %{customdata[6]}<br>"
                "Shots: %{customdata[7]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=season_survival,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Season Survival: {season_survival:.3f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=season_relief,
        line_dash="dot",
        line_color="#1f77b4",
        annotation_text=f"Season Relief: {season_relief:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Game Pressure Survival vs Pressure Relief",
        height=430,
        xaxis_title="Pressure Relief",
        yaxis_title="Pressure Survival",
        margin=dict(l=50, r=40, t=70, b=55),
    )
    return fig


st.set_page_config(layout="wide", page_title="SLC Dashboard")
st.title("SLC Dashboard")

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
            clear_data_caches()
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

seed_audit = cached_goalie_audit("", game_phase, get_master_report_version())
leaderboard = seed_audit["leaderboard"]

if leaderboard.empty:
    st.warning(f"No goalie data is available for {game_phase.lower()} games.")
    st.stop()

tab_league, tab_audit, tab_team_decision = st.tabs([
    "League Leaderboard",
    "Individual Audit",
    "Team Goalie Decision",
])

with tab_league:
    st.header(f"League Leaderboard: {game_phase}")
    default_min_gp = 20 if game_phase == "Regular Season" else 1
    min_gp = st.number_input("Minimum GP", min_value=1, max_value=82, value=default_min_gp, step=1)

    league_view = cached_league_leaderboard(game_phase, min_gp, get_master_report_version())
    qualified = league_view["qualified"]
    league_context = league_view.get("context", {})

    if qualified.empty:
        st.info(f"No qualified goalies are available for {game_phase.lower()} games.")
    else:
        st.caption(
            "This table is sorted by SLC Grade: pressure survival, pressure relief, and battery fit. "
            "Counting stats are intentionally kept off this view; use the audit page for the ingredients."
        )
        lm1, lm2, lm3, lm4 = st.columns(4)
        lm1.metric("Qualified Goalies", int(league_context.get("goalies", len(qualified))))
        lm2.metric("Median SLC Grade", f"{league_context.get('slc_grade_median', 0):.0f}")
        lm3.metric("Median Survival Grade", f"{league_context.get('pressure_survival_grade_median', 0):.0f}")
        lm4.metric("Median Relief Grade", f"{league_context.get('pressure_relief_grade_median', 0):.0f}")

        league_columns = [
            "Rank", "Goalie", "Team", "Team_System_Profile", "GP",
            "SLC_Grade", "SLC_Profile", "Pressure_Survival_Grade",
            "Pressure_Relief_Grade", "Battery_Fit_Grade", "SLC_per_Sequence",
            "Sequences_per_Game",
        ]
        display_league = qualified[league_columns].rename(columns={
            "SLC_Grade": "SLC Grade",
            "SLC_Profile": "SLC Profile",
            "Pressure_Survival_Grade": "Pressure Survival",
            "Pressure_Relief_Grade": "Pressure Relief",
            "Battery_Fit_Grade": "Battery Fit",
            "Team_System_Profile": "Team System",
            "SLC_per_Sequence": "SLC / Sequence",
            "Sequences_per_Game": "Sequences / Game",
        })
        st.dataframe(display_league, hide_index=True)
        st.download_button(
            "Download league leaderboard CSV",
            data=dataframe_csv(display_league),
            file_name=f"slc_league_leaderboard_{game_phase.lower().replace(' ', '_')}_{min_gp}_gp.csv",
            mime="text/csv",
        )

with tab_audit:
    goalie_options = leaderboard.sort_values("Goalie")["Goalie"].tolist()
    selected_goalie = st.selectbox("Select goalie", goalie_options)
    goalie_key = selected_goalie.lower().replace(" ", "_")
    audit = cached_goalie_audit(goalie_key, game_phase, get_master_report_version())

    summary = audit["summary"]
    verdict = audit["verdict"]
    context = audit.get("context", pd.DataFrame())
    translation = audit.get("translation", pd.DataFrame())
    team_profile = audit.get("team_profile", pd.DataFrame())
    team_fit_read = audit.get("team_fit_read", "")
    evidence = audit["evidence"]
    game_log = audit["game_log"]

    if not summary:
        st.warning(f"No game history found for {selected_goalie}.")
        st.stop()

    render_goalie_header(summary)

    st.subheader("Audit Statement")
    if summary["SLC_Grade"] >= 70:
        st.success(verdict["text"])
    elif summary["SLC_Grade"] >= 50:
        st.info(verdict["text"])
    else:
        st.warning(verdict["text"])
    st.caption(
        f"League context uses {verdict['context_goalies']} qualified goalies "
        f"({verdict['context_min_gp']}+ GP). Median is the snapshot benchmark; mean is kept as the elite/top-end pull."
    )

    if not team_profile.empty:
        st.subheader("Team System Profile")
        st.info(team_fit_read)
        profile_row = team_profile.iloc[0]
        tp1, tp2, tp3, tp4 = st.columns(4)
        tp1.metric("System", profile_row["System Profile"])
        tp2.metric("Avg 40s Wall Breaches", f"{profile_row['Avg 40s Wall Breaches']:.3f}")
        tp3.metric("40s Wall Efficiency", f"{profile_row['40s Wall Efficiency']:.1f}%")
        tp4.metric("Avg Sequence Duration", f"{profile_row['Avg Sequence Duration']:.1f}s")
        st.dataframe(team_profile, hide_index=True)
        st.caption(
            "This section profiles the team environment, not the goalie. "
            "Use it as context for what kind of goalie efficiency the system asks for."
        )

    st.subheader("SLC Grade Breakdown")
    st.caption(
        "Pressure Survival asks whether the goalie handled danger. "
        "Pressure Relief asks whether they gave the skaters relief. "
        "Battery Fit asks whether that blend matches this team environment."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SLC Grade", f"{summary['SLC_Grade']:.0f}")
    m2.metric("Pressure Survival", f"{summary['Pressure_Survival_Grade']:.0f}")
    m3.metric("Pressure Relief", f"{summary['Pressure_Relief_Grade']:.0f}")
    m4.metric("Battery Fit", f"{summary['Battery_Fit_Grade']:.0f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SLC / Sequence", f"{summary['SLC_per_Sequence']:.3f}", delta=f"{verdict['efficiency_delta']:+.3f} vs median")
    c2.metric("Relief Capture", f"{summary['Relief_Capture_Rate']:.3f}", delta=f"{verdict['relief_capture_delta']:+.3f} vs median")
    c3.metric("Net Load / Sequence", f"{summary['Net_Load_per_Sequence']:.3f}", delta=f"{verdict['net_load_delta']:+.3f} vs median")
    c4.metric("Games", int(summary["GP"]))

    if not translation.empty:
        with st.expander("Plain-English Metric Translation", expanded=False):
            st.dataframe(translation, hide_index=True)

    with st.expander("Counting Stat Ingredients", expanded=False):
        st.dataframe(evidence, hide_index=True)

    if not context.empty:
        with st.expander("League Context", expanded=False):
            st.dataframe(context, hide_index=True)

    st.subheader("Game Pressure Survival vs Pressure Relief")
    if game_log.empty:
        st.info("No game-level efficiency data is available for this goalie and game set.")
    else:
        st.plotly_chart(build_efficiency_workload_chart(game_log, summary, verdict))
        st.caption(
            "Each dot is one game. Higher means better pressure survival; farther right means more pressure relief. "
            "The best efficiency profile lives toward the upper-right."
        )

    st.subheader("Games")
    game_columns = [
        "Date", "Matchup", "SLC", "SLC_per_Sequence",
        "Pressure_Survival", "Relief_Efficiency", "Relief_Capture_Rate",
        "Net_Load_per_Sequence", "Sequences", "Shots", "Sovereignty",
    ]
    detail_game_columns = [
        "Date", "Matchup", "Service_per_Sequence", "Tax_per_Sequence",
        "Service", "Tax", "NPW", "NPW_SH",
        "UA", "UA_SH", "RP", "RP_SH", "iGvA", "iGvA_SH",
    ]
    display_log = game_log.copy()
    if not display_log.empty:
        display_log["Date"] = display_log["Date"].apply(format_date)
        st.dataframe(
            display_log.sort_values("Date", ascending=False)[game_columns],
        )
        with st.expander("Game-Level Counting Details", expanded=False):
            st.dataframe(
                display_log.sort_values("Date", ascending=False)[detail_game_columns],
            )
    else:
        st.info("No games are available for this goalie and game set.")

    st.subheader("Selected Game Period Detail")
    if not game_log.empty:
        game_options = game_log.sort_values("Date", ascending=False).copy()
        game_options["Label"] = game_options["Date"].apply(format_date) + " " + game_options["Matchup"].astype(str)
        selected_game_label = st.selectbox("Select game", game_options["Label"].tolist())
        selected_game = game_options[game_options["Label"] == selected_game_label].iloc[0]
        selected_periods = build_selected_game_periods(goalie_key, str(selected_game["Game_ID"]))

        if selected_periods.empty:
            st.info("No period detail is available for that game.")
        else:
            st.dataframe(selected_periods, hide_index=True)

    render_downloads(goalie_key, game_phase, game_log, evidence)

with tab_team_decision:
    st.header(f"Team Goalie Decision: {game_phase}")
    available_teams = sorted(team for team in leaderboard["Team"].dropna().unique() if team)
    if not available_teams:
        st.info("No team-linked goalie data is available for this game set.")
    else:
        selected_team = st.selectbox("Select team", available_teams)
        decision = cached_team_goalie_decision(selected_team, game_phase, get_master_report_version())
        decision_table = decision["summary"]
        decision_games = decision["games"]
        decision_context = decision.get("context", {})

        if decision_table.empty:
            st.info(f"No goalie decision data is available for {selected_team}.")
        else:
            st.caption(
                "This page asks whether a goalie's efficiency survives starter-like conditions. "
                "High-workload starts are games at or above the team's median goalie sequences, shots, or 40-second wall burden."
            )
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Current Workload Leader", decision_context.get("starter", ""))
            dc2.metric("Median Goalie Sequences", f"{decision_context.get('team_median_goalie_sequences', 0):.1f}")
            dc3.metric("Median Shots", f"{decision_context.get('team_median_shots', 0):.1f}")
            dc4.metric("Median Red-Line Shifts", f"{decision_context.get('team_median_red_line', 0):.1f}")

            display_decision = decision_table.rename(columns={
                "SLC_Grade": "SLC Grade",
                "SLC_Profile": "SLC Profile",
                "Pressure_Survival": "Pressure Survival",
                "Pressure_Relief": "Pressure Relief",
                "Battery_Fit": "Battery Fit",
                "High_Workload_Starts": "High-Workload Starts",
                "High_Workload_SLC_Grade": "High-Workload Grade",
                "Above_Median_Start_Rate": "Above-Median Start Rate",
                "Bad_Start_Rate": "Bad-Start Rate",
                "Stability_Grade": "Stability",
                "Role_Confidence": "Role Confidence",
                "Decision_Note": "Decision Note",
            })
            decision_columns = [
                "Goalie", "Recommendation", "Confidence", "GP", "SLC Grade",
                "SLC Profile", "Pressure Survival", "Pressure Relief", "Battery Fit",
                "High-Workload Starts", "High-Workload Grade", "Above-Median Start Rate",
                "Bad-Start Rate", "Stability", "Role Confidence", "Decision Note",
            ]
            st.dataframe(display_decision[decision_columns], hide_index=True)

            with st.expander("Starter-Like Game Evidence", expanded=False):
                game_detail = decision_games.copy()
                if not game_detail.empty:
                    game_detail["Date"] = game_detail["Date"].apply(format_date)
                    game_columns = [
                        "Date", "Goalie", "Matchup", "High_Workload_Start",
                        "Game_SLC_Grade", "Game_Survival_Grade", "Game_Relief_Grade",
                        "SLC", "Sequences", "Shots", "Team_Red_Line_Shifts",
                        "Pressure_Survival", "Relief_Efficiency", "Relief_Capture_Rate",
                    ]
                    st.dataframe(game_detail[game_columns], hide_index=True)

            st.download_button(
                "Download team goalie decision CSV",
                data=dataframe_csv(display_decision),
                file_name=f"slc_team_goalie_decision_{selected_team}_{game_phase.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )

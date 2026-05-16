import contextlib
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

from audit_engine import (
    build_goalie_audit,
    build_league_leaderboard,
    build_selected_game_periods,
    build_team_goalie_decision,
)
from report_exporter import dataframe_csv
from scrubber import get_player_metadata
from update_local_data import (
    build_processed_season_report,
    fetch_raw_season_data,
    has_pending_season_processing,
    load_season_pipeline_state,
    update_local_data,
)
from slc_prediction_engine import build_late_xga_collapse_proof
from utility import (
    available_master_report_seasons,
    clear_data_caches,
    master_report_path,
    set_active_data_season,
)
from utility import _get_goalie_team_abbrev, _late_xga, _load_raw_game, load_master_reports


def get_master_report_version(season):
    report_path = Path(master_report_path(season))
    return report_path.stat().st_mtime_ns if report_path.exists() else 0


@st.cache_data(show_spinner=False)
def cached_goalie_audit(goalie_key, game_phase, season, master_version, audit_schema_version=2):
    set_active_data_season(season)
    return build_goalie_audit(goalie_key, game_phase)


@st.cache_data(show_spinner=False)
def cached_league_leaderboard(game_phase, min_gp, season, master_version, league_schema_version=1):
    set_active_data_season(season)
    return build_league_leaderboard(game_phase, min_gp)


@st.cache_data(show_spinner=False)
def cached_team_goalie_decision(team_abbrev, game_phase, season, master_version, decision_schema_version=1):
    set_active_data_season(season)
    return build_team_goalie_decision(team_abbrev, game_phase)


@st.cache_data(show_spinner=False)
def cached_late_xga_collapse_proof(season, master_version, proof_schema_version=5):
    set_active_data_season(season)
    return build_late_xga_collapse_proof()


def format_collapse_proof_table(proof_risk):
    target_labels = {
        "Future_Team_Relative_Collapse_H1": ("Above team recent baseline", "Next game"),
        "Future_Team_Relative_Collapse_H3": ("Above team recent baseline", "Next 3 games"),
        "Future_Team_Relative_Collapse_H5": ("Above team recent baseline", "Next 5 games"),
    }

    rows = []
    for _, row in proof_risk.iterrows():
        collapse_type, window = target_labels.get(row["Target"], (row["Target"], ""))
        gap = float(row["Low_Minus_High"]) * 100
        rows.append({
            "Collapse Type": collapse_type,
            "Schedule": row["Schedule_Context"],
            "Window": window,
            "Low Recent SLC": f"{float(row['Low_SLC_Risk']) * 100:.1f}%",
            "High Recent SLC": f"{float(row['High_SLC_Risk']) * 100:.1f}%",
            "Low-SLC Added Risk": f"{gap:+.1f} pts",
            "Read": "Supports SLC heatsink signal" if gap > 0 else "No low-SLC risk edge",
            "Sample": f"{int(row['N']):,}",
        })

    order = ["Stressed", "Normal"]
    display = pd.DataFrame(rows)
    if display.empty:
        return display
    display["Schedule_Order"] = display["Schedule"].apply(lambda value: order.index(value) if value in order else 99)
    display["Window_Order"] = display["Window"].map({
        "Next game": 1,
        "Next 3 games": 3,
        "Next 5 games": 5,
    }).fillna(99)
    display["Collapse_Order"] = display["Collapse Type"].map({
        "Above team recent baseline": 1,
    }).fillna(99)
    display = display.sort_values(["Schedule_Order", "Collapse_Order", "Window_Order"])
    return display.drop(columns=["Schedule_Order", "Collapse_Order", "Window_Order"])


def format_model_lift_table(model_lift):
    if model_lift.empty:
        return model_lift

    rows = []
    for _, row in model_lift.iterrows():
        separation = float(row["Risk_Separation"]) * 100
        added_lift = row["Added_Lift_vs_Previous"]
        lift_text = "baseline" if pd.isna(added_lift) else f"{float(added_lift) * 100:+.1f} pts"
        lift_vs_load = row.get("Lift_vs_Load_Baseline")
        lift_vs_load_text = "baseline" if pd.isna(lift_vs_load) else f"{float(lift_vs_load) * 100:+.1f} pts"

        if row["Model"] == "Schedule + Load + SLC":
            read = "SLC improved separation" if pd.notna(added_lift) and added_lift > 0 else "No SLC lift"
        elif row["Model"] == "Schedule + Load + Relief Rate":
            read = "Relief improved separation" if pd.notna(added_lift) and added_lift > 0 else "No relief lift"
        elif row["Model"] == "Schedule + Load + SLC + Relief Rate":
            read = "Relief adds beyond SLC" if pd.notna(added_lift) and added_lift > 0 else "No added relief beyond SLC"
        elif row["Model"] == "Schedule + Load + SLC + Relief + Cost":
            read = "Cost adds beyond relief" if pd.notna(added_lift) and added_lift > 0 else "No added cost lift"
        elif row["Model"] == "Schedule + Recent Load":
            read = "Load improved baseline" if pd.notna(added_lift) and added_lift > 0 else "No load lift"
        else:
            read = "Starting baseline"

        rows.append({
            "Target": row["Target"],
            "Schedule": row["Schedule_Context"],
            "Model": row["Model"],
            "Low Predicted Risk": f"{float(row['Low_Predicted_Risk_Rate']) * 100:.1f}%",
            "High Predicted Risk": f"{float(row['High_Predicted_Risk_Rate']) * 100:.1f}%",
            "Risk Separation": f"{separation:+.1f} pts",
            "Added Lift": lift_text,
            "Lift vs Load": lift_vs_load_text,
            "Read": read,
            "Sample": f"{int(row['N']):,}",
        })

    schedule_order = {"Stressed": 1, "Normal": 2, "All": 3}
    model_order = {
        "Schedule Only": 1,
        "Schedule + Recent Load": 2,
        "Schedule + Load + SLC": 3,
        "Schedule + Load + Relief Rate": 4,
        "Schedule + Load + SLC + Relief Rate": 5,
        "Schedule + Load + SLC + Relief + Cost": 6,
    }
    display = pd.DataFrame(rows)
    display["Schedule_Order"] = display["Schedule"].map(schedule_order).fillna(99)
    display["Model_Order"] = display["Model"].map(model_order).fillna(99)
    display = display.sort_values(["Target", "Schedule_Order", "Model_Order"])
    return display.drop(columns=["Schedule_Order", "Model_Order"])


def format_profile_model_lift_table(model_lift):
    if model_lift.empty:
        return model_lift

    rows = []
    for _, row in model_lift.iterrows():
        lift_vs_load = row.get("Lift_vs_Load_Baseline")
        rows.append({
            "Profile": row["Volatility_Profile"],
            "Model": row["Model"],
            "Low Predicted Risk": f"{float(row['Low_Predicted_Risk_Rate']) * 100:.1f}%",
            "High Predicted Risk": f"{float(row['High_Predicted_Risk_Rate']) * 100:.1f}%",
            "Risk Separation": f"{float(row['Risk_Separation']) * 100:+.1f} pts",
            "Lift vs Load": "baseline" if pd.isna(lift_vs_load) else f"{float(lift_vs_load) * 100:+.1f} pts",
            "Sample": f"{int(row['N']):,}",
        })

    profile_order = {
        "Stable": 1,
        "Brittle": 2,
        "Controlled Chaos": 3,
        "Volatile": 4,
    }
    model_order = {
        "Schedule + Recent Load": 1,
        "Schedule + Recent Load + SLC": 2,
        "Schedule + Recent Load + Sovereignty": 3,
        "Schedule + Recent Load + Relief Rate": 4,
        "Schedule + Recent Load + Pressure": 5,
    }
    display = pd.DataFrame(rows)
    display["Profile_Order"] = display["Profile"].map(profile_order).fillna(99)
    display["Model_Order"] = display["Model"].map(model_order).fillna(99)
    display = display.sort_values(["Profile_Order", "Model_Order"])
    return display.drop(columns=["Profile_Order", "Model_Order"])


def format_profile_weight_optimizer_table(optimizer):
    if optimizer.empty:
        return optimizer

    rows = []
    for _, row in optimizer.iterrows():
        rows.append({
            "Profile": row["Volatility_Profile"],
            "Sovereignty Weight": f"{float(row['Sovereignty_Weight']):.1f}",
            "Relief Weight": f"{float(row['Relief_Weight']):.1f}",
            "Pressure Weight": f"{float(row['Pressure_Weight']):.1f}",
            "Base Separation": f"{float(row['Base_Separation']) * 100:+.1f} pts",
            "Optimized SLC Separation": f"{float(row['Optimized_SLC_Separation']) * 100:+.1f} pts",
            "Lift vs Load": f"{float(row['Lift_vs_Load_Baseline']) * 100:+.1f} pts",
            "Sample": f"{int(row['N']):,}",
        })

    profile_order = {
        "Stable": 1,
        "Brittle": 2,
        "Controlled Chaos": 3,
        "Volatile": 4,
    }
    display = pd.DataFrame(rows)
    display["Profile_Order"] = display["Profile"].map(profile_order).fillna(99)
    display = display.sort_values("Profile_Order")
    return display.drop(columns=["Profile_Order"])


def format_profile_weight_validation_table(validation):
    if validation.empty:
        return validation

    rows = []
    for _, row in validation.iterrows():
        rows.append({
            "Profile": row["Volatility_Profile"],
            "Sovereignty Weight": f"{float(row['Sovereignty_Weight']):.1f}",
            "Relief Weight": f"{float(row['Relief_Weight']):.1f}",
            "Pressure Weight": f"{float(row['Pressure_Weight']):.1f}",
            "Tuning Lift": f"{float(row['Tuning_Lift']) * 100:+.1f} pts",
            "Holdout Base": f"{float(row['Holdout_Base_Separation']) * 100:+.1f} pts",
            "Holdout SLC": f"{float(row['Holdout_SLC_Separation']) * 100:+.1f} pts",
            "Holdout Lift": f"{float(row['Holdout_Lift']) * 100:+.1f} pts",
            "Sample": f"{int(row['N']):,}",
        })

    profile_order = {
        "Stable": 1,
        "Brittle": 2,
        "Controlled Chaos": 3,
        "Volatile": 4,
    }
    display = pd.DataFrame(rows)
    display["Profile_Order"] = display["Profile"].map(profile_order).fillna(99)
    display = display.sort_values("Profile_Order")
    return display.drop(columns=["Profile_Order"])


def format_same_game_heatsink_table(same_game_risk):
    if same_game_risk.empty:
        return same_game_risk

    rows = []
    for _, row in same_game_risk.iterrows():
        gap = float(row["Low_Minus_High"]) * 100
        rows.append({
            "Game Context": row["Context"],
            "Early Signal": row["Signal"],
            "Low Signal P3 Collapse": f"{float(row['Low_Risk']) * 100:.1f}%",
            "High Signal P3 Collapse": f"{float(row['High_Risk']) * 100:.1f}%",
            "Low-Signal Added Risk": f"{gap:+.1f} pts",
            "Read": "Supports same-game heatsink signal" if gap > 0 else "No same-game edge",
            "Sample": f"{int(row['N']):,}",
        })

    display = pd.DataFrame(rows)
    context_order = {"High early pressure": 1, "Lower early pressure": 2}
    signal_order = {"Early SLC": 1, "Early relief rate": 2}
    display["Context_Order"] = display["Game Context"].map(context_order).fillna(99)
    display["Signal_Order"] = display["Early Signal"].map(signal_order).fillna(99)
    display = display.sort_values(["Context_Order", "Signal_Order"])
    return display.drop(columns=["Context_Order", "Signal_Order"])


def format_same_game_lift_table(same_game_lift):
    if same_game_lift.empty:
        return same_game_lift

    rows = []
    for _, row in same_game_lift.iterrows():
        added_lift = row["Added_Lift_vs_Previous"]
        rows.append({
            "Model": row["Model"],
            "Low Predicted Risk": f"{float(row['Low_Predicted_Risk_Rate']) * 100:.1f}%",
            "High Predicted Risk": f"{float(row['High_Predicted_Risk_Rate']) * 100:.1f}%",
            "Risk Separation": f"{float(row['Risk_Separation']) * 100:+.1f} pts",
            "Added Lift": "baseline" if pd.isna(added_lift) else f"{float(added_lift) * 100:+.1f} pts",
            "Added Signal Direction": row.get("Added_Signal_Direction", ""),
            "Read": "Adds signal" if pd.notna(added_lift) and added_lift > 0 else ("Baseline" if pd.isna(added_lift) else "No added signal"),
            "Sample": f"{int(row['N']):,}",
        })

    return pd.DataFrame(rows)


def run_dashboard_data_update(game_type):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        summary = update_local_data(season="20252026", game_type=game_type)
    return summary or {}, buffer.getvalue()


def run_fetch_raw_season_data(season, game_type, progress_callback):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        summary = fetch_raw_season_data(
            season=season,
            game_type=game_type,
            progress_callback=progress_callback,
        )
    return summary or {}, buffer.getvalue()


def run_build_processed_season_report(season, game_type, progress_callback):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        summary = build_processed_season_report(
            season=season,
            game_type=game_type,
            progress_callback=progress_callback,
        )
    return summary or {}, buffer.getvalue()


def format_date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d")


def render_goalie_header(summary):
    meta = get_player_metadata(summary["ID"])
    with st.container(border=True):
        c1, c2, c3, c4, c5, c6 = st.columns([1.1, 3.1, 1, 1, 1, 1])
        with c1:
            if meta.get("image"):
                st.markdown(f"""
                    <div style="width:100%; height:250px; overflow:hidden; border-radius:8px;">
                        <img src="{meta["image"]}" style="width:100%; height:90%; object-fit:cover; display:block;" alt="{summary['Goalie']}">
                    </div>
                """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                            <div class="player-card">
                                <h3>{summary['Goalie']}</h3>
                                <div class="meta" style="padding-bottom: 20px; padding-top: 10px">{meta['team']} | {meta['height']} | {meta['weight']} | {meta['age']}</div>
                                <div class="note" style="padding-bottom: 30px; padding-top: 10px">SLC Profile : {summary['SLC_Profile']}</div>
                            </div>
                        """, unsafe_allow_html=True)
            st.caption(
                "System Load Coefficient is the overall goalie grade for how well they perform inside their team system."
                "It balances pressure survival, pressure relief, and battery fit, while minimizing the impact of volume-driven inflation.")
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
                color=chart_df["Pressure_Efficiency_per_Sequence"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Pressure Efficiency"),
                line=dict(width=1, color="rgba(40,40,40,0.45)")
            ),
            customdata=chart_df[[
                "Date_Label", "Matchup", "SLC", "Net_Load_per_Sequence",
                "Relief_Capture_Rate", "Pressure_Efficiency_per_Sequence", "Sequences", "Shots"
            ]],
            hovertemplate=(
                "%{customdata[0]} %{customdata[1]}<br>"
                "Relief Efficiency: %{x:.3f}<br>"
                "Pressure Survival: %{y:.3f}<br>"
                "Net Load / Sequence: %{customdata[3]:.3f}<br>"
                "Relief Capture Rate: %{customdata[4]:.3f}<br>"
                "Pressure Efficiency per Sequence: %{customdata[5]:.3f}<br>"
                "Total Pressure Efficiency: %{customdata[2]:.3f}<br>"
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


@st.cache_data(show_spinner=False)
def cached_team_late_xga_volatility(game_phase, season, master_version, sandbox_schema_version=1):
    set_active_data_season(season)
    return build_team_late_xga_volatility(game_phase)


def _dashboard_game_type(game_id):
    game_id = str(game_id or "")
    if len(game_id) >= 6 and game_id[4:6].isdigit():
        return int(game_id[4:6])
    return None


def _dashboard_matches_phase(date_value, game_phase, game_id):
    game_type = _dashboard_game_type(game_id)
    if game_phase == "Playoffs":
        return game_type == 3
    return game_type == 2


def build_team_late_xga_volatility(game_phase):
    rows = []
    for games in load_master_reports().values():
        for game_id, report in games.items():
            raw_game = _load_raw_game(game_id)
            date_value = report.get("gameDate") or raw_game.get("gameDate")
            if not _dashboard_matches_phase(date_value, game_phase, game_id):
                continue
            team = _get_goalie_team_abbrev(raw_game, report.get("goalie_id"))
            if not team:
                continue
            rows.append({
                "Team": team,
                "Game_ID": str(game_id),
                "Date": pd.to_datetime(date_value, errors="coerce"),
                "Late_xGA": _late_xga(report),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    team_games = (
        df.dropna(subset=["Date"])
        .groupby(["Team", "Game_ID", "Date"], as_index=False)
        .agg({"Late_xGA": "sum"})
        .sort_values(["Team", "Date", "Game_ID"])
    )
    if team_games.empty:
        return team_games

    frames = []
    for _, group in team_games.groupby("Team"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)
        group["Rolling_Avg"] = group["Late_xGA"].shift(1).rolling(5, min_periods=3).mean()
        group["Rolling_Fluctuation"] = group["Late_xGA"].shift(1).rolling(5, min_periods=3).std()
        season_fluctuation = float(group["Late_xGA"].std()) if len(group) > 1 else 0.0
        group["Typical_Fluctuation"] = group["Rolling_Fluctuation"].fillna(season_fluctuation).fillna(0.0)
        group["Spike_Threshold"] = group["Rolling_Avg"] + group["Typical_Fluctuation"]
        group["Spike"] = np.where(
            group["Rolling_Avg"].notna(),
            group["Late_xGA"] > group["Spike_Threshold"],
            False,
        )
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def render_late_xga_volatility_sandbox(game_phase, season):
    volatility = cached_team_late_xga_volatility(game_phase, season, get_master_report_version(season))
    st.subheader("Team Late-xGA Volatility Sandbox")
    if volatility.empty:
        st.info(f"No team late-xGA data is available for {game_phase.lower()} games.")
        return

    teams = sorted(volatility["Team"].dropna().unique())
    selected_team = st.selectbox("Team", teams)
    team_df = volatility[volatility["Team"] == selected_team].sort_values("Date").copy()
    usable = team_df[team_df["Rolling_Avg"].notna()].copy()

    avg_late_xga = float(team_df["Late_xGA"].mean()) if not team_df.empty else 0.0
    median_late_xga = float(team_df["Late_xGA"].median()) if not team_df.empty else 0.0
    typical_fluctuation = float(team_df["Late_xGA"].std()) if len(team_df) > 1 else 0.0
    spike_rate = float(usable["Spike"].mean()) if not usable.empty else 0.0
    latest_threshold = float(team_df["Spike_Threshold"].dropna().iloc[-1]) if team_df["Spike_Threshold"].notna().any() else 0.0

    vm1, vm2, vm3, vm4, vm5 = st.columns(5)
    vm1.metric("Games", f"{len(team_df):,}")
    vm2.metric("Avg Late xGA", f"{avg_late_xga:.3f}")
    vm3.metric("Median Late xGA", f"{median_late_xga:.3f}")
    vm4.metric("Typical Fluctuation", f"{typical_fluctuation:.3f}")
    vm5.metric("Spike Rate", f"{spike_rate * 100:.1f}%", delta=f"Latest threshold {latest_threshold:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=team_df["Date"],
        y=team_df["Late_xGA"],
        mode="lines+markers",
        name="Late xGA",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=team_df["Date"],
        y=team_df["Rolling_Avg"],
        mode="lines",
        name="Rolling baseline",
        line=dict(color="#2ca02c", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=team_df["Date"],
        y=team_df["Spike_Threshold"],
        mode="lines",
        name="Spike threshold",
        line=dict(color="#d62728", width=2, dash="dot"),
    ))
    spike_df = team_df[team_df["Spike"]]
    if not spike_df.empty:
        fig.add_trace(go.Scatter(
            x=spike_df["Date"],
            y=spike_df["Late_xGA"],
            mode="markers",
            name="Outside normal range",
            marker=dict(color="#d62728", size=11, symbol="x"),
        ))
    fig.update_layout(
        height=430,
        xaxis_title="Game date",
        yaxis_title="Late xGA",
        margin=dict(l=50, r=40, t=30, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Game Values", expanded=False):
        display = team_df[["Date", "Late_xGA", "Rolling_Avg", "Typical_Fluctuation", "Spike_Threshold", "Spike"]].copy()
        display["Date"] = display["Date"].apply(format_date)
        st.dataframe(display, use_container_width=True, hide_index=True)

    leaderboard_rows = []
    for team, group in volatility.groupby("Team"):
        group = group.sort_values("Date")
        usable_group = group[group["Rolling_Avg"].notna()]
        spike_rate_value = float(usable_group["Spike"].mean()) if not usable_group.empty else 0.0
        leaderboard_rows.append({
            "Team": team,
            "Games": int(len(group)),
            "Avg Late xGA": round(float(group["Late_xGA"].mean()), 3),
            "Median Late xGA": round(float(group["Late_xGA"].median()), 3),
            "Typical Fluctuation": round(float(group["Late_xGA"].std()) if len(group) > 1 else 0.0, 3),
            "Spike Rate": f"{spike_rate_value * 100:.1f}%",
            "Spike_Rate_Order": spike_rate_value,
        })

    st.subheader("League Overhead")
    league_overhead = pd.DataFrame(leaderboard_rows).sort_values("Spike_Rate_Order", ascending=False)
    st.dataframe(
        league_overhead.drop(columns=["Spike_Rate_Order"]),
        use_container_width=True,
        hide_index=True,
    )


def render_predictive_label_comparison(season):
    proof = cached_late_xga_collapse_proof(season, get_master_report_version(season))
    model_lift_by_profile = proof.get("model_lift_by_profile", pd.DataFrame())
    profile_weight_optimizer = proof.get("profile_weight_optimizer", pd.DataFrame())
    profile_weight_validation = proof.get("profile_weight_validation", pd.DataFrame())

    st.subheader("Predictive Table by Team Volatility Profile")
    st.caption(
        "This uses the team recent baseline target and compares each goalie signal inside the team's volatility profile."
    )

    if model_lift_by_profile.empty:
        st.info("No predictive table is available.")
    else:
        st.dataframe(
            format_profile_model_lift_table(model_lift_by_profile),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Profile Weight Optimizer**")
    if profile_weight_optimizer.empty:
        st.info("No optimizer table is available.")
    else:
        st.dataframe(
            format_profile_weight_optimizer_table(profile_weight_optimizer),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Validated Profile Weight Optimizer**")
    if profile_weight_validation.empty:
        st.info("No validated optimizer table is available.")
    else:
        st.dataframe(
            format_profile_weight_validation_table(profile_weight_validation),
            use_container_width=True,
            hide_index=True,
        )


def render_season_data_tool(game_phase):
    game_type = {"Regular Season": 2, "Playoffs": 3}[game_phase]
    state = load_season_pipeline_state()
    pending = has_pending_season_processing()

    with st.container(border=True):
        tool_title, tool_status = st.columns([1.2, 2.8])
        with tool_title:
            st.markdown("**Season Data Tool**")
        with tool_status:
            if pending:
                st.caption(
                    f"Pending processing: {state.get('season')} raw data is fetched. "
                    "Build the processed report before fetching another season."
                )
            elif state.get("processed_complete"):
                st.caption(f"Last processed season: {state.get('season')}")
            else:
                st.caption("Fetch a full season into raw games, then build that season's master report.")

        if st.session_state.get("season_tool_notice"):
            st.success(st.session_state.pop("season_tool_notice"))
        if st.session_state.get("season_tool_error"):
            st.error(st.session_state.pop("season_tool_error"))

        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.3, 1.3])
        with c1:
            season = st.text_input("Season", value=str(state.get("season") or "20242025"))
        with c2:
            auto_continue = st.checkbox("Continue automatically", value=False)
        with c3:
            fetch_clicked = st.button(
                "Fetch Raw Season Data",
                disabled=pending,
                use_container_width=True,
            )
        with c4:
            build_season = str(state.get("season") or season).strip()
            build_clicked = st.button(
                "Build Processed Season Report",
                disabled=not pending,
                use_container_width=True,
            )

        progress_bar = st.progress(0)
        progress_text = st.empty()

        def update_progress(progress):
            progress_bar.progress(min(int(progress.get("percent", 0)), 100))
            progress_text.caption(
                f"{progress.get('stage', '')}: {progress.get('current', 0)}/"
                f"{progress.get('total', 0)} ({progress.get('percent', 0):.1f}%) "
                f"{progress.get('message', '')}"
            )

        if fetch_clicked:
            season = season.strip()
            if not season:
                st.error("Enter a season before fetching raw data.")
                return

            with st.spinner(f"Fetching raw game data for {season}."):
                try:
                    fetch_summary, fetch_log = run_fetch_raw_season_data(season, game_type, update_progress)
                    st.success(
                        f"Raw fetch complete for {season}: "
                        f"{fetch_summary.get('fetched', 0)} fetched, "
                        f"{fetch_summary.get('cached', 0)} cached, "
                        f"{fetch_summary.get('failed', 0)} failed."
                    )
                    with st.expander("Raw fetch log"):
                        st.text(fetch_log)

                    if auto_continue:
                        with st.spinner(f"Building processed report for {season}."):
                            build_summary, build_log = run_build_processed_season_report(
                                season,
                                game_type,
                                update_progress,
                            )
                            clear_data_caches()
                            st.cache_data.clear()
                            st.success(
                                f"Processed report complete: "
                                f"{build_summary.get('processed_reports', 0)} goalie reports written."
                            )
                            with st.expander("Processed report log"):
                                st.text(build_log)
                    else:
                        st.info("Raw data is ready. The processed report button is now available.")
                    st.session_state["season_tool_notice"] = (
                        f"Raw fetch complete for {season}. "
                        "Processed report was built automatically."
                        if auto_continue
                        else f"Raw fetch complete for {season}. Build processed report is now available."
                    )
                    st.rerun()
                except Exception as exc:
                    st.session_state["season_tool_error"] = f"Season raw fetch failed: {exc}"
                    st.rerun()

        if build_clicked:
            with st.spinner(f"Building processed report for {build_season}."):
                try:
                    build_summary, build_log = run_build_processed_season_report(
                        build_season,
                        game_type,
                        update_progress,
                    )
                    clear_data_caches()
                    st.cache_data.clear()
                    st.success(
                        f"Processed report complete: "
                        f"{build_summary.get('processed_reports', 0)} goalie reports written."
                    )
                    with st.expander("Processed report log"):
                        st.text(build_log)
                    st.session_state["season_tool_notice"] = (
                        f"Processed report complete for {build_season}: "
                        f"{build_summary.get('processed_reports', 0)} goalie reports written."
                    )
                    st.rerun()
                except Exception as exc:
                    st.session_state["season_tool_error"] = f"Processed report build failed: {exc}"
                    st.rerun()


st.set_page_config(layout="wide", page_title="SLC Dashboard")
st.title("System Load Coefficient Dashboard")

available_seasons = available_master_report_seasons()
if "20252026" not in available_seasons:
    available_seasons.insert(0, "20252026")

top_season, top_phase, top_update_button = st.columns([1.2, 2, 1])
with top_season:
    selected_season = st.selectbox("Season", available_seasons, index=0)
    set_active_data_season(selected_season)
with top_phase:
    game_phase = st.radio("Game Set", ["Regular Season", "Playoffs"], horizontal=True)
with top_update_button:
    st.write("")
    st.write("")
    update_requested = st.button("Update Local Data", type="primary")

if update_requested:
    game_type = {"Regular Season": 2, "Playoffs": 3}[game_phase]
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

render_season_data_tool(game_phase)

# Temporarily disabled while comparing pre-label and post-label predictive tables.
# render_late_xga_volatility_sandbox(game_phase, selected_season)
render_predictive_label_comparison(selected_season)
st.stop()

seed_audit = cached_goalie_audit("", game_phase, selected_season, get_master_report_version(selected_season))
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

    league_view = cached_league_leaderboard(game_phase, min_gp, selected_season, get_master_report_version(selected_season))
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

        proof = cached_late_xga_collapse_proof(selected_season, get_master_report_version(selected_season))
        proof_summary = proof.get("summary", {})
        proof_risk = proof.get("risk_by_schedule", pd.DataFrame())
        model_lift = proof.get("model_lift", pd.DataFrame())

        st.subheader("SLC Research Snapshot")
        if proof_risk.empty:
            st.info("Not enough league data is available to test late-xGA collapse risk.")
        else:
            normal_top15_lift = model_lift[
                (model_lift["Target"] == "Above team recent baseline, next game") &
                (model_lift["Schedule_Context"] == "Normal") &
                (model_lift["Model"] == "Schedule + Load + SLC")
            ] if not model_lift.empty else pd.DataFrame()
            stressed_top15_lift = model_lift[
                (model_lift["Target"] == "Above team recent baseline, next game") &
                (model_lift["Schedule_Context"] == "Stressed") &
                (model_lift["Model"] == "Schedule + Load + SLC")
            ] if not model_lift.empty else pd.DataFrame()
            cr1, cr2 = st.columns(2)
            cr1.metric("Games Tested", f"{proof_summary.get('Goalie_Games', 0):,}")
            if not normal_top15_lift.empty:
                row = normal_top15_lift.iloc[0]
                cr2.metric(
                    "Normal Schedule Signal",
                    f"{float(row.get('Lift_vs_Load_Baseline', 0)) * 100:+.1f} pts",
                    delta="SLC lift vs schedule/load",
                )

            if not stressed_top15_lift.empty:
                stressed_lift = float(stressed_top15_lift.iloc[0].get("Lift_vs_Load_Baseline", 0)) * 100
                if stressed_lift < 0:
                    st.warning(
                        "Current read: relief-weighted SLC is helping in normal schedule spots, "
                        "but it still breaks in stressed schedule spots. That stressed-context problem is the next thing to solve."
                    )
                else:
                    st.success(
                        "Current read: SLC is adding signal after schedule/load context, including stressed schedule spots."
                    )
            else:
                st.info("Current read: not enough stressed schedule rows are available for the compact signal check.")

            with st.expander("Show Research Tables", expanded=False):
                st.markdown("**Predictive Lift Comparison**")
                st.caption(
                    "Positive Lift vs Load means the added signal separated future collapse risk better than schedule and recent load alone."
                )
                if not model_lift.empty:
                    st.dataframe(
                        format_model_lift_table(model_lift),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.markdown("**Low-SLC vs High-SLC Collapse Rates**")
                st.dataframe(
                    format_collapse_proof_table(proof_risk),
                    use_container_width=True,
                    hide_index=True,
                )
        league_columns = [
            "Rank", "Goalie", "Team", "Team_System_Profile", "GP",
            "SLC_Grade", "SLC_Profile", "Pressure_Survival_Grade",
            "Pressure_Relief_Grade", "Battery_Fit_Grade", "Pressure_Efficiency_per_Sequence",
            "Sequences_per_Game",
        ]
        display_league = qualified[league_columns].rename(columns={
            "SLC_Grade": "SLC Grade",
            "SLC_Profile": "SLC Profile",
            "Pressure_Survival_Grade": "Pressure Survival",
            "Pressure_Relief_Grade": "Pressure Relief",
            "Battery_Fit_Grade": "Battery Fit",
            "Team_System_Profile": "Team System",
            "Pressure_Efficiency_per_Sequence": "Pressure Efficiency",
            "Sequences_per_Game": "Sequences / Game",
        })
        st.dataframe(display_league, use_container_width=True, hide_index=True)
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
    audit = cached_goalie_audit(goalie_key, game_phase, selected_season, get_master_report_version(selected_season))

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
        st.dataframe(team_profile, use_container_width=True, hide_index=True)
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
    c1.metric("Pressure Efficiency", f"{summary['Pressure_Efficiency_per_Sequence']:.3f}", delta=f"{verdict['efficiency_delta']:+.3f} vs median")
    c2.metric("Relief Capture", f"{summary['Relief_Capture_Rate']:.3f}", delta=f"{verdict['relief_capture_delta']:+.3f} vs median")
    c3.metric("Net Load / Sequence", f"{summary['Net_Load_per_Sequence']:.3f}", delta=f"{verdict['net_load_delta']:+.3f} vs median")
    c4.metric("Games", int(summary["GP"]))

    st.subheader("SLC Component Breakdown")
    st.caption(
        "Survival is puck-stopping value. Relief is pressure cooling/reset value. "
        "Pressure Cost is the burden that continued or returned to the team."
    )
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Survival / Game", f"{summary.get('Survival_Component_per_Game', 0):.3f}")
    sc2.metric("Relief / Game", f"{summary.get('Relief_Component_per_Game', 0):.3f}")
    sc3.metric("Pressure Cost / Game", f"{summary.get('Pressure_Cost_per_Game', 0):.3f}")
    sc4.metric("Net Component / Game", f"{summary.get('Net_Component_per_Game', 0):.3f}")

    if not game_log.empty:
        component_columns = [
            "Date", "Matchup", "SLC", "Raw_SLC", "SLC_Confidence", "Survival_Component",
            "Relief_Component", "Pressure_Cost_Component",
            "Net_Load_per_Sequence", "Sequences", "Shots",
        ]
        with st.expander("Game-Level SLC Components", expanded=False):
            component_log = game_log.copy()
            component_log["Date"] = component_log["Date"].apply(format_date)
            st.dataframe(
                component_log.sort_values("Date", ascending=False)[component_columns],
                use_container_width=True,
                hide_index=True,
            )

    if not translation.empty:
        with st.expander("Plain-English Metric Translation", expanded=False):
            st.dataframe(translation,use_container_width=True, hide_index=True)

    with st.expander("Counting Stat Ingredients", expanded=False):
        st.dataframe(evidence, use_container_width=True, hide_index=True)

    if not context.empty:
        with st.expander("League Context", expanded=False):
            st.dataframe(context, use_container_width=True, hide_index=True)

    st.subheader("Game Pressure Survival vs Pressure Relief")
    if game_log.empty:
        st.info("No game-level efficiency data is available for this goalie and game set.")
    else:
        st.plotly_chart(build_efficiency_workload_chart(game_log, summary, verdict), use_container_width=True)
        st.caption(
            "Each dot is one game. Higher means better pressure survival; farther right means more pressure relief. "
            "The best efficiency profile lives toward the upper-right."
        )

    st.subheader("Games")
    game_columns = [
        "Date", "Matchup", "SLC_Grade", "Pressure_Efficiency_per_Sequence",
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
            use_container_width=True
        )
        with st.expander("Game-Level Counting Details", expanded=False):
            st.dataframe(
                display_log.sort_values("Date", ascending=False)[detail_game_columns],
                use_container_width=True,
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
            st.dataframe(selected_periods, use_container_width=True, hide_index=True)

    render_downloads(goalie_key, game_phase, game_log, evidence)

with tab_team_decision:
    st.header(f"Team Goalie Decision: {game_phase}")
    available_teams = sorted(team for team in leaderboard["Team"].dropna().unique() if team)
    if not available_teams:
        st.info("No team-linked goalie data is available for this game set.")
    else:
        selected_team = st.selectbox("Select team", available_teams)
        decision = cached_team_goalie_decision(selected_team, game_phase, selected_season, get_master_report_version(selected_season))
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

import pandas as pd

from slc_prediction_engine import _goalie_game_research_rows
from utility import available_master_report_seasons, set_active_data_season


PROFILE_ORDER = {
    "Stable": 1,
    "Brittle": 2,
    "Controlled Chaos": 3,
    "Volatile": 4,
}


def _safe_scale(series):
    spread = float(series.quantile(0.75) - series.quantile(0.25))
    if spread > 0:
        return spread

    spread = float(series.max() - series.min())
    return spread if spread > 0 else 1.0


def _profile_label(fluctuation, load_sensitivity, fluctuation_cutoff, load_sensitivity_cutoff):
    high_fluctuation = fluctuation >= fluctuation_cutoff
    high_load_sensitivity = load_sensitivity >= load_sensitivity_cutoff
    if not high_fluctuation and not high_load_sensitivity:
        return "Stable"
    if not high_fluctuation and high_load_sensitivity:
        return "Brittle"
    if high_fluctuation and not high_load_sensitivity:
        return "Controlled Chaos"
    return "Volatile"


def _season_team_games(season):
    set_active_data_season(season)
    rows = _goalie_game_research_rows()
    if rows.empty:
        return rows

    return (
        rows[rows["Team"].astype(bool)]
        .dropna(subset=["Date"])
        .groupby(["Team", "Game_ID", "Date"], as_index=False)
        .agg({"Late_xGA": "sum"})
        .sort_values(["Team", "Date", "Game_ID"])
    )


def build_team_profile_table(season):
    team_games = _season_team_games(season)
    if team_games.empty:
        return pd.DataFrame()

    team_rows = []
    for team, group in team_games.groupby("Team"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)
        season_fluctuation = float(group["Late_xGA"].std()) if len(group) > 1 else 0.0
        group["Rest_Days"] = group["Date"].diff().dt.days.clip(lower=0, upper=10)
        group["Games_Last_7"] = [
            int((group.loc[: i - 1, "Date"] >= row["Date"] - pd.Timedelta(days=7)).sum())
            for i, row in group.iterrows()
        ]
        stressed = group[(group["Games_Last_7"] >= 3) | (group["Rest_Days"] <= 1)]
        normal = group[(group["Games_Last_7"] < 3) & ((group["Rest_Days"] > 1) | group["Rest_Days"].isna())]
        stressed_late_xga = float(stressed["Late_xGA"].mean()) if not stressed.empty else float(group["Late_xGA"].mean())
        normal_late_xga = float(normal["Late_xGA"].mean()) if not normal.empty else float(group["Late_xGA"].mean())
        load_sensitivity = stressed_late_xga - normal_late_xga
        team_rows.append({
            "Season": str(season),
            "Team": team,
            "Games": int(len(group)),
            "Avg_Late_xGA": round(float(group["Late_xGA"].mean()), 4),
            "Median_Late_xGA": round(float(group["Late_xGA"].median()), 4),
            "Fluctuation": season_fluctuation,
            "Stressed_Late_xGA": stressed_late_xga,
            "Normal_Late_xGA": normal_late_xga,
            "Load_Sensitivity": load_sensitivity,
        })

    profiles = pd.DataFrame(team_rows)
    fluctuation_cutoff = float(profiles["Fluctuation"].median())
    load_sensitivity_cutoff = float(profiles["Load_Sensitivity"].median())
    fluctuation_scale = _safe_scale(profiles["Fluctuation"])
    load_sensitivity_scale = _safe_scale(profiles["Load_Sensitivity"])

    profiles["Fluctuation_Cutoff"] = fluctuation_cutoff
    profiles["Load_Sensitivity_Cutoff"] = load_sensitivity_cutoff
    profiles["Fluctuation_Distance"] = profiles["Fluctuation"] - fluctuation_cutoff
    profiles["Load_Sensitivity_Distance"] = profiles["Load_Sensitivity"] - load_sensitivity_cutoff
    profiles["Fluctuation_Distance_Scaled"] = profiles["Fluctuation_Distance"].abs() / fluctuation_scale
    profiles["Load_Sensitivity_Distance_Scaled"] = profiles["Load_Sensitivity_Distance"].abs() / load_sensitivity_scale
    profiles["Profile_Confidence"] = profiles[[
        "Fluctuation_Distance_Scaled",
        "Load_Sensitivity_Distance_Scaled",
    ]].min(axis=1)
    profiles["Confidence_Label"] = pd.cut(
        profiles["Profile_Confidence"],
        bins=[-0.01, 0.15, 0.35, 100],
        labels=["Borderline", "Moderate", "Strong"],
    ).astype(str)
    profiles["Profile"] = profiles.apply(
        lambda row: _profile_label(
            row["Fluctuation"],
            row["Load_Sensitivity"],
            fluctuation_cutoff,
            load_sensitivity_cutoff,
        ),
        axis=1,
    )

    display_cols = [
        "Season",
        "Team",
        "Profile",
        "Confidence_Label",
        "Profile_Confidence",
        "Games",
        "Avg_Late_xGA",
        "Median_Late_xGA",
        "Fluctuation",
        "Stressed_Late_xGA",
        "Normal_Late_xGA",
        "Load_Sensitivity",
        "Fluctuation_Cutoff",
        "Load_Sensitivity_Cutoff",
        "Fluctuation_Distance",
        "Load_Sensitivity_Distance",
    ]
    return profiles[display_cols].sort_values(["Season", "Profile", "Team"]).reset_index(drop=True)


def build_profile_summary(team_profiles):
    if team_profiles.empty:
        return pd.DataFrame()

    summary = (
        team_profiles
        .groupby(["Season", "Profile"], as_index=False)
        .agg(
            Teams=("Team", "count"),
            Avg_Confidence=("Profile_Confidence", "mean"),
            Borderline_Teams=("Confidence_Label", lambda values: int((values == "Borderline").sum())),
            Fluctuation_Mean=("Fluctuation", "mean"),
            Fluctuation_Spread=("Fluctuation", "std"),
            Load_Sensitivity_Mean=("Load_Sensitivity", "mean"),
            Load_Sensitivity_Spread=("Load_Sensitivity", "std"),
        )
        .fillna(0)
    )
    summary["Profile_Order"] = summary["Profile"].map(PROFILE_ORDER).fillna(99)
    return summary.sort_values(["Season", "Profile_Order"]).drop(columns=["Profile_Order"]).reset_index(drop=True)


def build_profile_transition_summary(team_profiles):
    if team_profiles.empty:
        return pd.DataFrame()

    profile_counts = (
        team_profiles
        .sort_values(["Team", "Season"])
        .groupby("Team")
        .agg(
            Seasons=("Season", "count"),
            Profiles=("Profile", lambda values: " -> ".join(values)),
            Unique_Profiles=("Profile", "nunique"),
            Avg_Confidence=("Profile_Confidence", "mean"),
            Borderline_Seasons=("Confidence_Label", lambda values: int((values == "Borderline").sum())),
        )
        .reset_index()
    )
    return profile_counts.sort_values(["Unique_Profiles", "Borderline_Seasons", "Team"], ascending=[False, False, True])


def build_team_profile_diagnostics(seasons=None):
    seasons = seasons or available_master_report_seasons()
    team_profiles = pd.concat(
        [build_team_profile_table(season) for season in seasons],
        ignore_index=True,
    )
    return {
        "team_profiles": team_profiles,
        "profile_summary": build_profile_summary(team_profiles),
        "transition_summary": build_profile_transition_summary(team_profiles),
    }

import numpy as np
import pandas as pd

from utility import (
    _combine_stats,
    _get_goalie_team_abbrev,
    _get_period_stats,
    _late_xga,
    _load_raw_game,
    _slc_weights_for_profile,
    calculate_service_tax,
    calculate_slc_score,
    load_master_reports,
    report_team_profile,
)


def _proof_safe_div(numerator, denominator):
    return round(float(numerator) / float(denominator), 3) if denominator else 0.0


def _goalie_game_research_rows():
    """Build temporary goalie-game rows from master_report for proof analysis."""
    master_report = load_master_reports()
    rows = []

    for goalie_key, games in master_report.items():
        for game_id, report in games.items():
            stats = report.get("total", {}).get("stats", {})
            raw_game = _load_raw_game(game_id)
            date_value = report.get("gameDate") or raw_game.get("gameDate")
            sequences = stats.get("Sequences", 0)
            slc, scored_stats = calculate_slc_score(stats, team_profile=report_team_profile(report))
            service, tax = calculate_service_tax(stats, include_baseline=False)

            rows.append({
                "Goalie_Key": goalie_key,
                "Game_ID": str(game_id),
                "Date": pd.to_datetime(date_value, errors="coerce"),
                "Team": _get_goalie_team_abbrev(raw_game, report.get("goalie_id")) or "",
                "Volatility_Profile": report_team_profile(report) or "Unknown",
                "SLC": slc,
                "Sovereignty_Rate": float(scored_stats.get("Survival_Rate", 0) or 0),
                "Relief_Rate": float(scored_stats.get("Relief_Rate", 0) or 0),
                "Pressure_Cost_Rate": float(scored_stats.get("Pressure_Cost_Rate", 0) or 0),
                "Pressure_Efficiency": _proof_safe_div(slc, sequences),
                "Net_Load_per_Sequence": _proof_safe_div(service - tax, sequences),
                "Sequences": sequences,
                "Late_xGA": _late_xga(report),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["Goalie_Key", "Date", "Game_ID"]).reset_index(drop=True)


def _attach_team_recent_late_xga_baseline(df):
    if df.empty or "Team" not in df.columns:
        return df

    team_games = (
        df[df["Team"].astype(bool)]
        .groupby(["Team", "Game_ID", "Date"], as_index=False)
        .agg({"Late_xGA": "sum"})
        .sort_values(["Team", "Date", "Game_ID"])
    )
    if team_games.empty:
        df["Team_Late_xGA_Roll5"] = np.nan
        df["Team_Relative_Collapse"] = np.nan
        df["Team_Rest_Days"] = np.nan
        df["Team_Games_Last_7"] = np.nan
        return df

    frames = []
    for _, group in team_games.groupby("Team"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)
        group["Team_Late_xGA_Roll5"] = group["Late_xGA"].shift(1).rolling(5, min_periods=3).mean()
        group["Team_Rest_Days"] = group["Date"].diff().dt.days.clip(lower=0, upper=10)
        group["Team_Games_Last_7"] = [
            int((group.loc[: i - 1, "Date"] >= row["Date"] - pd.Timedelta(days=7)).sum())
            for i, row in group.iterrows()
        ]
        group["Team_Relative_Collapse"] = np.where(
            group["Team_Late_xGA_Roll5"].notna(),
            (group["Late_xGA"] >= group["Team_Late_xGA_Roll5"] + 0.35).astype(int),
            np.nan,
        )
        frames.append(group)

    team_context = pd.concat(frames, ignore_index=True)[[
        "Team",
        "Game_ID",
        "Team_Late_xGA_Roll5",
        "Team_Relative_Collapse",
        "Team_Rest_Days",
        "Team_Games_Last_7",
    ]]
    return df.merge(team_context, on=["Team", "Game_ID"], how="left")


def _attach_team_volatility_profiles(df):
    if df.empty or "Team" not in df.columns:
        return df

    df = df.drop(columns=["Volatility_Profile"], errors="ignore")

    team_games = (
        df[df["Team"].astype(bool)]
        .groupby(["Team", "Game_ID", "Date"], as_index=False)
        .agg({"Late_xGA": "sum"})
        .sort_values(["Team", "Date", "Game_ID"])
    )
    if team_games.empty:
        df["Volatility_Profile"] = "Unknown"
        return df

    team_rows = []
    for team, group in team_games.groupby("Team"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)
        rolling_avg = group["Late_xGA"].shift(1).rolling(5, min_periods=3).mean()
        rolling_fluctuation = group["Late_xGA"].shift(1).rolling(5, min_periods=3).std()
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
            "Team": team,
            "Team_Fluctuation": season_fluctuation,
            "Team_Load_Sensitivity": load_sensitivity,
        })

    profiles = pd.DataFrame(team_rows)
    fluctuation_line = profiles["Team_Fluctuation"].median()
    load_sensitivity_line = profiles["Team_Load_Sensitivity"].median()

    def label_profile(row):
        high_fluctuation = row["Team_Fluctuation"] >= fluctuation_line
        high_load_sensitivity = row["Team_Load_Sensitivity"] >= load_sensitivity_line
        if not high_fluctuation and not high_load_sensitivity:
            return "Stable"
        if not high_fluctuation and high_load_sensitivity:
            return "Brittle"
        if high_fluctuation and not high_load_sensitivity:
            return "Controlled Chaos"
        return "Volatile"

    profiles["Volatility_Profile"] = profiles.apply(label_profile, axis=1)
    return df.merge(profiles, on="Team", how="left")


def build_team_context_by_game(master_report=None):
    """Build rolling team volatility context keyed by (team, game_id)."""
    master_report = master_report if master_report is not None else load_master_reports()
    rows = []
    for games in master_report.values():
        for game_id, report in games.items():
            raw_game = _load_raw_game(game_id)
            team = _get_goalie_team_abbrev(raw_game, report.get("goalie_id"))
            date_value = report.get("gameDate") or raw_game.get("gameDate")
            if not team:
                continue
            rows.append({
                "Team": team,
                "Game_ID": str(game_id),
                "Date": pd.to_datetime(date_value, errors="coerce"),
                "Late_xGA": _late_xga(report),
            })

    if not rows:
        return {}

    team_games = (
        pd.DataFrame(rows)
        .dropna(subset=["Date"])
        .groupby(["Team", "Game_ID", "Date"], as_index=False)
        .agg({"Late_xGA": "sum"})
        .sort_values(["Team", "Date", "Game_ID"])
    )
    if team_games.empty:
        return {}

    frames = []
    for _, group in team_games.groupby("Team"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)
        group["Rolling_Avg"] = group["Late_xGA"].shift(1).rolling(5, min_periods=3).mean()
        group["Typical_Fluctuation"] = group["Late_xGA"].shift(1).rolling(5, min_periods=3).std()
        group["Rest_Days"] = group["Date"].diff().dt.days.clip(lower=0, upper=10)
        group["Games_Last_7"] = [
            int((group.loc[: i - 1, "Date"] >= row["Date"] - pd.Timedelta(days=7)).sum())
            for i, row in group.iterrows()
        ]
        group["Stressed_Game"] = (group["Games_Last_7"] >= 3) | (group["Rest_Days"] <= 1)
        prior_stressed_mean = []
        prior_normal_mean = []
        for i, row in group.iterrows():
            prior = group.loc[: i - 1]
            stressed_prior = prior[prior["Stressed_Game"]]
            normal_prior = prior[~prior["Stressed_Game"]]
            prior_stressed_mean.append(float(stressed_prior["Late_xGA"].mean()) if len(stressed_prior) >= 2 else np.nan)
            prior_normal_mean.append(float(normal_prior["Late_xGA"].mean()) if len(normal_prior) >= 2 else np.nan)
        group["Load_Sensitivity"] = np.array(prior_stressed_mean) - np.array(prior_normal_mean)
        frames.append(group)

    context = pd.concat(frames, ignore_index=True)
    fluctuation_line = context["Typical_Fluctuation"].median()
    load_sensitivity_line = context["Load_Sensitivity"].median()

    def label_profile(row):
        if pd.isna(row["Typical_Fluctuation"]) or pd.isna(row["Load_Sensitivity"]):
            return "Unknown"
        high_fluctuation = row["Typical_Fluctuation"] >= fluctuation_line
        high_load_sensitivity = row["Load_Sensitivity"] >= load_sensitivity_line
        if not high_fluctuation and not high_load_sensitivity:
            return "Stable"
        if not high_fluctuation and high_load_sensitivity:
            return "Brittle"
        if high_fluctuation and not high_load_sensitivity:
            return "Controlled Chaos"
        return "Volatile"

    context["Volatility_Profile"] = context.apply(label_profile, axis=1)
    context_map = {}
    for _, row in context.iterrows():
        context_map[(row["Team"], str(row["Game_ID"]))] = {
            "team": row["Team"],
            "volatility_profile": row["Volatility_Profile"],
            "late_xga_rolling_avg": round(float(row["Rolling_Avg"]), 4) if pd.notna(row["Rolling_Avg"]) else None,
            "typical_fluctuation": round(float(row["Typical_Fluctuation"]), 4) if pd.notna(row["Typical_Fluctuation"]) else None,
            "load_sensitivity": round(float(row["Load_Sensitivity"]), 4) if pd.notna(row["Load_Sensitivity"]) else None,
        }
    return context_map


def attach_team_context_to_master(master_report=None):
    """Attach rolling team context to each stored report."""
    master_report = master_report if master_report is not None else load_master_reports()
    context_map = build_team_context_by_game(master_report)
    updated = 0
    for games in master_report.values():
        for game_id, report in games.items():
            raw_game = _load_raw_game(game_id)
            team = _get_goalie_team_abbrev(raw_game, report.get("goalie_id"))
            context = context_map.get((team, str(game_id))) if team else None
            if not context:
                continue
            if report.get("team_context") != context:
                report["team_context"] = context
                updated += 1
    return master_report, updated


def build_late_xga_collapse_proof():
    """
    Test whether low recent SLC is associated with more future late-xGA collapses.

    This is intentionally in-memory: the selected season master report remains the source of truth.
    """
    df = _goalie_game_research_rows()
    if df.empty:
        return {
            "summary": {},
            "risk_by_schedule": pd.DataFrame(),
            "risk_by_feature": pd.DataFrame(),
        }

    df = _attach_team_recent_late_xga_baseline(df)
    if "Volatility_Profile" not in df.columns or df["Volatility_Profile"].eq("Unknown").all():
        df = _attach_team_volatility_profiles(df)
    frames = []
    for _, group in df.groupby("Goalie_Key"):
        group = group.sort_values(["Date", "Game_ID"]).reset_index(drop=True)

        group["Rest_Days"] = group["Team_Rest_Days"]
        group["Games_Last_7"] = group["Team_Games_Last_7"]
        group["Schedule_Context"] = group.apply(
            lambda row: "Stressed"
            if row["Games_Last_7"] >= 3 or (pd.notna(row["Rest_Days"]) and row["Rest_Days"] <= 1)
            else "Normal",
            axis=1,
        )
        group["SLC_Roll3"] = group["SLC"].shift(1).rolling(3, min_periods=1).mean()
        group["SLC_Roll5"] = group["SLC"].shift(1).rolling(5, min_periods=1).mean()
        group["Sovereignty_Rate_Roll3"] = group["Sovereignty_Rate"].shift(1).rolling(3, min_periods=1).mean()
        group["Relief_Rate_Roll3"] = group["Relief_Rate"].shift(1).rolling(3, min_periods=1).mean()
        group["Pressure_Cost_Rate_Roll3"] = group["Pressure_Cost_Rate"].shift(1).rolling(3, min_periods=1).mean()
        group["SLC_Roll3"] = group.apply(
            lambda row: (
                (_slc_weights_for_profile(row.get("Volatility_Profile"))["survival"] * row["Sovereignty_Rate_Roll3"]) +
                (_slc_weights_for_profile(row.get("Volatility_Profile"))["relief"] * row["Relief_Rate_Roll3"]) -
                (_slc_weights_for_profile(row.get("Volatility_Profile"))["pressure_cost"] * row["Pressure_Cost_Rate_Roll3"])
            ),
            axis=1,
        )
        group["PE_Roll3"] = group["Pressure_Efficiency"].shift(1).rolling(3, min_periods=1).mean()
        group["Late_xGA_Roll5"] = group["Team_Late_xGA_Roll5"]
        group["Sequences_Roll3"] = group["Sequences"].shift(1).rolling(3, min_periods=1).mean()

        late_xga_values = group["Late_xGA"].astype(float).to_numpy()
        baseline_values = group["Late_xGA_Roll5"].astype(float).to_numpy()
        for horizon in [1, 3, 5]:
            future_deviation = []
            for i in range(len(group)):
                start = i + 1
                end = i + 1 + horizon
                late_window = late_xga_values[start:end]
                if end <= len(group) and len(late_window) and not pd.isna(baseline_values[i]):
                    future_deviation.append(float(np.nanmax(np.abs(late_window - baseline_values[i]))))
                else:
                    future_deviation.append(np.nan)
            group[f"Future_Late_xGA_Deviation_H{horizon}"] = future_deviation

        for collapse_col in ["Team_Relative_Collapse"]:
            values = group[collapse_col].astype(float).to_numpy()
            for horizon in [1, 3, 5]:
                future = []
                for i in range(len(group)):
                    start = i + 1
                    end = i + 1 + horizon
                    window = values[start:end]
                    if end <= len(group) and len(window) and not np.all(np.isnan(window)):
                        future.append(float(np.nanmax(window)))
                    else:
                        future.append(np.nan)
                group[f"Future_{collapse_col}_H{horizon}"] = future

        frames.append(group)

    proof_df = pd.concat(frames, ignore_index=True)

    risk_rows = []
    for target in [
        "Future_Team_Relative_Collapse_H1",
        "Future_Team_Relative_Collapse_H3",
        "Future_Team_Relative_Collapse_H5",
    ]:
        for context, subset in proof_df.groupby("Schedule_Context"):
            use = subset[["SLC_Roll3", target]].dropna()
            if len(use) < 50:
                continue

            use = use.copy()
            use["SLC_Bucket"] = pd.qcut(use["SLC_Roll3"], 3, labels=["Low", "Mid", "High"], duplicates="drop")
            risks = use.groupby("SLC_Bucket", observed=True)[target].mean()
            counts = use.groupby("SLC_Bucket", observed=True)[target].count()
            if not {"Low", "High"}.issubset(set(risks.index)):
                continue

            risk_rows.append({
                "Target": target,
                "Schedule_Context": context,
                "Low_SLC_Risk": round(float(risks.loc["Low"]), 3),
                "High_SLC_Risk": round(float(risks.loc["High"]), 3),
                "Low_Minus_High": round(float(risks.loc["Low"] - risks.loc["High"]), 3),
                "Low_N": int(counts.loc["Low"]),
                "High_N": int(counts.loc["High"]),
                "N": int(len(use)),
            })

    feature_rows = []
    for feature in ["SLC_Roll3", "SLC_Roll5", "PE_Roll3"]:
        use = proof_df[[feature, "Future_Team_Relative_Collapse_H1"]].dropna()
        if len(use) < 50:
            continue
        corr = np.corrcoef(use[feature], use["Future_Team_Relative_Collapse_H1"])[0, 1]
        feature_rows.append({
            "Feature": feature,
            "Target": "Future_Team_Relative_Collapse_H1",
            "Pearson": round(float(corr), 3),
            "R2": round(float(corr * corr), 4),
            "N": int(len(use)),
        })

    risk_by_schedule = pd.DataFrame(risk_rows)
    risk_by_feature = pd.DataFrame(feature_rows)
    model_lift = _build_collapse_model_lift(proof_df)
    model_lift_by_profile = _build_collapse_model_lift_by_profile(proof_df)
    profile_weight_optimizer = _build_profile_weight_optimizer(proof_df)
    profile_weight_validation = _build_profile_weight_validation(proof_df)
    summary = {
        "Goalie_Games": int(len(proof_df)),
        "Usable_Next_Game_Rows": int(proof_df["Future_Team_Relative_Collapse_H1"].notna().sum()),
        "Team_Baseline_Window": 5,
        "Team_Baseline_Buffer": 0.35,
    }

    return {
        "summary": summary,
        "risk_by_schedule": risk_by_schedule,
        "risk_by_feature": risk_by_feature,
        "model_lift": model_lift,
        "model_lift_by_profile": model_lift_by_profile,
        "profile_weight_optimizer": profile_weight_optimizer,
        "profile_weight_validation": profile_weight_validation,
        "same_game": _build_same_game_heatsink_proof(),
    }


def _period_pair_stats(report, periods):
    return _combine_stats([
        _get_period_stats(report, period)
        for period in periods
    ])


def _build_same_game_heatsink_proof():
    rows = []
    master_report = load_master_reports()

    for goalie_key, games in master_report.items():
        for game_id, report in games.items():
            early_stats = _period_pair_stats(report, [1, 2])
            p3_stats = _get_period_stats(report, 3)
            early_shots = early_stats.get("S_saves", 0) + early_stats.get("S_goals", 0)
            p3_shots = p3_stats.get("S_saves", 0) + p3_stats.get("S_goals", 0)
            if early_shots <= 0 or p3_shots <= 0:
                continue

            early_slc, scored_early = calculate_slc_score(early_stats)
            early_sequences = early_stats.get("Sequences", 0)
            early_service, early_tax = calculate_service_tax(early_stats, include_baseline=False)
            early_relief = _proof_safe_div(early_service - early_tax, early_sequences)
            early_relief_rate = float(scored_early.get("Relief_Rate", 0) or 0)
            early_cost_rate = float(scored_early.get("Pressure_Cost_Rate", 0) or 0)
            early_pressure = float(early_stats.get("Pressure_Weight", 0) or early_shots)
            p3_xga = float(p3_stats.get("xG", 0) or 0)

            rows.append({
                "Goalie_Key": goalie_key,
                "Game_ID": str(game_id),
                "Date": pd.to_datetime(report.get("gameDate"), errors="coerce"),
                "Early_SLC": early_slc,
                "Early_Relief": early_relief,
                "Early_Relief_Rate": early_relief_rate,
                "Early_Cost_Rate": early_cost_rate,
                "Early_Net_Load": early_service - early_tax,
                "Early_Pressure": early_pressure,
                "Early_Sequences": early_sequences,
                "P3_xGA": p3_xga,
            })

    if not rows:
        return {
            "summary": {},
            "risk_by_early_slc": pd.DataFrame(),
            "lift": pd.DataFrame(),
        }

    df = pd.DataFrame(rows)
    p3_threshold = df["P3_xGA"].quantile(0.85)
    df["P3_Collapse"] = (df["P3_xGA"] >= p3_threshold).astype(int)
    df["Early_Pressure_Context"] = np.where(
        df["Early_Pressure"] >= df["Early_Pressure"].median(),
        "High early pressure",
        "Lower early pressure",
    )

    risk_rows = []
    for context, subset in df.groupby("Early_Pressure_Context"):
        use = subset[["Early_SLC", "Early_Relief_Rate", "P3_Collapse"]].dropna()
        if len(use) < 100:
            continue

        for feature, label in [
            ("Early_SLC", "Early SLC"),
            ("Early_Relief_Rate", "Early relief rate"),
        ]:
            feature_use = use[[feature, "P3_Collapse"]].copy()
            if feature_use[feature].nunique() < 5:
                continue
            feature_use["Bucket"] = pd.qcut(
                feature_use[feature],
                3,
                labels=["Low", "Middle", "High"],
                duplicates="drop",
            )
            rates = feature_use.groupby("Bucket", observed=True)["P3_Collapse"].mean()
            counts = feature_use.groupby("Bucket", observed=True)["P3_Collapse"].count()
            if not {"Low", "High"}.issubset(set(rates.index)):
                continue

            gap = rates.loc["Low"] - rates.loc["High"]
            risk_rows.append({
                "Context": context,
                "Signal": label,
                "Low_Risk": round(float(rates.loc["Low"]), 3),
                "High_Risk": round(float(rates.loc["High"]), 3),
                "Low_Minus_High": round(float(gap), 3),
                "Low_N": int(counts.loc["Low"]),
                "High_N": int(counts.loc["High"]),
                "N": int(len(feature_use)),
            })

    lift_rows = []
    split_date = df["Date"].quantile(0.70)
    train_df = df[df["Date"] <= split_date]
    test_df = df[df["Date"] > split_date]
    model_specs = {
        "Early Pressure Only": ["Early_Pressure", "Early_Sequences"],
        "Early Pressure + SLC": ["Early_Pressure", "Early_Sequences", "Early_SLC"],
        "Early Pressure + Relief Rate": ["Early_Pressure", "Early_Sequences", "Early_Relief_Rate"],
    }
    previous = None
    for model_name, features in model_specs.items():
        model = _fit_linear_risk_model(train_df, "P3_Collapse", features)
        preds = _predict_linear_risk(model, test_df)
        added_signal_direction = ""
        if model is not None and model_name == "Early Pressure + SLC":
            slc_coef = model["coef"][1 + features.index("Early_SLC")]
            added_signal_direction = "Higher early SLC raised predicted risk" if slc_coef > 0 else "Higher early SLC lowered predicted risk"
        if model is not None and model_name == "Early Pressure + Relief Rate":
            relief_coef = model["coef"][1 + features.index("Early_Relief_Rate")]
            added_signal_direction = "Higher early relief rate raised predicted risk" if relief_coef > 0 else "Higher early relief rate lowered predicted risk"
        use = pd.DataFrame({
            "P3_Collapse": test_df["P3_Collapse"],
            "Predicted_Risk": preds,
        }).dropna()
        if len(use) < 100 or use["Predicted_Risk"].nunique() < 3:
            continue
        use["Risk_Bucket"] = pd.qcut(use["Predicted_Risk"], 3, labels=False, duplicates="drop")
        rates = use.groupby("Risk_Bucket", observed=True)["P3_Collapse"].mean()
        counts = use.groupby("Risk_Bucket", observed=True)["P3_Collapse"].count()
        if len(rates) < 2:
            continue
        low_bucket = rates.index.min()
        high_bucket = rates.index.max()
        separation = float(rates.loc[high_bucket] - rates.loc[low_bucket])
        added_lift = np.nan if previous is None else separation - previous
        previous = separation
        lift_rows.append({
            "Model": model_name,
            "Low_Predicted_Risk_Rate": round(float(rates.loc[low_bucket]), 3),
            "High_Predicted_Risk_Rate": round(float(rates.loc[high_bucket]), 3),
            "Risk_Separation": round(separation, 3),
            "Added_Lift_vs_Previous": round(float(added_lift), 3) if pd.notna(added_lift) else np.nan,
            "Added_Signal_Direction": added_signal_direction,
            "Low_N": int(counts.loc[low_bucket]),
            "High_N": int(counts.loc[high_bucket]),
            "N": int(len(use)),
        })

    return {
        "summary": {
            "Games": int(len(df)),
            "P3_xGA_Top15_Threshold": round(float(p3_threshold), 3),
        },
        "risk_by_early_slc": pd.DataFrame(risk_rows),
        "lift": pd.DataFrame(lift_rows),
    }


def _rank_feature(series, ascending=True):
    ranked = series.rank(pct=True, ascending=ascending)
    return ranked.fillna(ranked.median()).fillna(0.5)


def _fit_linear_risk_model(train_df, target_col, feature_cols):
    train = train_df[[target_col] + feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < len(feature_cols) + 20 or train[target_col].nunique() < 2:
        return None

    means = train[feature_cols].mean()
    stds = train[feature_cols].std().replace(0, 1).fillna(1)
    x = (train[feature_cols] - means) / stds
    matrix = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    y = train[target_col].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    return {"coef": coef, "means": means, "stds": stds, "features": feature_cols}


def _predict_linear_risk(model, df):
    if model is None:
        return pd.Series(np.nan, index=df.index)

    features = model["features"]
    x = df[features].replace([np.inf, -np.inf], np.nan)
    valid = x.notna().all(axis=1)
    preds = pd.Series(np.nan, index=df.index)
    if not valid.any():
        return preds

    scaled = (x.loc[valid] - model["means"]) / model["stds"]
    matrix = np.column_stack([np.ones(len(scaled)), scaled.to_numpy(dtype=float)])
    preds.loc[valid] = np.clip(matrix @ model["coef"], 0, 1)
    return preds


def _build_collapse_model_lift(proof_df):
    rows = []
    targets = {
        "Future_Team_Relative_Collapse_H1": "Above team recent baseline, next game",
    }
    model_specs = {
        "Schedule Only": ["Rest_Days", "Games_Last_7"],
        "Schedule + Recent Load": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3"],
        "Schedule + Load + SLC": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "SLC_Roll3"],
        "Schedule + Load + Relief Rate": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "Relief_Rate_Roll3"],
        "Schedule + Load + SLC + Relief Rate": [
            "Rest_Days",
            "Games_Last_7",
            "Late_xGA_Roll5",
            "Sequences_Roll3",
            "SLC_Roll3",
            "Relief_Rate_Roll3",
        ],
        "Schedule + Load + SLC + Relief + Cost": [
            "Rest_Days",
            "Games_Last_7",
            "Late_xGA_Roll5",
            "Sequences_Roll3",
            "SLC_Roll3",
            "Relief_Rate_Roll3",
            "Pressure_Cost_Rate_Roll3",
        ],
    }

    proof_df = proof_df.sort_values("Date").copy()
    split_date = proof_df["Date"].quantile(0.70)
    train_df = proof_df[proof_df["Date"] <= split_date]
    test_df = proof_df[proof_df["Date"] > split_date]

    for schedule_context in ["All", "Normal", "Stressed"]:
        context_df = test_df if schedule_context == "All" else test_df[test_df["Schedule_Context"] == schedule_context]
        if context_df.empty:
            continue

        for target_col, target_label in targets.items():
            previous_separation = None
            load_baseline_separation = None
            for model_name, feature_cols in model_specs.items():
                model = _fit_linear_risk_model(train_df, target_col, feature_cols)
                predictions = _predict_linear_risk(model, context_df)
                model_use = pd.DataFrame({
                    target_col: context_df[target_col],
                    "Predicted_Risk": predictions,
                }).dropna()
                if len(model_use) < 100 or model_use["Predicted_Risk"].nunique() < 3:
                    continue

                model_use = model_use.copy()
                model_use["Risk_Bucket"] = pd.qcut(
                    model_use["Predicted_Risk"],
                    3,
                    labels=False,
                    duplicates="drop",
                )
                rates = model_use.groupby("Risk_Bucket", observed=True)[target_col].mean()
                counts = model_use.groupby("Risk_Bucket", observed=True)[target_col].count()
                if len(rates) < 2:
                    continue

                low_bucket = rates.index.min()
                high_bucket = rates.index.max()
                low_rate = float(rates.loc[low_bucket])
                high_rate = float(rates.loc[high_bucket])
                separation = high_rate - low_rate
                added_lift = np.nan if previous_separation is None else separation - previous_separation
                previous_separation = separation
                if model_name == "Schedule + Recent Load":
                    load_baseline_separation = separation
                lift_vs_load = (
                    separation - load_baseline_separation
                    if load_baseline_separation is not None and model_name != "Schedule + Recent Load"
                    else np.nan
                )

                rows.append({
                    "Target": target_label,
                    "Schedule_Context": schedule_context,
                    "Model": model_name,
                    "Low_Predicted_Risk_Rate": round(low_rate, 3),
                    "High_Predicted_Risk_Rate": round(high_rate, 3),
                    "Risk_Separation": round(separation, 3),
                    "Added_Lift_vs_Previous": round(float(added_lift), 3) if pd.notna(added_lift) else np.nan,
                    "Lift_vs_Load_Baseline": round(float(lift_vs_load), 3) if pd.notna(lift_vs_load) else np.nan,
                    "Low_N": int(counts.loc[low_bucket]),
                    "High_N": int(counts.loc[high_bucket]),
                    "N": int(len(model_use)),
                })

    return pd.DataFrame(rows)


def _build_collapse_model_lift_by_profile(proof_df):
    if "Volatility_Profile" not in proof_df.columns:
        return pd.DataFrame()

    rows = []
    model_specs = {
        "Schedule + Recent Load": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3"],
        "Schedule + Recent Load + SLC": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "SLC_Roll3"],
        "Schedule + Recent Load + Sovereignty": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "Sovereignty_Rate_Roll3"],
        "Schedule + Recent Load + Relief Rate": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "Relief_Rate_Roll3"],
        "Schedule + Recent Load + Pressure": ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3", "Pressure_Cost_Rate_Roll3"],
    }

    proof_df = proof_df.sort_values("Date").copy()
    split_date = proof_df["Date"].quantile(0.70)
    train_df = proof_df[proof_df["Date"] <= split_date]
    test_df = proof_df[proof_df["Date"] > split_date]

    for profile, profile_test_df in test_df.groupby("Volatility_Profile"):
        if not profile or profile == "Unknown":
            continue

        profile_train_df = train_df[train_df["Volatility_Profile"] == profile]
        if profile_train_df.empty:
            continue
        target_col, target_label = _profile_prediction_target(profile)

        load_baseline_separation = None
        for model_name, feature_cols in model_specs.items():
            model = _fit_linear_risk_model(profile_train_df, target_col, feature_cols)
            predictions = _predict_linear_risk(model, profile_test_df)
            model_use = pd.DataFrame({
                target_col: profile_test_df[target_col],
                "Predicted_Risk": predictions,
            }).dropna()
            if len(model_use) < 50 or model_use["Predicted_Risk"].nunique() < 3:
                continue

            model_use = model_use.copy()
            model_use["Risk_Bucket"] = pd.qcut(
                model_use["Predicted_Risk"],
                3,
                labels=False,
                duplicates="drop",
            )
            rates = model_use.groupby("Risk_Bucket", observed=True)[target_col].mean()
            counts = model_use.groupby("Risk_Bucket", observed=True)[target_col].count()
            if len(rates) < 2:
                continue

            low_bucket = rates.index.min()
            high_bucket = rates.index.max()
            low_rate = float(rates.loc[low_bucket])
            high_rate = float(rates.loc[high_bucket])
            separation = high_rate - low_rate
            if model_name == "Schedule + Recent Load":
                load_baseline_separation = separation
            lift_vs_load = (
                separation - load_baseline_separation
                if load_baseline_separation is not None and model_name != "Schedule + Recent Load"
                else np.nan
            )

            rows.append({
                "Volatility_Profile": profile,
                "Target": target_label,
                "Model": model_name,
                "Low_Predicted_Risk_Rate": round(low_rate, 3),
                "High_Predicted_Risk_Rate": round(high_rate, 3),
                "Risk_Separation": round(separation, 3),
                "Lift_vs_Load_Baseline": round(float(lift_vs_load), 3) if pd.notna(lift_vs_load) else np.nan,
                "Low_N": int(counts.loc[low_bucket]),
                "High_N": int(counts.loc[high_bucket]),
                "N": int(len(model_use)),
            })

    return pd.DataFrame(rows)


def _risk_separation_for_features(train_df, test_df, target_col, feature_cols):
    model = _fit_linear_risk_model(train_df, target_col, feature_cols)
    predictions = _predict_linear_risk(model, test_df)
    model_use = pd.DataFrame({
        target_col: test_df[target_col],
        "Predicted_Risk": predictions,
    }).dropna()
    if len(model_use) < 50 or model_use["Predicted_Risk"].nunique() < 3:
        return None

    model_use = model_use.copy()
    model_use["Risk_Bucket"] = pd.qcut(
        model_use["Predicted_Risk"],
        3,
        labels=False,
        duplicates="drop",
    )
    rates = model_use.groupby("Risk_Bucket", observed=True)[target_col].mean()
    counts = model_use.groupby("Risk_Bucket", observed=True)[target_col].count()
    if len(rates) < 2:
        return None

    low_bucket = rates.index.min()
    high_bucket = rates.index.max()
    return {
        "low_rate": float(rates.loc[low_bucket]),
        "high_rate": float(rates.loc[high_bucket]),
        "separation": float(rates.loc[high_bucket] - rates.loc[low_bucket]),
        "low_n": int(counts.loc[low_bucket]),
        "high_n": int(counts.loc[high_bucket]),
        "n": int(len(model_use)),
    }


def _profile_prediction_target(profile):
    if profile == "Controlled Chaos":
        return "Future_Late_xGA_Deviation_H1", "Future late-xGA deviation, next game"
    return "Future_Team_Relative_Collapse_H1", "Above team recent baseline, next game"


def _build_profile_weight_optimizer(proof_df):
    if "Volatility_Profile" not in proof_df.columns:
        return pd.DataFrame()

    base_features = ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3"]
    rows = []

    proof_df = proof_df.sort_values("Date").copy()
    split_date = proof_df["Date"].quantile(0.70)
    train_df = proof_df[proof_df["Date"] <= split_date]
    test_df = proof_df[proof_df["Date"] > split_date]

    for profile, profile_test_df in test_df.groupby("Volatility_Profile"):
        if not profile or profile == "Unknown":
            continue

        profile_train_df = train_df[train_df["Volatility_Profile"] == profile].copy()
        profile_test_df = profile_test_df.copy()
        if profile_train_df.empty or profile_test_df.empty:
            continue
        target_col, _ = _profile_prediction_target(profile)

        best = _best_profile_weights(
            profile_train_df,
            profile_test_df,
            target_col,
            base_features,
        )
        if not best:
            continue

        rows.append({
            "Volatility_Profile": profile,
            "Sovereignty_Weight": best["sovereignty_weight"],
            "Relief_Weight": best["relief_weight"],
            "Pressure_Weight": best["pressure_weight"],
            "Base_Separation": round(best["base_separation"], 3),
            "Optimized_SLC_Separation": round(best["optimized_separation"], 3),
            "Lift_vs_Load_Baseline": round(best["lift"], 3),
            "Final_Search_Step": best["final_step"],
            "Low_N": best["low_n"],
            "High_N": best["high_n"],
            "N": best["n"],
        })

    return pd.DataFrame(rows)


def _weight_candidates(center=None, step=0.1, radius=None):
    values = [round(i * step, 4) for i in range(int(round(1 / step)) + 1)]
    candidates = []
    for sovereignty_weight in values:
        for relief_weight in values:
            pressure_weight = round(1.0 - sovereignty_weight - relief_weight, 4)
            if pressure_weight < -0.0001 or pressure_weight > 1.0001:
                continue
            pressure_weight = round(max(0.0, min(1.0, pressure_weight)), 4)
            if center and radius is not None:
                if abs(sovereignty_weight - center["sovereignty_weight"]) > radius:
                    continue
                if abs(relief_weight - center["relief_weight"]) > radius:
                    continue
                if abs(pressure_weight - center["pressure_weight"]) > radius:
                    continue
            candidates.append((sovereignty_weight, relief_weight, pressure_weight))
    return candidates


def _weighted_slc_series(df, sovereignty_weight, relief_weight, pressure_weight):
    return (
        (df["Sovereignty_Rate_Roll3"] * sovereignty_weight) +
        (df["Relief_Rate_Roll3"] * relief_weight) -
        (df["Pressure_Cost_Rate_Roll3"] * pressure_weight)
    )


def _best_profile_weights(tune_train_df, tune_test_df, target_col, base_features):
    base_result = _risk_separation_for_features(
        tune_train_df,
        tune_test_df,
        target_col,
        base_features,
    )
    if not base_result:
        return None

    best = None
    search_plan = [
        {"step": 0.10, "radius": None},
        {"step": 0.05, "radius": 0.10},
        {"step": 0.01, "radius": 0.05},
    ]

    for search in search_plan:
        pass_best = None
        for sovereignty_weight, relief_weight, pressure_weight in _weight_candidates(
            center=best,
            step=search["step"],
            radius=search["radius"],
        ):
            train_use = tune_train_df.copy()
            test_use = tune_test_df.copy()
            train_use["Optimized_SLC_Roll3"] = _weighted_slc_series(
                train_use,
                sovereignty_weight,
                relief_weight,
                pressure_weight,
            )
            test_use["Optimized_SLC_Roll3"] = _weighted_slc_series(
                test_use,
                sovereignty_weight,
                relief_weight,
                pressure_weight,
            )
            result = _risk_separation_for_features(
                train_use,
                test_use,
                target_col,
                base_features + ["Optimized_SLC_Roll3"],
            )
            if not result:
                continue

            lift = result["separation"] - base_result["separation"]
            candidate = {
                "sovereignty_weight": sovereignty_weight,
                "relief_weight": relief_weight,
                "pressure_weight": pressure_weight,
                "base_separation": base_result["separation"],
                "optimized_separation": result["separation"],
                "lift": lift,
                "final_step": search["step"],
                "low_n": result["low_n"],
                "high_n": result["high_n"],
                "n": result["n"],
            }
            if (
                pass_best is None or
                candidate["lift"] > pass_best["lift"] or
                (
                    candidate["lift"] == pass_best["lift"] and
                    candidate["optimized_separation"] > pass_best["optimized_separation"]
                )
            ):
                pass_best = candidate

        if pass_best:
            best = pass_best

    return best


def _build_profile_weight_validation(proof_df):
    if "Volatility_Profile" not in proof_df.columns:
        return pd.DataFrame()

    base_features = ["Rest_Days", "Games_Last_7", "Late_xGA_Roll5", "Sequences_Roll3"]
    rows = []

    proof_df = proof_df.sort_values("Date").copy()
    outer_split = proof_df["Date"].quantile(0.70)
    train_df = proof_df[proof_df["Date"] <= outer_split]
    holdout_df = proof_df[proof_df["Date"] > outer_split]

    for profile, profile_holdout_df in holdout_df.groupby("Volatility_Profile"):
        if not profile or profile == "Unknown":
            continue

        profile_train_df = train_df[train_df["Volatility_Profile"] == profile].sort_values("Date").copy()
        profile_holdout_df = profile_holdout_df.sort_values("Date").copy()
        if len(profile_train_df) < 100 or len(profile_holdout_df) < 50:
            continue
        target_col, _ = _profile_prediction_target(profile)

        inner_split = profile_train_df["Date"].quantile(0.70)
        tune_train_df = profile_train_df[profile_train_df["Date"] <= inner_split]
        tune_test_df = profile_train_df[profile_train_df["Date"] > inner_split]
        if len(tune_train_df) < 50 or len(tune_test_df) < 50:
            continue

        best = _best_profile_weights(
            tune_train_df,
            tune_test_df,
            target_col,
            base_features,
        )
        if not best:
            continue

        base_holdout = _risk_separation_for_features(
            profile_train_df,
            profile_holdout_df,
            target_col,
            base_features,
        )
        if not base_holdout:
            continue

        train_use = profile_train_df.copy()
        holdout_use = profile_holdout_df.copy()
        train_use["Validated_SLC_Roll3"] = _weighted_slc_series(
            train_use,
            best["sovereignty_weight"],
            best["relief_weight"],
            best["pressure_weight"],
        )
        holdout_use["Validated_SLC_Roll3"] = _weighted_slc_series(
            holdout_use,
            best["sovereignty_weight"],
            best["relief_weight"],
            best["pressure_weight"],
        )
        holdout_result = _risk_separation_for_features(
            train_use,
            holdout_use,
            target_col,
            base_features + ["Validated_SLC_Roll3"],
        )
        if not holdout_result:
            continue

        rows.append({
            "Volatility_Profile": profile,
            "Sovereignty_Weight": best["sovereignty_weight"],
            "Relief_Weight": best["relief_weight"],
            "Pressure_Weight": best["pressure_weight"],
            "Tuning_Base_Separation": round(best["base_separation"], 3),
            "Tuning_SLC_Separation": round(best["optimized_separation"], 3),
            "Tuning_Lift": round(best["lift"], 3),
            "Holdout_Base_Separation": round(base_holdout["separation"], 3),
            "Holdout_SLC_Separation": round(holdout_result["separation"], 3),
            "Holdout_Lift": round(holdout_result["separation"] - base_holdout["separation"], 3),
            "N": holdout_result["n"],
        })

    return pd.DataFrame(rows)



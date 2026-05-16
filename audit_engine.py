import pandas as pd

from utility import (
    _get_goalie_team_abbrev,
    _load_raw_game,
    calculate_slc_score,
    calculate_service_tax,
    current_period_score,
    current_report_score,
    get_available_teams,
    get_team_games,
    get_team_sequence_data,
    get_goalie_matchup_label,
    load_master_reports,
    report_team_profile,
)


PLAYOFF_CUTOFF = pd.Timestamp("2026-04-16")
LEADERBOARD_MIN_RANKING_SEQUENCES = 5


def _safe_div(numerator, denominator):
    return round(float(numerator) / float(denominator), 3) if denominator else 0.0


def _relief_capture_rate(stats):
    service, tax = calculate_service_tax(stats, include_baseline=False)
    return _safe_div(service, service + tax)


def _pressure_survival_score(stats):
    shots = stats.get("S_saves", 0) + stats.get("S_goals", 0)
    pressure_weight = float(stats.get("Pressure_Weight", 0) or shots)
    if pressure_weight <= 0:
        return 0.0
    return round((float(stats.get("SLC_event", 0)) / pressure_weight) * 10, 3)


def _relief_efficiency_score(stats):
    if all(key in stats for key in ["Survival_Event", "Relief_Event", "Pressure_Cost_Event"]):
        _, scored_stats = calculate_slc_score(stats, team_profile=report_team_profile(report))
        relief_rate = float(scored_stats.get("Relief_Rate", 0) or 0)
        pressure_cost_rate = float(scored_stats.get("Pressure_Cost_Rate", 0) or 0)
        return round((relief_rate - pressure_cost_rate) * 10, 3)

    sequences = stats.get("Sequences", 0)
    if not sequences:
        return 0.0

    service, tax = calculate_service_tax(stats, include_baseline=False)
    capture_rate = _safe_div(service, service + tax)
    net_load_rate = _safe_div(service - tax, sequences)
    return round(net_load_rate * capture_rate, 3)


def _grade_series(series):
    if series.empty:
        return series
    return (series.rank(pct=True, method="average") * 100).round(0)


def _slc_profile(survival_grade, relief_grade):
    if survival_grade >= 50 and relief_grade >= 50:
        return "Complete Stabilizer"
    if survival_grade >= 50 and relief_grade < 50:
        return "Pressure Survivor"
    if survival_grade < 50 and relief_grade >= 50:
        return "Tempo Preserver"
    return "System Strain"


def _battery_fit_grade(system_profile, survival_grade, relief_grade, stress_rate, stress_median):
    if system_profile == "Suppressive":
        fit = (relief_grade * 0.70) + (survival_grade * 0.30)
    elif system_profile == "Volatile":
        fit = (survival_grade * 0.60) + (relief_grade * 0.40)
    else:
        fit = (survival_grade * 0.50) + (relief_grade * 0.50)

    if stress_rate > stress_median and relief_grade < 50:
        stress_gap = min((stress_rate - stress_median) * 100, 10)
        relief_gap = min((50 - relief_grade) * 0.15, 7.5)
        fit -= stress_gap + relief_gap

    return round(max(0, min(100, fit)), 0)


def _assign_slc_grades(leaderboard, master_report, game_phase):
    if leaderboard.empty:
        return leaderboard

    graded = leaderboard.copy()
    graded["Pressure_Survival_Grade"] = _grade_series(graded["Pressure_Survival_per_Game"])
    net_load_grade = _grade_series(graded["Net_Load_per_Sequence"])
    capture_grade = _grade_series(graded["Relief_Capture_Rate"])
    graded["Pressure_Relief_Grade"] = ((net_load_grade + capture_grade) / 2).round(0)

    reference, baselines = _build_team_system_reference(game_phase)
    team_profiles = reference.set_index("Team").to_dict("index") if not reference.empty else {}
    stress_median = baselines.get("stress_median", 0)

    battery_grades = []
    teams = []
    system_profiles = []
    for _, row in graded.iterrows():
        team = _infer_goalie_primary_team(master_report.get(row["Goalie_Key"], {}), game_phase)
        profile = team_profiles.get(team, {})
        system_profile = profile.get("System Profile", "Unknown")
        stress_rate = float(profile.get("Stress Sequence Rate", stress_median) or stress_median)

        teams.append(team or "")
        system_profiles.append(system_profile)
        battery_grades.append(
            _battery_fit_grade(
                system_profile,
                float(row["Pressure_Survival_Grade"]),
                float(row["Pressure_Relief_Grade"]),
                stress_rate,
                stress_median,
            )
        )

    graded["Team"] = teams
    graded["Team_System_Profile"] = system_profiles
    graded["Battery_Fit_Grade"] = battery_grades
    graded["SLC_Grade"] = (
        (graded["Pressure_Survival_Grade"] * 0.35) +
        (graded["Pressure_Relief_Grade"] * 0.45) +
        (graded["Battery_Fit_Grade"] * 0.20)
    ).round(0)
    graded["SLC_Profile"] = graded.apply(
        lambda row: _slc_profile(row["Pressure_Survival_Grade"], row["Pressure_Relief_Grade"]),
        axis=1,
    )
    graded["SLC_Rank"] = graded["SLC_Grade"].rank(ascending=False, method="min").astype(int)
    return graded


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


def _report_date(game_id, report):
    return report.get("gameDate") or report.get("date") or "0000-00-00"


def _empty_goalie_totals(goalie_key, first_report):
    return {
        "Goalie_Key": goalie_key,
        "Goalie": goalie_key.replace("_", " ").title(),
        "ID": first_report.get("goalie_id"),
        "GP": 0,
        "Total_SLC": 0.0,
        "Ranking_GP": 0,
        "Ranking_Total_SLC": 0.0,
        "Pressure_Survival_Total": 0.0,
        "Relief_Efficiency_Total": 0.0,
        "Ranking_Sequences": 0,
        "Small_Sample_Games": 0,
        "Shots": 0,
        "Saves": 0,
        "Goals": 0,
        "Sequences": 0,
        "Service": 0.0,
        "Tax": 0.0,
        "xG": 0.0,
        "xS": 0.0,
        "NPW": 0,
        "NPW_SH": 0,
        "UA": 0,
        "UA_SH": 0,
        "RP": 0,
        "RP_SH": 0,
        "iGvA": 0,
        "iGvA_SH": 0,
        "iTkA": 0,
        "Survival_Event": 0.0,
        "Relief_Event": 0.0,
        "Pressure_Cost_Event": 0.0,
    }


def _add_report_to_totals(totals, report):
    stats = report.get("total", {}).get("stats", {})
    saves = stats.get("S_saves", 0)
    goals = stats.get("S_goals", 0)
    sequences = stats.get("Sequences", 0)
    slc = current_report_score(report)
    pressure_survival = _pressure_survival_score(stats)
    relief_efficiency = _relief_efficiency_score(stats)
    service, tax = calculate_service_tax(stats, include_baseline=True)

    totals["GP"] += 1
    totals["Total_SLC"] += slc
    totals["Pressure_Survival_Total"] += pressure_survival
    totals["Relief_Efficiency_Total"] += relief_efficiency
    if sequences >= LEADERBOARD_MIN_RANKING_SEQUENCES:
        totals["Ranking_GP"] += 1
        totals["Ranking_Total_SLC"] += slc
        totals["Ranking_Sequences"] += sequences
    else:
        totals["Small_Sample_Games"] += 1
    totals["Shots"] += saves + goals
    totals["Saves"] += saves
    totals["Goals"] += goals
    totals["Sequences"] += sequences
    totals["Service"] += service
    totals["Tax"] += tax
    totals["xG"] += stats.get("xG", 0)
    totals["xS"] += stats.get("xS", 0)
    totals["Survival_Event"] += stats.get("Survival_Event", stats.get("SLC_event", 0))
    totals["Relief_Event"] += stats.get("Relief_Event", 0)
    totals["Pressure_Cost_Event"] += stats.get("Pressure_Cost_Event", 0)

    for key in ["NPW", "NPW_SH", "UA", "UA_SH", "RP", "RP_SH", "iGvA", "iGvA_SH", "iTkA"]:
        totals[key] += stats.get(key, 0)


def _finalize_totals(totals):
    shots = totals["Shots"]
    sequences = totals["Sequences"]
    actual_sv = _safe_div(totals["Saves"], shots)
    expected_saves = totals["xS"] if totals["xS"] else shots - totals["xG"]
    expected_sv = _safe_div(expected_saves, shots)

    totals["Total_SLC"] = round(totals["Total_SLC"], 3)
    totals["Ranking_Total_SLC"] = round(totals["Ranking_Total_SLC"], 3)
    totals["Pressure_Survival_Total"] = round(totals["Pressure_Survival_Total"], 3)
    totals["Relief_Efficiency_Total"] = round(totals["Relief_Efficiency_Total"], 3)
    totals["Avg_SLC"] = _safe_div(totals["Total_SLC"], totals["GP"])
    totals["Pressure_Efficiency_per_Sequence"] = _safe_div(totals["Ranking_Total_SLC"], totals["Ranking_Sequences"])
    totals["Pressure_Survival_per_Game"] = _safe_div(totals["Pressure_Survival_Total"], totals["GP"])
    totals["Relief_Efficiency_per_Game"] = _safe_div(totals["Relief_Efficiency_Total"], totals["GP"])
    totals["Net_Load"] = round(totals["Service"] - totals["Tax"], 3)
    totals["Net_Load_per_Sequence"] = _safe_div(totals["Net_Load"], sequences)
    totals["Service_per_Sequence"] = _safe_div(totals["Service"], sequences)
    totals["Tax_per_Sequence"] = _safe_div(totals["Tax"], sequences)
    totals["Relief_Capture_Rate"] = _relief_capture_rate(totals)
    totals["Sequences_per_Game"] = _safe_div(sequences, totals["GP"])
    totals["Shots_per_Game"] = _safe_div(shots, totals["GP"])
    totals["Survival_Component_per_Game"] = _safe_div(totals["Survival_Event"], totals["GP"])
    totals["Relief_Component_per_Game"] = _safe_div(totals["Relief_Event"], totals["GP"])
    totals["Pressure_Cost_per_Game"] = _safe_div(totals["Pressure_Cost_Event"], totals["GP"])
    totals["Net_Component_per_Game"] = _safe_div(
        totals["Survival_Event"] + totals["Relief_Event"] - totals["Pressure_Cost_Event"],
        totals["GP"],
    )
    totals["SV%"] = round(actual_sv, 3) if shots else 0.0
    totals["Sovereignty"] = round((actual_sv - expected_sv) * 100, 3) if shots else 0.0
    totals["Reset_Ratio"] = _safe_div(totals["Saves"], sequences)
    return totals


def _build_goalie_game_log(goalie_key, games, game_phase):
    rows = []
    for game_id, report in games.items():
        date_value = _report_date(game_id, report)
        if not _matches_game_phase(date_value, game_phase, game_id):
            continue

        stats = report.get("total", {}).get("stats", {})
        saves = stats.get("S_saves", 0)
        goals = stats.get("S_goals", 0)
        shots = saves + goals
        sequences = stats.get("Sequences", 0)
        service, tax = calculate_service_tax(stats, include_baseline=True)
        _, scored_stats = calculate_slc_score(stats)
        slc = current_report_score(report)
        x_saves = stats.get("xS", 0) or shots - stats.get("xG", 0)
        actual_sv = _safe_div(saves, shots)
        expected_sv = _safe_div(x_saves, shots)
        net_load = round(service - tax, 3)

        rows.append({
            "Date": date_value,
            "Matchup": get_goalie_matchup_label(game_id, report.get("goalie_id")),
            "Game_ID": str(game_id),
            "SLC": slc,
            "Sequences": sequences,
            "Pressure_Efficiency_per_Sequence": _safe_div(slc, sequences),
            "Pressure_Survival": _pressure_survival_score(stats),
            "Relief_Efficiency": _relief_efficiency_score(stats),
            "Net_Load": net_load,
            "Net_Load_per_Sequence": _safe_div(net_load, sequences),
            "Service": service,
            "Service_per_Sequence": _safe_div(service, sequences),
            "Tax": tax,
            "Tax_per_Sequence": _safe_div(tax, sequences),
            "Relief_Capture_Rate": _relief_capture_rate(stats),
            "Survival_Component": stats.get("Survival_Event", stats.get("SLC_event", 0)),
            "Relief_Component": stats.get("Relief_Event", 0),
            "Pressure_Cost_Component": stats.get("Pressure_Cost_Event", 0),
            "Raw_SLC": scored_stats.get("Raw_SLC", slc),
            "SLC_Confidence": scored_stats.get("SLC_Confidence", 1.0),
            "Shots": shots,
            "Sovereignty": round((actual_sv - expected_sv) * 100, 3) if shots else 0.0,
            "NPW": stats.get("NPW", 0),
            "NPW_SH": stats.get("NPW_SH", 0),
            "UA": stats.get("UA", 0),
            "UA_SH": stats.get("UA_SH", 0),
            "RP": stats.get("RP", 0),
            "RP_SH": stats.get("RP_SH", 0),
            "iGvA": stats.get("iGvA", 0),
            "iGvA_SH": stats.get("iGvA_SH", 0),
        })

    if not rows:
        return pd.DataFrame()

    game_log = pd.DataFrame(rows)
    game_log["Date"] = pd.to_datetime(game_log["Date"], errors="coerce")
    # Calculate per-game grade percentiles for survival and relief, then combine
    if not game_log.empty:
        game_log["Game_Survival_Grade"] = _grade_series(game_log["Pressure_Survival"])
        game_log["Game_Relief_Grade"] = _grade_series(game_log["Relief_Efficiency"])
        game_log["SLC_Grade"] = ((game_log["Game_Survival_Grade"] + game_log["Game_Relief_Grade"]) / 2).round(0)
    return game_log.sort_values("Date")


def _infer_goalie_primary_team(games, game_phase):
    team_counts = {}
    for game_id, report in games.items():
        date_value = _report_date(game_id, report)
        if not _matches_game_phase(date_value, game_phase, game_id):
            continue

        raw_game = _load_raw_game(game_id)
        team_abbrev = _get_goalie_team_abbrev(raw_game, report.get("goalie_id"))
        if team_abbrev:
            team_counts[team_abbrev] = team_counts.get(team_abbrev, 0) + 1

    if not team_counts:
        return None
    return sorted(team_counts.items(), key=lambda item: item[1], reverse=True)[0][0]


def _label_team_system(avg_sequences, wall_rate, volume_median, stress_median):
    high_volume = avg_sequences >= volume_median
    high_stress = wall_rate >= stress_median

    if not high_volume and not high_stress:
        return "Suppressive"
    if high_volume and high_stress:
        return "Treadmill"
    if not high_volume and high_stress:
        return "Volatile"
    return "Managed Chaos"


def _team_system_need(system_label):
    needs = {
        "Suppressive": "Needs clean handling and fast terminations. The goalie should not turn rare pressure into extra work.",
        "Treadmill": "Needs active relief. The goalie has to end sequences before the team's legs start paying interest.",
        "Volatile": "Needs damage control. The workload may not be constant, but broken sequences become expensive quickly.",
        "Stress-Heavy": "Needs clock kills and rebound control. Too many shifts are reaching the danger wall.",
        "Managed Chaos": "Needs steady resets. The system allows activity, so the goalie has to keep it from compounding.",
    }
    return needs.get(system_label, "Needs steady reset work that matches the team's defensive environment.")


def _build_team_fit_read(goalie_row, profile_row):
    system_label = profile_row["System Profile"]
    net_load = goalie_row["Net_Load_per_Sequence"]
    tax = goalie_row["Tax_per_Sequence"]
    wall_rate = profile_row["Stress Sequence Rate"]

    if system_label == "Suppressive" and tax > 0.50:
        return (
            f"{goalie_row['Goalie']} is playing behind a suppressive system, so the ask is precision. "
            "Their Tax rate is elevated for that environment, which means the system can absorb it only while the team has margin."
        )
    if system_label in {"Treadmill", "Stress-Heavy"} and net_load >= 0.55:
        return (
            f"{goalie_row['Goalie']} is giving meaningful relief inside a high-stress environment. "
            "That matters because this team sees enough extended shifts for goalie management to become a fatigue lever."
        )
    if wall_rate > 0.20 and net_load < 0.45:
        return (
            f"{goalie_row['Goalie']} is not creating enough relief for a system with frequent wall pressure. "
            "The saves may still matter, but the workload is likely to compound."
        )
    if net_load >= 0.55:
        return (
            f"{goalie_row['Goalie']} fits this environment well from a load-management view. "
            "Service is beating Tax strongly enough to help preserve the skaters."
        )
    return (
        f"{goalie_row['Goalie']} is giving a mixed fit signal. "
        "The team context explains what kind of help is needed; compare that need against the goalie rates below."
    )


def _team_sequence_profile_row(team_abbrev, game_phase):
    frames = []
    game_count = 0
    for game in get_team_games(team_abbrev):
        if not _matches_game_phase(game.get("Date"), game_phase, game.get("Game_ID")):
            continue

        sequence_df = get_team_sequence_data(team_abbrev, game["Game_ID"])
        if sequence_df.empty:
            continue

        frame = sequence_df.copy()
        frame["Game_ID"] = game["Game_ID"]
        frames.append(frame)
        game_count += 1

    if not frames:
        return None

    all_sequences = pd.concat(frames, ignore_index=True)
    total_sequences = len(all_sequences)
    wall_breaches = int((all_sequences["Duration"] >= 40).sum())
    under_40 = int((all_sequences["Duration"] < 40).sum())
    stress_sequences = int(all_sequences["Stress"].sum()) if "Stress" in all_sequences else wall_breaches
    avg_sequences = _safe_div(total_sequences, game_count)
    wall_rate = _safe_div(wall_breaches, total_sequences)
    wall_breaches_per_game = _safe_div(wall_breaches, game_count)

    return {
        "Team": team_abbrev,
        "Games": game_count,
        "Avg Defensive Sequences": avg_sequences,
        "Avg Sequence Duration": round(float(all_sequences["Duration"].mean()), 1),
        "Avg 40s Wall Breaches": wall_breaches_per_game,
        "40s Wall Efficiency": round((under_40 / total_sequences) * 100, 1) if total_sequences else 0.0,
        "Stress Sequence Rate": wall_rate,
        "Stress Sequences": stress_sequences,
        "Total Sequences": total_sequences,
    }


def _build_team_system_reference(game_phase):
    rows = []
    for team_abbrev in get_available_teams():
        row = _team_sequence_profile_row(team_abbrev, game_phase)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame(), {"volume_median": 0.0, "stress_median": 0.0}

    reference = pd.DataFrame(rows)
    volume_median = round(float(reference["Avg Defensive Sequences"].median()), 3)
    stress_median = round(float(reference["Stress Sequence Rate"].median()), 3)
    reference["System Profile"] = reference.apply(
        lambda row: _label_team_system(
            row["Avg Defensive Sequences"],
            row["Stress Sequence Rate"],
            volume_median,
            stress_median,
        ),
        axis=1,
    )
    reference["System Need"] = reference["System Profile"].apply(_team_system_need)
    return reference, {
        "volume_median": volume_median,
        "stress_median": stress_median,
    }


def _build_team_system_profile(goalie_key, games, game_phase, goalie_row):
    team_abbrev = _infer_goalie_primary_team(games, game_phase)
    if not team_abbrev:
        return pd.DataFrame(), ""

    reference, baselines = _build_team_system_reference(game_phase)
    if reference.empty:
        return pd.DataFrame(), ""

    profile = reference[reference["Team"] == team_abbrev].copy()
    if profile.empty:
        return pd.DataFrame(), ""

    profile["League Median Sequences"] = baselines["volume_median"]
    profile["League Median Stress Rate"] = baselines["stress_median"]
    display_columns = [
        "Team",
        "System Profile",
        "Games",
        "Avg Defensive Sequences",
        "League Median Sequences",
        "Avg Sequence Duration",
        "Avg 40s Wall Breaches",
        "40s Wall Efficiency",
        "Stress Sequence Rate",
        "League Median Stress Rate",
        "System Need",
    ]
    profile = profile[display_columns]
    return profile, _build_team_fit_read(goalie_row, profile.iloc[0].to_dict())


def _build_phase_leaderboard(master_report, game_phase):
    rows = []
    for goalie_key, games in master_report.items():
        if not games:
            continue

        first_report = next(iter(games.values()))
        totals = _empty_goalie_totals(goalie_key, first_report)
        for game_id, report in games.items():
            date_value = _report_date(game_id, report)
            if _matches_game_phase(date_value, game_phase, game_id):
                _add_report_to_totals(totals, report)

        if totals["GP"] > 0:
            rows.append(_finalize_totals(totals))

    if not rows:
        return pd.DataFrame()

    leaderboard = pd.DataFrame(rows)
    leaderboard["Pressure_Efficiency_Rank"] = leaderboard["Pressure_Efficiency_per_Sequence"].rank(
        ascending=False,
        method="min"
    ).astype(int)
    leaderboard["Efficiency_Percentile"] = (
        leaderboard["Pressure_Efficiency_per_Sequence"].rank(pct=True) * 100
    ).round(1)
    leaderboard = _assign_slc_grades(leaderboard, master_report, game_phase)
    return leaderboard.sort_values("SLC_Grade", ascending=False)


def _context_min_gp(game_phase):
    return 20 if game_phase == "Regular Season" else 1


def _qualified_context_from_min_gp(leaderboard, min_gp):
    qualified = leaderboard[leaderboard["GP"] >= min_gp].copy()
    if qualified.empty:
        qualified = leaderboard.copy()
        min_gp = 1

    return {
        "min_gp": min_gp,
        "goalies": int(len(qualified)),
        "pressure_efficiency_per_sequence_median": round(float(qualified["Pressure_Efficiency_per_Sequence"].median()), 3),
        "pressure_efficiency_per_sequence_mean": round(float(qualified["Pressure_Efficiency_per_Sequence"].mean()), 3),
        "net_load_per_sequence_median": round(float(qualified["Net_Load_per_Sequence"].median()), 3),
        "service_per_sequence_median": round(float(qualified["Service_per_Sequence"].median()), 3),
        "tax_per_sequence_median": round(float(qualified["Tax_per_Sequence"].median()), 3),
        "relief_capture_rate_median": round(float(qualified["Relief_Capture_Rate"].median()), 3),
        "relief_capture_rate_mean": round(float(qualified["Relief_Capture_Rate"].mean()), 3),
        "slc_grade_median": round(float(qualified["SLC_Grade"].median()), 0),
        "pressure_survival_grade_median": round(float(qualified["Pressure_Survival_Grade"].median()), 0),
        "pressure_relief_grade_median": round(float(qualified["Pressure_Relief_Grade"].median()), 0),
        "battery_fit_grade_median": round(float(qualified["Battery_Fit_Grade"].median()), 0),
        "sequences_per_game_median": round(float(qualified["Sequences_per_Game"].median()), 3),
        "sequences_per_game_mean": round(float(qualified["Sequences_per_Game"].mean()), 3),
    }


def _qualified_context(leaderboard, game_phase):
    return _qualified_context_from_min_gp(leaderboard, _context_min_gp(game_phase))


def build_league_leaderboard(game_phase, min_gp=None):
    master_report = load_master_reports()
    leaderboard = _build_phase_leaderboard(master_report, game_phase)
    if leaderboard.empty:
        return {
            "leaderboard": leaderboard,
            "qualified": leaderboard,
            "context": {},
        }

    min_gp = _context_min_gp(game_phase) if min_gp is None else int(min_gp)
    qualified = leaderboard[leaderboard["GP"] >= min_gp].copy()
    if qualified.empty:
        qualified = leaderboard.copy()

    qualified = qualified.sort_values("SLC_Grade", ascending=False).copy()
    qualified["Rank"] = range(1, len(qualified) + 1)

    return {
        "leaderboard": leaderboard,
        "qualified": qualified,
        "context": _qualified_context_from_min_gp(leaderboard, min_gp),
    }


def _goalie_games_for_team(games, team_abbrev, game_phase):
    filtered = {}
    for game_id, report in games.items():
        date_value = _report_date(game_id, report)
        if not _matches_game_phase(date_value, game_phase, game_id):
            continue

        raw_game = _load_raw_game(game_id)
        goalie_team = _get_goalie_team_abbrev(raw_game, report.get("goalie_id"))
        if goalie_team == team_abbrev:
            filtered[game_id] = report
    return filtered


def _team_game_stress_map(team_abbrev, game_phase):
    rows = []
    for game in get_team_games(team_abbrev):
        if not _matches_game_phase(game.get("Date"), game_phase, game.get("Game_ID")):
            continue

        sequence_df = get_team_sequence_data(team_abbrev, game["Game_ID"])
        if sequence_df.empty:
            continue

        wall_breaches = int((sequence_df["Duration"] >= 40).sum())
        rows.append({
            "Game_ID": str(game["Game_ID"]),
            "Team_Sequences": int(len(sequence_df)),
            "Team_Red_Line_Shifts": wall_breaches,
            "Team_Avg_Sequence_Duration": round(float(sequence_df["Duration"].mean()), 1),
        })

    env = pd.DataFrame(rows)
    if env.empty:
        return {}, {
            "team_median_sequences": 0.0,
            "team_median_shots": 0.0,
            "team_median_red_line": 0.0,
        }

    return env.set_index("Game_ID").to_dict("index"), {
        "team_median_sequences": round(float(env["Team_Sequences"].median()), 3),
        "team_median_red_line": round(float(env["Team_Red_Line_Shifts"].median()), 3),
    }


def _confidence_label(score):
    if score >= 75:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def _recommend_workload(row, starter_row):
    if row["GP"] < 3:
        return "Too Little Data"

    if row["Goalie"] == starter_row["Goalie"]:
        if row["SLC_Grade"] >= 55:
            return "Keep Starter"
        return "Starter At Risk"

    grade_edge = row["SLC_Grade"] - starter_row["SLC_Grade"]
    high_workload_edge = row["High_Workload_SLC_Grade"] - starter_row["High_Workload_SLC_Grade"]
    bad_start_gap = row["Bad_Start_Rate"] - starter_row["Bad_Start_Rate"]

    if grade_edge >= 10 and high_workload_edge >= 0 and bad_start_gap <= 0.15 and row["Role_Confidence"] >= 45:
        return "Increase Workload"
    if grade_edge >= 10 and row["Role_Confidence"] < 45:
        return "Test Larger Role"
    if grade_edge >= 10:
        return "Protected Expansion"
    if row["SLC_Grade"] >= 60 and row["Role_Confidence"] >= 45:
        return "Useful Tandem Role"
    return "Hold Role"


def _recommendation_note(row, starter_row):
    if row["Recommendation"] == "Increase Workload":
        return (
            f"{row['Goalie']} has a {row['SLC_Grade'] - starter_row['SLC_Grade']:+.0f} SLC Grade edge over "
            f"{starter_row['Goalie']} and the edge survives starter-like workload."
        )
    if row["Recommendation"] == "Test Larger Role":
        return (
            f"{row['Goalie']} has the efficiency edge, but the workload evidence is still light. "
            "Increase usage before making a full starter call."
        )
    if row["Recommendation"] == "Protected Expansion":
        return (
            f"{row['Goalie']} has the efficiency edge, but the stress-test profile is not clean enough for an immediate role flip."
        )
    if row["Recommendation"] == "Starter At Risk":
        return (
            f"{row['Goalie']} owns the workload, but the SLC Grade is below the level expected from a secure starter."
        )
    if row["Recommendation"] == "Keep Starter":
        return (
            f"{row['Goalie']} carries the largest role and grades strongly enough to keep the starter workload."
        )
    if row["Recommendation"] == "Useful Tandem Role":
        return (
            f"{row['Goalie']} grades well enough to protect the starter and absorb meaningful starts."
        )
    if row["Recommendation"] == "Too Little Data":
        return "Not enough team-specific starts to make a workload recommendation."
    return "Current role is supported by the available SLC evidence."


def build_team_goalie_decision(team_abbrev, game_phase):
    master_report = load_master_reports()
    leaderboard = _build_phase_leaderboard(master_report, game_phase)
    if leaderboard.empty:
        return {
            "summary": pd.DataFrame(),
            "games": pd.DataFrame(),
            "context": {},
        }

    team_board = leaderboard[leaderboard["Team"] == team_abbrev].copy()
    if team_board.empty:
        return {
            "summary": pd.DataFrame(),
            "games": pd.DataFrame(),
            "context": {},
        }

    stress_map, context = _team_game_stress_map(team_abbrev, game_phase)
    game_frames = []
    for _, goalie_row in team_board.iterrows():
        team_games = _goalie_games_for_team(
            master_report.get(goalie_row["Goalie_Key"], {}),
            team_abbrev,
            game_phase,
        )
        game_log = _build_goalie_game_log(goalie_row["Goalie_Key"], team_games, game_phase)
        if game_log.empty:
            continue

        game_log = game_log.copy()
        game_log["Goalie"] = goalie_row["Goalie"]
        for col in ["Team_Sequences", "Team_Red_Line_Shifts", "Team_Avg_Sequence_Duration"]:
            game_log[col] = game_log["Game_ID"].map(lambda game_id: stress_map.get(str(game_id), {}).get(col, 0))
        game_frames.append(game_log)

    if not game_frames:
        return {
            "summary": pd.DataFrame(),
            "games": pd.DataFrame(),
            "context": context,
        }

    games = pd.concat(game_frames, ignore_index=True)
    context["team_median_shots"] = round(float(games["Shots"].median()), 3)
    context["team_median_goalie_sequences"] = round(float(games["Sequences"].median()), 3)
    games["High_Workload_Start"] = (
        (games["Sequences"] >= context["team_median_goalie_sequences"]) |
        (games["Shots"] >= context["team_median_shots"]) |
        (games["Team_Red_Line_Shifts"] >= context["team_median_red_line"])
    )

    games["Game_Survival_Grade"] = _grade_series(games["Pressure_Survival"])
    games["Game_Relief_Grade"] = _grade_series(games["Relief_Efficiency"])
    games["Game_SLC_Grade"] = ((games["Game_Survival_Grade"] + games["Game_Relief_Grade"]) / 2).round(0)

    summary_rows = []
    for _, goalie_row in team_board.iterrows():
        goalie_games = games[games["Goalie"] == goalie_row["Goalie"]].copy()
        if goalie_games.empty:
            continue

        high_games = goalie_games[goalie_games["High_Workload_Start"]]
        game_grade_std = float(goalie_games["Game_SLC_Grade"].std()) if len(goalie_games) > 1 else 40.0
        stability = round(max(0, min(100, 100 - (game_grade_std * 2))), 0)
        high_workload_grade = round(float(high_games["Game_SLC_Grade"].mean()), 0) if not high_games.empty else 0.0
        high_workload_starts = int(len(high_games))
        bad_start_rate = _safe_div(int((goalie_games["Game_SLC_Grade"] < 40).sum()), len(goalie_games))
        above_median_rate = _safe_div(int((goalie_games["Game_SLC_Grade"] >= 50).sum()), len(goalie_games))
        gp_score = min((len(goalie_games) / 40) * 100, 100)
        high_workload_score = min((high_workload_starts / 15) * 100, 100)
        role_confidence = round((gp_score * 0.40) + (high_workload_score * 0.40) + (stability * 0.20), 0)

        summary_rows.append({
            "Goalie": goalie_row["Goalie"],
            "GP": int(len(goalie_games)),
            "SLC_Grade": goalie_row["SLC_Grade"],
            "SLC_Profile": goalie_row["SLC_Profile"],
            "Pressure_Survival": goalie_row["Pressure_Survival_Grade"],
            "Pressure_Relief": goalie_row["Pressure_Relief_Grade"],
            "Battery_Fit": goalie_row["Battery_Fit_Grade"],
            "High_Workload_Starts": high_workload_starts,
            "High_Workload_SLC_Grade": high_workload_grade,
            "Above_Median_Start_Rate": above_median_rate,
            "Bad_Start_Rate": bad_start_rate,
            "Stability_Grade": stability,
            "Role_Confidence": role_confidence,
            "Confidence": _confidence_label(role_confidence),
        })

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return {
            "summary": summary,
            "games": games,
            "context": context,
        }

    starter = summary.sort_values(["GP", "SLC_Grade"], ascending=False).iloc[0].to_dict()
    summary["Recommendation"] = summary.apply(lambda row: _recommend_workload(row, starter), axis=1)
    summary["Decision_Note"] = summary.apply(lambda row: _recommendation_note(row, starter), axis=1)
    summary = summary.sort_values(["Recommendation", "SLC_Grade"], ascending=[True, False])

    context["starter"] = starter["Goalie"]
    context["team"] = team_abbrev
    return {
        "summary": summary,
        "games": games.sort_values(["Date", "Goalie"], ascending=[False, True]),
        "context": context,
    }


def _build_verdict(goalie_row, leaderboard, game_phase):
    context = _qualified_context(leaderboard, game_phase)
    league_pressure_efficiency_seq = context["pressure_efficiency_per_sequence_median"]
    league_seq_game = context["sequences_per_game_median"]
    league_net_load_seq = context["net_load_per_sequence_median"]
    league_capture_rate = context["relief_capture_rate_median"]
    efficiency_delta = float(goalie_row["Pressure_Efficiency_per_Sequence"] - league_pressure_efficiency_seq)
    net_load_delta = float(goalie_row["Net_Load_per_Sequence"] - league_net_load_seq)
    capture_delta = float(goalie_row["Relief_Capture_Rate"] - league_capture_rate)
    low_volume = goalie_row["Sequences_per_Game"] < league_seq_game
    efficient = efficiency_delta >= 0
    strong_relief = net_load_delta >= 0
    grade = int(goalie_row["SLC_Grade"])
    profile = goalie_row["SLC_Profile"]
    team_profile = goalie_row.get("Team_System_Profile", "Unknown")

    verdict = (
        f"SLC grades {goalie_row['Goalie']} as a {grade}/100 {profile}. "
        f"The profile is built from pressure survival ({goalie_row['Pressure_Survival_Grade']:.0f}), "
        f"pressure relief ({goalie_row['Pressure_Relief_Grade']:.0f}), and battery fit "
        f"({goalie_row['Battery_Fit_Grade']:.0f}) inside a {team_profile} team context."
    )

    return {
        "text": verdict,
        "slc_grade": grade,
        "slc_profile": profile,
        "league_pressure_efficiency_per_sequence": round(league_pressure_efficiency_seq, 3),
        "league_sequences_per_game": round(league_seq_game, 3),
        "league_net_load_per_sequence": round(league_net_load_seq, 3),
        "league_relief_capture_rate": round(league_capture_rate, 3),
        "league_pressure_efficiency_per_sequence_mean": context["pressure_efficiency_per_sequence_mean"],
        "league_sequences_per_game_mean": context["sequences_per_game_mean"],
        "context_min_gp": context["min_gp"],
        "context_goalies": context["goalies"],
        "efficiency_delta": round(efficiency_delta, 3),
        "net_load_delta": round(net_load_delta, 3),
        "relief_capture_delta": round(capture_delta, 3),
        "low_volume": bool(low_volume),
        "efficient": bool(efficient),
        "strong_relief": bool(strong_relief),
    }


def _build_evidence_rows(goalie_row):
    return pd.DataFrame([
        {
            "Evidence": "SLC Grade",
            "Value": goalie_row["SLC_Grade"],
            "What it means": "Overall goalie efficiency grade: survival, relief, and battery fit",
        },
        {
            "Evidence": "Pressure Survival Grade",
            "Value": goalie_row["Pressure_Survival_Grade"],
            "What it means": "How well the goalie handled dangerous pressure",
        },
        {
            "Evidence": "Pressure Relief Grade",
            "Value": goalie_row["Pressure_Relief_Grade"],
            "What it means": "How well the goalie converted pressure moments into relief",
        },
        {
            "Evidence": "Battery Fit Grade",
            "Value": goalie_row["Battery_Fit_Grade"],
            "What it means": "How well the goalie profile fits the team's pressure environment",
        },
        {
            "Evidence": "Pressure Efficiency",
            "Value": goalie_row["Pressure_Efficiency_per_Sequence"],
            "What it means": "How well the goalie handles each pressure event in isolation.",
        },
        {
            "Evidence": "Net Load / Sequence",
            "Value": goalie_row["Net_Load_per_Sequence"],
            "What it means": "Service minus Tax, normalized for workload",
        },
        {
            "Evidence": "Relief Capture Rate",
            "Value": goalie_row["Relief_Capture_Rate"],
            "What it means": "Share of Service/Tax opportunities resolved as Service",
        },
        {
            "Evidence": "Service / Sequence",
            "Value": goalie_row["Service_per_Sequence"],
            "What it means": "Clean relief actions per sequence",
        },
        {
            "Evidence": "Tax / Sequence",
            "Value": goalie_row["Tax_per_Sequence"],
            "What it means": "Unresolved pressure per sequence",
        },
        {
            "Evidence": "Sequences / Game",
            "Value": goalie_row["Sequences_per_Game"],
            "What it means": "Workload context, not the ranking itself",
        },
        {
            "Evidence": "Total Pressure Efficiency",
            "Value": goalie_row["Total_SLC"],
            "What it means": "Accumulated season value",
        },
    ])


def _build_context_rows(goalie_row, context):
    return pd.DataFrame([
        {
            "Lens": "Efficiency snapshot",
            "Goalie": goalie_row["SLC_Grade"],
            "Qualified median": context["slc_grade_median"],
            "Qualified mean": None,
            "Read": "SLC Grade combines pressure survival, pressure relief, and battery fit.",
        },
        {
            "Lens": "Pressure survival",
            "Goalie": goalie_row["Pressure_Survival_Grade"],
            "Qualified median": context["pressure_survival_grade_median"],
            "Qualified mean": None,
            "Read": "Did the goalie handle dangerous moments?",
        },
        {
            "Lens": "Pressure relief",
            "Goalie": goalie_row["Pressure_Relief_Grade"],
            "Qualified median": context["pressure_relief_grade_median"],
            "Qualified mean": None,
            "Read": "Did the goalie convert pressure into relief for the skaters?",
        },
        {
            "Lens": "Battery fit",
            "Goalie": goalie_row["Battery_Fit_Grade"],
            "Qualified median": context["battery_fit_grade_median"],
            "Qualified mean": None,
            "Read": "Does the goalie profile match the team's pressure environment?",
        },
        {
            "Lens": "Workload context",
            "Goalie": goalie_row["Sequences_per_Game"],
            "Qualified median": context["sequences_per_game_median"],
            "Qualified mean": context["sequences_per_game_mean"],
            "Read": "Lower workload should not hurt the goalie; it tells you how much pressure they actually faced.",
        },
        {
            "Lens": "Net relief",
            "Goalie": goalie_row["Net_Load_per_Sequence"],
            "Qualified median": context["net_load_per_sequence_median"],
            "Qualified mean": None,
            "Read": "Higher than median means stronger relief for the skaters after normalizing for workload.",
        },
        {
            "Lens": "Relief capture",
            "Goalie": goalie_row["Relief_Capture_Rate"],
            "Qualified median": context["relief_capture_rate_median"],
            "Qualified mean": context["relief_capture_rate_mean"],
            "Read": "Opportunity-based read: when Service or Tax was available, how often did it become relief?",
        },
        {
            "Lens": "Service",
            "Goalie": goalie_row["Service_per_Sequence"],
            "Qualified median": context["service_per_sequence_median"],
            "Qualified mean": None,
            "Read": "Clean relief actions per pressure sequence.",
        },
        {
            "Lens": "Tax",
            "Goalie": goalie_row["Tax_per_Sequence"],
            "Qualified median": context["tax_per_sequence_median"],
            "Qualified mean": None,
            "Read": "Unresolved pressure per sequence; lower is better.",
        },
    ])


def _build_translation(goalie_row, context):
    efficiency_delta = round(goalie_row["Pressure_Efficiency_per_Sequence"] - context["pressure_efficiency_per_sequence_median"], 3)
    volume_delta = round(goalie_row["Sequences_per_Game"] - context["sequences_per_game_median"], 3)
    net_delta = round(goalie_row["Net_Load_per_Sequence"] - context["net_load_per_sequence_median"], 3)
    capture_delta = round(goalie_row["Relief_Capture_Rate"] - context["relief_capture_rate_median"], 3)

    efficiency_read = "above" if efficiency_delta >= 0 else "below"
    volume_read = "higher" if volume_delta >= 0 else "lower"
    net_read = "above" if net_delta >= 0 else "below"
    capture_read = "above" if capture_delta >= 0 else "below"
    mean_read = "above" if goalie_row["Pressure_Efficiency_per_Sequence"] >= context["pressure_efficiency_per_sequence_mean"] else "below"

    return pd.DataFrame([
        {
            "Question": "Is this volume-built?",
            "Answer": (
                f"{goalie_row['Sequences_per_Game']:.3f} sequences/game is {abs(volume_delta):.3f} "
                f"{volume_read} than the qualified median."
            ),
        },
        {
            "Question": "Is the efficiency real?",
            "Answer": (
                f"{goalie_row['Pressure_Efficiency_per_Sequence']:.3f} Pressure Efficiency is {abs(efficiency_delta):.3f} "
                f"{efficiency_read} the qualified median."
            ),
        },
        {
            "Question": "Is this elite-level?",
            "Answer": (
                f"The qualified mean is {context['pressure_efficiency_per_sequence_mean']:.3f}; this goalie is {mean_read} "
                "that top-end-pulled benchmark."
            ),
        },
        {
            "Question": "Is Service beating Tax?",
            "Answer": (
                f"{goalie_row['Net_Load_per_Sequence']:.3f} net load/sequence is {abs(net_delta):.3f} "
                f"{net_read} the qualified median."
            ),
        },
        {
            "Question": "Are relief chances being captured?",
            "Answer": (
                f"{goalie_row['Relief_Capture_Rate']:.3f} capture rate is {abs(capture_delta):.3f} "
                f"{capture_read} the qualified median."
            ),
        },
    ])


def build_goalie_audit(goalie_key, game_phase):
    master_report = load_master_reports()
    leaderboard = _build_phase_leaderboard(master_report, game_phase)
    if leaderboard.empty or goalie_key.lower() not in set(leaderboard["Goalie_Key"]):
        return {
            "leaderboard": leaderboard,
            "summary": {},
            "verdict": {},
            "context": pd.DataFrame(),
            "translation": pd.DataFrame(),
            "team_profile": pd.DataFrame(),
            "team_fit_read": "",
            "evidence": pd.DataFrame(),
            "game_log": pd.DataFrame(),
            "selected_game_periods": pd.DataFrame(),
        }

    goalie_row = leaderboard[leaderboard["Goalie_Key"] == goalie_key.lower()].iloc[0].to_dict()
    context = _qualified_context(leaderboard, game_phase)
    game_log = _build_goalie_game_log(goalie_key.lower(), master_report.get(goalie_key.lower(), {}), game_phase)
    team_profile, team_fit_read = _build_team_system_profile(
        goalie_key.lower(),
        master_report.get(goalie_key.lower(), {}),
        game_phase,
        goalie_row,
    )
    selected_game_id = str(game_log.sort_values("Date", ascending=False).iloc[0]["Game_ID"]) if not game_log.empty else None
    period_df = build_selected_game_periods(goalie_key.lower(), selected_game_id) if selected_game_id else pd.DataFrame()

    return {
        "leaderboard": leaderboard,
        "summary": goalie_row,
        "verdict": _build_verdict(goalie_row, leaderboard, game_phase),
        "context": _build_context_rows(goalie_row, context),
        "translation": _build_translation(goalie_row, context),
        "team_profile": team_profile,
        "team_fit_read": team_fit_read,
        "evidence": _build_evidence_rows(goalie_row),
        "game_log": game_log,
        "selected_game_periods": period_df,
    }


def build_selected_game_periods(goalie_key, game_id):
    report = load_master_reports().get(goalie_key.lower(), {}).get(str(game_id), {})
    if not report:
        return pd.DataFrame()

    rows = []
    for period, period_data in sorted(report.get("periods", {}).items(), key=lambda item: int(item[0])):
        stats = period_data.get("stats", {})
        sequences = stats.get("Sequences", 0)
        service, tax = calculate_service_tax(stats, include_baseline=True)
        slc = current_period_score(report, period)
        net_load = round(service - tax, 3)
        period_number = int(period)
        period_minutes = 5 if period_number == 4 else 20
        xga_per_60 = (stats.get("xG", 0) / period_minutes) * 60 if period_minutes else 0

        rows.append({
            "Period": f"P{period}",
            "SLC": slc,
            "Sequences": sequences,
            "Pressure_Efficiency_per_Sequence": _safe_div(slc, sequences),
            "Pressure_Survival": _pressure_survival_score(stats),
            "Relief_Efficiency": _relief_efficiency_score(stats),
            "Net_Load": net_load,
            "Net_Load_per_Sequence": _safe_div(net_load, sequences),
            "Service": service,
            "Service_per_Sequence": _safe_div(service, sequences),
            "Tax": tax,
            "Tax_per_Sequence": _safe_div(tax, sequences),
            "Relief_Capture_Rate": _relief_capture_rate(stats),
            "xGA_Per_60": round(xga_per_60, 3),
            "NPW": stats.get("NPW", 0),
            "NPW_SH": stats.get("NPW_SH", 0),
            "UA": stats.get("UA", 0),
            "UA_SH": stats.get("UA_SH", 0),
            "RP": stats.get("RP", 0),
            "RP_SH": stats.get("RP_SH", 0),
            "iGvA": stats.get("iGvA", 0),
            "iGvA_SH": stats.get("iGvA_SH", 0),
        })

    return pd.DataFrame(rows)

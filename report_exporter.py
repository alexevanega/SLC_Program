from datetime import datetime


def _fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.000"


def dataframe_csv(df):
    if df is None or df.empty:
        return ""
    return df.to_csv(index=False)


def rankings_summary(leaderboard_df, game_phase):
    lines = [
        "SLC Rankings Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Game Set: {game_phase}",
        "",
    ]
    if leaderboard_df.empty:
        lines.append("No leaderboard data available.")
        return "\n".join(lines)

    lines.extend([
        f"Goalies shown: {len(leaderboard_df)}",
        f"Avg SLC baseline: {_fmt(leaderboard_df['Avg_SLC'].mean())}",
        f"Sovereignty baseline: {_fmt(leaderboard_df['Sovereignty'].mean())}",
        "",
        "Category Counts:",
    ])
    for category, count in leaderboard_df["Category"].value_counts().items():
        lines.append(f"- {category}: {int(count)}")

    lines.extend(["", "Leaderboard:"])
    for rank, (_, row) in enumerate(leaderboard_df.sort_values("Total_SLC", ascending=False).iterrows(), start=1):
        lines.append(
            f"{rank}. {row['Goalie']} | {int(row['GP'])} GP | "
            f"Total SLC {_fmt(row['Total_SLC'])} | Avg SLC {_fmt(row['Avg_SLC'])} | "
            f"Sovereignty {_fmt(row['Sovereignty'])} | Category {row['Category']}"
        )
    return "\n".join(lines)


def goalie_audit_summary(goalie_name, game_phase, goalie_row, goalie_df, selected_game, period_df, loan_row=None):
    category = goalie_row.get("Category", goalie_row.get("Identity", "Unclassified"))
    lines = [
        "SLC Individual Goalie Audit",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Goalie: {goalie_name}",
        f"Game Set: {game_phase}",
        "",
        f"Category: {category}",
        f"GP: {int(goalie_row['GP'])}",
        f"Total SLC: {_fmt(goalie_row['Total_SLC'])}",
        f"Avg SLC: {_fmt(goalie_row['Avg_SLC'])}",
        f"Sovereignty: {_fmt(goalie_row['Sovereignty'])}",
        f"Reset Ratio: {_fmt(goalie_row['Reset_Ratio'])}",
        "",
    ]

    if selected_game is not None:
        lines.extend([
            "Selected Game:",
            f"{selected_game['Date']} {selected_game['Matchup']} | Game ID {selected_game['Game_ID']}",
            f"SLC: {_fmt(selected_game['SLC'])} | Service: {_fmt(selected_game['Service'], 0)} | "
            f"Tax: {_fmt(selected_game['Tax'], 0)} | Sovereignty: {_fmt(selected_game['Sovereignty'])}",
            "",
        ])

    if period_df is not None and not period_df.empty:
        lines.append("Selected Game Period Pressure:")
        for _, row in period_df.iterrows():
            lines.append(f"{row['Period']}: SLC {_fmt(row['SLC'])} | xGA/60 {_fmt(row['xGA_Per_60'])}")
        lines.append("")

    if loan_row:
        lines.extend([
            "High-Interest Loan:",
            f"Early Debt: {_fmt(loan_row.get('Early_Debt'))}",
            f"Early Red-Line Sequences: {int(loan_row.get('Early_Red_Line', 0))}",
            f"3rd Period Sovereignty: {_fmt(loan_row.get('P3_Sovereignty'))}",
            f"3rd Period GA: {int(loan_row.get('P3_GA', 0))}",
            "",
        ])

    lines.append("Game Log:")
    if goalie_df is None or goalie_df.empty:
        lines.append("No game log available.")
    else:
        for _, row in goalie_df.sort_values("Date").iterrows():
            lines.append(
                f"{row['Date']} {row['Matchup']} | SLC {_fmt(row['SLC'])} | "
                f"Service {_fmt(row['Service'], 0)} | Tax {_fmt(row['Tax'], 0)} | "
                f"Sovereignty {_fmt(row['Sovereignty'])}"
            )
    return "\n".join(lines)


def comparison_summary(goalie_a, goalie_b, game_phase, start_d, end_d, hyp_a, hyp_b, df_a, df_b):
    lines = [
        "SLC Dual-Goalie Comparison",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Game Set: {game_phase}",
        f"Window: {start_d} to {end_d}",
        f"Goalies: {goalie_a} vs {goalie_b}",
        "",
    ]

    if hyp_a is not None and hyp_b is not None and not hyp_a.empty and not hyp_b.empty:
        a_pressure = hyp_a["Pressure_Ratio"].mean()
        b_pressure = hyp_b["Pressure_Ratio"].mean()
        a_reset = hyp_a["Reset_Ratio"].mean()
        b_reset = hyp_b["Reset_Ratio"].mean()
        a_redline = hyp_a["Red_Line_Shifts"].mean()
        b_redline = hyp_b["Red_Line_Shifts"].mean()
        lines.extend([
            "Hypothesis Profile:",
            f"{goalie_a}: Pressure Ratio {_fmt(a_pressure)} | Reset {_fmt(a_reset)} | Red-Line {_fmt(a_redline)}",
            f"{goalie_b}: Pressure Ratio {_fmt(b_pressure)} | Reset {_fmt(b_reset)} | Red-Line {_fmt(b_redline)}",
            "",
        ])

    lines.append("Game Logs:")
    for goalie_name, df in [(goalie_a, df_a), (goalie_b, df_b)]:
        lines.append(f"{goalie_name}:")
        if df is None or df.empty:
            lines.append("No games in this window.")
        else:
            for _, row in df.sort_values("Date").iterrows():
                lines.append(f"{row['Date']} {row['Matchup']} | SLC {_fmt(row['SLC'])} | Sovereignty {_fmt(row['Sovereignty'])}")
        lines.append("")
    return "\n".join(lines)


def team_redline_summary(team, game_phase, selected_game, phase, metrics, sequence_df):
    lines = [
        "SLC Team Redline Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Team: {team}",
        f"Game Set: {game_phase}",
        f"Phase: {phase}",
        "",
        f"Game: {selected_game['Label']}",
        f"Efficiency Rating: {_fmt(metrics.get('Efficiency'), 1)}%",
        f"Median Sequence: {_fmt(metrics.get('Median'), 1)}s",
        f"Total Sequences: {int(metrics.get('Total', 0))}",
        "",
        "Sequence Durations:",
    ]
    if sequence_df is None or sequence_df.empty:
        lines.append("No sequence data available.")
    else:
        for _, row in sequence_df.iterrows():
            stress = f" | Stress: {row['Stress_Reasons']}" if row.get("Stress_Reasons") else ""
            lines.append(f"Period {row['Period']} | {row['Duration']}s{stress}")
    return "\n".join(lines)

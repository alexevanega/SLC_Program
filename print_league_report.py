import contextlib
import io
import logging
import os
from pathlib import Path

os.environ["STREAMLIT_LOG_LEVEL"] = "error"
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

with contextlib.redirect_stderr(io.StringIO()):
    from leaderboard_engine import get_league_leaderboard
from utility import load_goalie_data
from datetime import date

REPORT_DIR = Path("reports")


def format_number(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.000"


def goalie_key(goalie_name):
    return goalie_name.lower().replace(" ", "_")


def verdict_sentence(name, identity, sovereignty):
    if identity == "Stabilizer":
        return f"{name} controls shot quality and kills sequences. Net positive for their team."
    if identity == "Systems Man":
        return f"{name} manages the system calmly but is leaking on shot quality. Accountable, predictable, and somewhat taxing."
    if identity == "Functional Drain":
        return f"{name} shows save-making activity but keeps pressure alive. The talent is visible, but the workflow taxes the skaters."
    return f"{name} is leaking in both directions: sequence strain and shot-quality failure. Measurable systemic drain."


def interpret_avg_slc(value):
    value = float(value)
    if value > 0:
        return "positive means this goalie is adding more stability than strain"
    if value < 0:
        return "negative means this goalie is costing more stability than they provide"
    return "even means their stabilizing value and strain are balanced"


def interpret_reset(value):
    value = float(value)
    if value >= 1:
        return "more saves than defensive sequences faced"
    if value > 0:
        return "fewer saves than defensive sequences faced"
    return "no measurable reset sequence data"


def interpret_sovereignty(value):
    value = float(value)
    if value > 0:
        return "positive means they are beating expected shot quality"
    if value < 0:
        return "negative means goals allowed are higher than expected shot quality"
    return "neutral means actual results match expected shot quality"


def print_header(leaderboard):
    print("=============================================")
    print("  SYSTEMIC LOAD COEFFICIENT - LEAGUE REPORT")
    print(f"  Generated: {date.today().isoformat()}")
    print(f"  Goalies tracked: {len(leaderboard)}")
    print(f"  Total games processed: {int(leaderboard['GP'].sum())}")
    print("=============================================")
    print()


def print_verdict_list(leaderboard):
    print("---------------------------------------------")
    print("VERDICT LIST")
    print("---------------------------------------------")
    for rank, (_, row) in enumerate(leaderboard.sort_values("Total_SLC", ascending=False).iterrows(), start=1):
        name = row["Goalie"]
        identity = row["Identity"]
        sovereignty = float(row["Sovereignty"])
        print(
            f"#{rank}  {name}  |  {int(row['GP'])} GP  |  "
            f"Avg SLC: {format_number(row['Avg_SLC'])} ({interpret_avg_slc(row['Avg_SLC'])})  |  "
            f"{identity}  |  "
            f"Reset: {format_number(row['Reset_Ratio'])} ({interpret_reset(row['Reset_Ratio'])})  |  "
            f"Sovereignty: {format_number(sovereignty)} ({interpret_sovereignty(sovereignty)})"
        )
        print(f"       -> {verdict_sentence(name, identity, sovereignty)}")
        print()


def print_league_splits(leaderboard):
    total_goalies = len(leaderboard)
    category_counts = leaderboard["Identity"].value_counts()

    best_reset = leaderboard.loc[leaderboard["Reset_Ratio"].idxmax()]
    worst_reset = leaderboard.loc[leaderboard["Reset_Ratio"].idxmin()]
    best_sovereignty = leaderboard.loc[leaderboard["Sovereignty"].idxmax()]
    worst_sovereignty = leaderboard.loc[leaderboard["Sovereignty"].idxmin()]
    highest_slc = leaderboard.loc[leaderboard["Total_SLC"].idxmax()]
    lowest_slc = leaderboard.loc[leaderboard["Total_SLC"].idxmin()]

    print("---------------------------------------------")
    print("LEAGUE SPLITS")
    print("---------------------------------------------")
    for category, explanation in [
        ("Stabilizer", "effective and efficient"),
        ("Systems Man", "accountable but taxing"),
        ("Functional Drain", "active but leaking"),
        ("Systemic Drain", "ineffective and inefficient"),
    ]:
        count = int(category_counts.get(category, 0))
        pct = (count / total_goalies) * 100 if total_goalies else 0
        print(f"{category}: {count} goalies  ({pct:.1f}% of tracked) - {explanation}")
    print()
    print(f"League Avg SLC:        {format_number(leaderboard['Avg_SLC'].mean())} - average per-game stabilizing value across tracked goalies")
    print(f"League Avg Reset:      {format_number(leaderboard['Reset_Ratio'].mean())} - average saves per defensive sequence")
    print(f"League Avg Sovereignty:{format_number(leaderboard['Sovereignty'].mean())} - average shot quality control above or below expectation")
    print()
    print(f"Best Reset Ratio:    {best_reset['Goalie']}  ({format_number(best_reset['Reset_Ratio'])}) - most saves per defensive sequence")
    print(f"Worst Reset Ratio:   {worst_reset['Goalie']}  ({format_number(worst_reset['Reset_Ratio'])}) - fewest saves per defensive sequence")
    print(f"Best Sovereignty:    {best_sovereignty['Goalie']}  ({format_number(best_sovereignty['Sovereignty'])}) - strongest shot quality control")
    print(f"Worst Sovereignty:   {worst_sovereignty['Goalie']}  ({format_number(worst_sovereignty['Sovereignty'])}) - weakest shot quality control")
    print(f"Highest Total SLC:   {highest_slc['Goalie']}  ({format_number(highest_slc['Total_SLC'])}) - largest total stabilizing contribution")
    print(f"Lowest Total SLC:    {lowest_slc['Goalie']}  ({format_number(lowest_slc['Total_SLC'])}) - largest total drain")
    print()


def print_goalie_deep_dives(leaderboard):
    for _, row in leaderboard.sort_values("Total_SLC", ascending=False).iterrows():
        name = row["Goalie"]
        identity = row["Identity"]
        sovereignty = float(row["Sovereignty"])
        games = load_goalie_data(goalie_key(name))

        print("=============================================")
        print(f"{name}  -  {identity}")
        print(
            f"{int(row['GP'])} games  |  "
            f"Total SLC: {format_number(row['Total_SLC'])} ({interpret_avg_slc(row['Total_SLC'])})  |  "
            f"Avg SLC: {format_number(row['Avg_SLC'])} ({interpret_avg_slc(row['Avg_SLC'])})"
        )
        print(
            f"Reset Ratio: {format_number(row['Reset_Ratio'])} ({interpret_reset(row['Reset_Ratio'])})  |  "
            f"Sovereignty: {format_number(sovereignty)} ({interpret_sovereignty(sovereignty)})"
        )
        print("---------------------------------------------")
        print("GAME LOG:")

        total_service = 0.0
        total_tax = 0.0
        if games.empty:
            print("No game log found in the local vault.")
        else:
            for _, game in games.sort_values("Date").iterrows():
                total_service += float(game["Service"])
                total_tax += float(game["Tax"])
                print(
                    f"{game['Date']}  {game['Matchup']}  |  "
                    f"SLC: {format_number(game['SLC'])} ({interpret_avg_slc(game['SLC'])})  |  "
                    f"Service: {format_number(game['Service'], 0)} (sequence-ending help)  |  "
                    f"Tax: {format_number(game['Tax'], 0)} (extra defensive work created)  |  "
                    f"Sovereignty: {format_number(game['Sovereignty'])} ({interpret_sovereignty(game['Sovereignty'])})"
                )

        net_load = total_service - total_tax
        net_load_text = "net helper" if net_load >= 0 else "net drain"
        print()
        print("SEASON SUMMARY:")
        print(f"Total Service (NPW + TkA): {format_number(total_service, 0)}  - how many times they ended a sequence or won a puck battle")
        print(f"Total Tax (UA + RP + GvA):  {format_number(total_tax, 0)}  - how much extra work they created for their defense")
        print(f"Net Load:                   {format_number(net_load, 0)}  - Service minus Tax. Positive = net helper. Negative = net drain. This goalie is a {net_load_text}.")
        print()
        print(f"{name} in one sentence: {verdict_sentence(name, identity, sovereignty)}")
        print("=============================================")
        print()


def print_footer():
    print("=============================================")
    print("END OF REPORT")
    print("SLC measures goaltending as a service to the defensive unit.")
    print("Stabilizer: above league average SLC and above league average Sovereignty.")
    print("Systems Man: above league average SLC and below league average Sovereignty.")
    print("Functional Drain: below league average SLC and above league average Sovereignty.")
    print("Systemic Drain: below league average SLC and below league average Sovereignty.")
    print("Sovereignty measures shot quality control (actual SV% minus expected SV%).")
    print("Reset Ratio measures saves per defensive sequence.")
    print("A positive Net Load means the goalie is a net asset to their team's energy.")
    print("A negative Net Load means the goalie is costing their team more than they are giving back.")
    print("=============================================")


def main():
    leaderboard_fn = getattr(get_league_leaderboard, "__wrapped__", get_league_leaderboard)
    with contextlib.redirect_stderr(io.StringIO()):
        leaderboard = leaderboard_fn()
    if leaderboard.empty:
        print("No leaderboard data found. Run the SLC pipeline first.")
        return

    print_header(leaderboard)
    print_verdict_list(leaderboard)
    print_league_splits(leaderboard)
    print_goalie_deep_dives(leaderboard)
    print_footer()


def write_report():
    REPORT_DIR.mkdir(exist_ok=True)
    output_path = REPORT_DIR / f"slc_league_report_{date.today().isoformat()}.txt"
    report_buffer = io.StringIO()

    with contextlib.redirect_stdout(report_buffer):
        main()

    output_path.write_text(report_buffer.getvalue(), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    report_path = write_report()
    print(f"SLC league report written to: {report_path}")

import json
from datetime import datetime
from pathlib import Path

from slc_by_period_progression import analyze_progression_from_raw
from utility import (
    MASTER_REPORT_FILE,
    RAW_DATA_DIR,
    load_master_reports,
    prepare_report_for_master,
    save_master_reports,
)


def rebuild_master_report():
    master = load_master_reports()
    if not master:
        print("No master report found to rebuild.")
        return

    master_path = Path(MASTER_REPORT_FILE)
    backup_path = master_path.with_name(
        f"{master_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{master_path.suffix}"
    )
    raw_dir = Path(RAW_DATA_DIR)

    if master_path.exists():
        backup_path.write_text(master_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup written to {backup_path}")

    rebuilt_master = {}
    rebuilt = 0
    skipped_missing_raw = 0
    skipped_unusable = 0

    for goalie_key, games in master.items():
        rebuilt_master[goalie_key] = {}
        for game_id, old_report in games.items():
            raw_path = raw_dir / f"{game_id}.json"
            goalie_id = old_report.get("goalie_id")

            if not raw_path.exists():
                rebuilt_master[goalie_key][game_id] = prepare_report_for_master(old_report)
                skipped_missing_raw += 1
                print(f"Missing raw game {game_id}; keeping existing report for {goalie_key}.")
                continue

            with raw_path.open("r", encoding="utf-8") as f:
                raw_game = json.load(f)

            rebuilt_report = analyze_progression_from_raw(raw_game, goalie_id)
            if not rebuilt_report:
                rebuilt_master[goalie_key][game_id] = prepare_report_for_master(old_report)
                skipped_unusable += 1
                print(f"Could not rebuild game {game_id}; keeping existing report for {goalie_key}.")
                continue

            rebuilt_report["gameDate"] = old_report.get("gameDate") or raw_game.get("gameDate")

            rebuilt_master[goalie_key][game_id] = prepare_report_for_master(rebuilt_report)
            rebuilt += 1

    save_master_reports(rebuilt_master)

    print()
    print("Master report rebuild complete.")
    print(f"Rebuilt reports:      {rebuilt}")
    print(f"Missing raw skipped:  {skipped_missing_raw}")
    print(f"Unusable raw skipped: {skipped_unusable}")
    print(f"Updated master:       {master_path}")


if __name__ == "__main__":
    rebuild_master_report()

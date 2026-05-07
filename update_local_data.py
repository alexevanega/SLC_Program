import argparse
import json
from pathlib import Path

import requests

from slc_by_period_progression import analyze_progression_from_raw
from utility import (
    RAW_DATA_DIR,
    SCHEDULE_FILE,
    fetch_and_vault_raw_data,
    load_master_reports,
    save_master_reports,
    sanitize_goalie_name,
)


NHL_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

COMPLETED_STATES = {"FINAL", "OFF"}


def load_schedule():
    if not SCHEDULE_FILE.exists():
        return {}
    with SCHEDULE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_schedule(schedule):
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SCHEDULE_FILE.open("w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=4)


def fetch_completed_games(season="now", game_type=2):
    games_by_id = {}
    for team in NHL_TEAMS:
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
        print(f"Checking {team} schedule...")
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"  Schedule lookup failed for {team}: {exc}")
            continue

        for game in data.get("games", []):
            if game_type != "all" and game.get("gameType") != int(game_type):
                continue
            if game.get("gameState") not in COMPLETED_STATES:
                continue
            game_id = str(game.get("id"))
            if not game_id or game_id == "None":
                continue
            games_by_id[game_id] = game

    return games_by_id


def discover_local_raw_games(game_type=2):
    games_by_id = {}
    raw_dir = Path(RAW_DATA_DIR)
    if not raw_dir.exists():
        return games_by_id

    for raw_path in raw_dir.glob("*.json"):
        raw_game = load_raw_game(raw_path.stem)
        if not raw_game:
            continue
        if game_type != "all" and raw_game.get("gameType") != int(game_type):
            continue

        away = raw_game.get("awayTeam", {})
        home = raw_game.get("homeTeam", {})
        games_by_id[str(raw_path.stem)] = {
            "id": int(raw_path.stem) if raw_path.stem.isdigit() else raw_path.stem,
            "gameDate": raw_game.get("gameDate", "0000-00-00"),
            "gameType": raw_game.get("gameType"),
            "gameState": raw_game.get("gameState", "OFF"),
            "awayTeam": {"abbrev": away.get("abbrev", "UNK")},
            "homeTeam": {"abbrev": home.get("abbrev", "UNK")},
        }

    return games_by_id


def update_schedule_from_games(schedule, games_by_id):
    changed = 0
    for game_id, game in games_by_id.items():
        away = game.get("awayTeam", {}).get("abbrev", "UNK")
        home = game.get("homeTeam", {}).get("abbrev", "UNK")
        entry = {
            "matchup": f"{away} @ {home}",
            "date": game.get("gameDate", "0000-00-00"),
        }
        if schedule.get(game_id) != entry:
            schedule[game_id] = entry
            changed += 1
    return changed


def load_raw_game(game_id):
    raw_path = Path(RAW_DATA_DIR) / f"{game_id}.json"
    if not raw_path.exists():
        return None
    with raw_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def goalie_name_map(raw_game):
    names = {}
    for player in raw_game.get("rosterSpots", []):
        if player.get("positionCode") != "G":
            continue
        first = player.get("firstName", {}).get("default", "")
        last = player.get("lastName", {}).get("default", "")
        player_id = player.get("playerId")
        if player_id:
            names[player_id] = f"{first} {last}".strip()
    return names


def active_goalies(raw_game):
    goalies = set()
    for play in raw_game.get("plays", []):
        if play.get("typeDescKey") not in {"shot-on-goal", "goal"}:
            continue
        details = play.get("details", {})
        if not isinstance(details, dict):
            continue
        goalie_id = details.get("goalieInNetId")
        if goalie_id:
            goalies.add(goalie_id)
    return sorted(goalies)


def report_exists(master, game_id, goalie_id, goalie_name):
    goalie_key = sanitize_goalie_name(goalie_name)
    if str(game_id) in master.get(goalie_key, {}):
        return True

    for games in master.values():
        report = games.get(str(game_id))
        if report and report.get("goalie_id") == goalie_id:
            return True

    return False


def update_local_data(season="now", game_type=2, local_raw_only=False):
    raw_dir = Path(RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_reports()
    schedule = load_schedule()
    completed_games = (
        discover_local_raw_games(game_type=game_type)
        if local_raw_only
        else fetch_completed_games(season=season, game_type=game_type)
    )
    schedule_updates = update_schedule_from_games(schedule, completed_games)

    fetched_raw = 0
    processed_reports = 0
    skipped_existing_reports = 0
    skipped_no_raw = 0
    skipped_no_goalies = 0

    for game_id, game in sorted(completed_games.items(), key=lambda item: item[1].get("gameDate", "")):
        raw_path = raw_dir / f"{game_id}.json"
        if raw_path.exists():
            raw_game = load_raw_game(game_id)
        else:
            print(f"Fetching raw play-by-play for {game_id}...")
            raw_game = fetch_and_vault_raw_data(game_id)
            if raw_game:
                fetched_raw += 1

        if not raw_game or not raw_game.get("plays"):
            skipped_no_raw += 1
            continue

        names = goalie_name_map(raw_game)
        goalie_ids = active_goalies(raw_game)
        if not goalie_ids:
            skipped_no_goalies += 1
            continue

        for goalie_id in goalie_ids:
            goalie_name = names.get(goalie_id, f"Goalie {goalie_id}")
            goalie_key = sanitize_goalie_name(goalie_name)

            if report_exists(master, game_id, goalie_id, goalie_name):
                skipped_existing_reports += 1
                continue

            report = analyze_progression_from_raw(raw_game, goalie_id)
            if not report:
                continue

            total_stats = report.get("total", {}).get("stats", {})
            shots = total_stats.get("S_saves", 0) + total_stats.get("S_goals", 0)
            if shots <= 0:
                continue

            report["gameDate"] = game.get("gameDate")
            master.setdefault(goalie_key, {})[str(game_id)] = report
            processed_reports += 1
            print(f"Processed {goalie_name}: {game.get('gameDate')} game {game_id}")

    save_schedule(schedule)
    save_master_reports(master)

    print()
    print("Local data update complete.")
    print(f"Completed games discovered: {len(completed_games)}")
    print(f"Schedule entries updated:    {schedule_updates}")
    print(f"Raw games fetched:           {fetched_raw}")
    print(f"New goalie reports written:  {processed_reports}")
    print(f"Existing reports skipped:    {skipped_existing_reports}")
    print(f"No raw/play data skipped:    {skipped_no_raw}")
    print(f"No active goalies skipped:   {skipped_no_goalies}")
    return {
        "completed_games": len(completed_games),
        "schedule_updates": schedule_updates,
        "raw_games_fetched": fetched_raw,
        "processed_reports": processed_reports,
        "existing_reports_skipped": skipped_existing_reports,
        "no_raw_skipped": skipped_no_raw,
        "no_active_goalies_skipped": skipped_no_goalies,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch missing NHL raw games and append missing SLC goalie reports.")
    parser.add_argument("--season", default="now", help="Season id, such as 20252026, or 'now'. Default: now")
    parser.add_argument("--game-type", default=2, help="Game type to process, usually 2 for regular season. Use 'all' for every completed game type.")
    parser.add_argument("--local-raw-only", action="store_true", help="Process missing reports from raw games that are already saved locally.")
    args = parser.parse_args()
    update_local_data(season=args.season, game_type=args.game_type, local_raw_only=args.local_raw_only)

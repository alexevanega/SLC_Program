import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import requests

from slc_prediction_engine import attach_team_context_to_master
from slc_by_period_progression import analyze_progression_from_raw
from utility import (
    BASE_DATA_DIR,
    RAW_DATA_BASE_DIR,
    RAW_DATA_DIR,
    SCHEDULE_FILE,
    fetch_and_vault_raw_data,
    load_master_reports,
    prepare_report_for_master,
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
SEASON_PIPELINE_STATE_FILE = Path("./data/season_pipeline_state.json")


def _emit_progress(progress_callback, stage, current, total, message=""):
    percent = round((current / total) * 100, 1) if total else 100.0
    progress = {
        "stage": stage,
        "current": current,
        "total": total,
        "percent": percent,
        "message": message,
    }
    if progress_callback:
        progress_callback(progress)
    else:
        print(f"{stage}: {current}/{total} ({percent:.1f}%) {message}".rstrip())


def _request_json_with_retry(url, timeout=25, max_retries=3, throttle_seconds=0.5):
    for attempt in range(max_retries + 1):
        if throttle_seconds:
            time.sleep(throttle_seconds)

        response = requests.get(url, timeout=timeout)
        if response.status_code == 429 and attempt < max_retries:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 60
            print(f"Rate limited. Waiting {wait_seconds}s before retrying...")
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        return response.json()

    return {}


def season_raw_dir(season):
    return Path(RAW_DATA_BASE_DIR) / str(season)


def season_master_report_path(season):
    return Path(BASE_DATA_DIR) / f"master_report_{season}.json"


def season_backup_dir(season):
    return Path(BASE_DATA_DIR) / "backups" / str(season)


def load_season_pipeline_state():
    if not SEASON_PIPELINE_STATE_FILE.exists():
        return {}
    with SEASON_PIPELINE_STATE_FILE.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_season_pipeline_state(state):
    SEASON_PIPELINE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SEASON_PIPELINE_STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)


def clear_season_pipeline_state():
    if SEASON_PIPELINE_STATE_FILE.exists():
        SEASON_PIPELINE_STATE_FILE.unlink()


def has_pending_season_processing():
    state = load_season_pipeline_state()
    return bool(state.get("season") and state.get("raw_complete") and not state.get("processed_complete"))


def load_season_master_reports(season):
    report_path = season_master_report_path(season)
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def backup_season_master_report(season):
    report_path = season_master_report_path(season)
    if not report_path.exists():
        return None

    backup_dir = season_backup_dir(season)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"master_report_backup_{stamp}.json"
    backup_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(backup_path)


def save_season_master_reports(master, season, backup=True):
    report_path = season_master_report_path(season)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = backup_season_master_report(season) if backup else None
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=4)
    return {"path": str(report_path), "backup_path": backup_path}


def load_schedule():
    if not SCHEDULE_FILE.exists():
        return {}
    with SCHEDULE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_schedule(schedule):
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SCHEDULE_FILE.open("w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=4)


def fetch_completed_games(
    season="now",
    game_type=2,
    throttle_seconds=0,
    max_retries=0,
    progress_callback=None,
):
    games_by_id = {}
    for index, team in enumerate(NHL_TEAMS, start=1):
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
        _emit_progress(progress_callback, "schedule", index, len(NHL_TEAMS), f"Checking {team}")
        try:
            data = _request_json_with_retry(
                url,
                timeout=20,
                max_retries=max_retries,
                throttle_seconds=throttle_seconds,
            )
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


def discover_local_raw_games(game_type=2, raw_dir=None):
    games_by_id = {}
    raw_dir = Path(raw_dir or RAW_DATA_DIR)
    if not raw_dir.exists():
        return games_by_id

    for raw_path in raw_dir.glob("*.json"):
        raw_game = load_raw_game(raw_path.stem, raw_dir=raw_dir)
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


def load_raw_game(game_id, raw_dir=None):
    raw_path = Path(raw_dir or RAW_DATA_DIR) / f"{game_id}.json"
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


def fetch_raw_season_data(
    season,
    game_type=2,
    throttle_seconds=0.5,
    max_retries=3,
    progress_callback=None,
):
    """Fetch completed season games into data/raw_games/<season> and mark them pending processing."""
    if has_pending_season_processing():
        pending = load_season_pipeline_state()
        if str(pending.get("season")) != str(season):
            raise RuntimeError(
                f"Season {pending.get('season')} is already fetched and waiting to be processed."
            )

    raw_dir = season_raw_dir(season)
    raw_dir.mkdir(parents=True, exist_ok=True)

    schedule = load_schedule()
    completed_games = fetch_completed_games(
        season=season,
        game_type=game_type,
        throttle_seconds=throttle_seconds,
        max_retries=max_retries,
        progress_callback=progress_callback,
    )
    schedule_updates = update_schedule_from_games(schedule, completed_games)
    save_schedule(schedule)

    counts = {"fetched": 0, "cached": 0, "empty": 0, "failed": 0}
    game_ids = sorted(
        completed_games,
        key=lambda game_id: completed_games[game_id].get("gameDate", ""),
    )
    for index, game_id in enumerate(game_ids, start=1):
        _emit_progress(progress_callback, "raw_games", index, len(game_ids), f"Game {game_id}")
        raw_path = raw_dir / f"{game_id}.json"
        if raw_path.exists():
            raw_game = load_raw_game(game_id, raw_dir=raw_dir)
            counts["cached" if raw_game and raw_game.get("plays") else "empty"] += 1
            continue

        try:
            raw_game = fetch_and_vault_raw_data(
                game_id,
                raw_dir=str(raw_dir),
                throttle_seconds=throttle_seconds,
                max_retries=max_retries,
            )
            counts["fetched" if raw_game and raw_game.get("plays") else "empty"] += 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"Raw game fetch failed for {game_id}: {exc}")

    summary = {
        "season": str(season),
        "game_type": game_type,
        "unique_games": len(game_ids),
        "schedule_updates": schedule_updates,
        "raw_directory": str(raw_dir),
        **counts,
    }
    save_season_pipeline_state({
        **summary,
        "raw_complete": True,
        "processed_complete": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    return summary


def build_processed_season_report(season, game_type=2, progress_callback=None):
    """Build data/processedGames/master_report_<season>.json from saved raw games."""
    raw_dir = season_raw_dir(season)
    if not raw_dir.exists():
        raise FileNotFoundError(f"No raw game directory exists for season {season}: {raw_dir}")

    master = load_season_master_reports(season)
    schedule = load_schedule()
    completed_games = discover_local_raw_games(game_type=game_type, raw_dir=raw_dir)
    schedule_updates = update_schedule_from_games(schedule, completed_games)

    processed_reports = 0
    skipped_existing_reports = 0
    skipped_no_raw = 0
    skipped_no_goalies = 0
    game_ids = sorted(
        completed_games,
        key=lambda game_id: completed_games[game_id].get("gameDate", ""),
    )

    for index, game_id in enumerate(game_ids, start=1):
        _emit_progress(progress_callback, "processed_reports", index, len(game_ids), f"Game {game_id}")
        game = completed_games[game_id]
        raw_game = load_raw_game(game_id, raw_dir=raw_dir)
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
            master.setdefault(goalie_key, {})[str(game_id)] = prepare_report_for_master(report)
            processed_reports += 1

    master, team_context_updates = attach_team_context_to_master(master)
    save_schedule(schedule)
    save_result = save_season_master_reports(master, season)

    state = load_season_pipeline_state()
    if str(state.get("season")) == str(season):
        save_season_pipeline_state({
            **state,
            "processed_complete": True,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "master_report_path": save_result["path"],
        })

    return {
        "season": str(season),
        "game_type": game_type,
        "raw_games_found": len(game_ids),
        "schedule_updates": schedule_updates,
        "processed_reports": processed_reports,
        "existing_reports_skipped": skipped_existing_reports,
        "team_context_updates": team_context_updates,
        "no_raw_skipped": skipped_no_raw,
        "no_active_goalies_skipped": skipped_no_goalies,
        **save_result,
    }


def update_local_data(season="now", game_type=2, local_raw_only=False):
    raw_dir = Path(RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_season = None if season == "now" else season

    master = load_master_reports(season=report_season)
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
            master.setdefault(goalie_key, {})[str(game_id)] = prepare_report_for_master(report)
            processed_reports += 1
            print(f"Processed {goalie_name}: {game.get('gameDate')} game {game_id}")

    master, team_context_updates = attach_team_context_to_master(master)
    save_schedule(schedule)
    save_master_reports(master, season=report_season)

    print()
    print("Local data update complete.")
    print(f"Completed games discovered: {len(completed_games)}")
    print(f"Schedule entries updated:    {schedule_updates}")
    print(f"Raw games fetched:           {fetched_raw}")
    print(f"New goalie reports written:  {processed_reports}")
    print(f"Existing reports skipped:    {skipped_existing_reports}")
    print(f"Team contexts updated:       {team_context_updates}")
    print(f"No raw/play data skipped:    {skipped_no_raw}")
    print(f"No active goalies skipped:   {skipped_no_goalies}")
    return {
        "completed_games": len(completed_games),
        "schedule_updates": schedule_updates,
        "raw_games_fetched": fetched_raw,
        "processed_reports": processed_reports,
        "existing_reports_skipped": skipped_existing_reports,
        "team_context_updates": team_context_updates,
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

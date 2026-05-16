import argparse
import json
import shutil
import sys
from pathlib import Path

import requests

from update_local_data import COMPLETED_STATES, NHL_TEAMS
from utility import RAW_DATA_BASE_DIR


RAW_BASE = Path(RAW_DATA_BASE_DIR)


def _season_raw_dir(season):
    return RAW_BASE / str(season)


def _progress(current, total, label):
    percent = (current / total * 100) if total else 100
    sys.stdout.write(f"\r{label}: {current}/{total} ({percent:5.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def organize_existing_raw_games(season):
    """Move loose raw game JSON files into the season-specific raw directory."""
    RAW_BASE.mkdir(parents=True, exist_ok=True)
    season_dir = _season_raw_dir(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    for raw_path in RAW_BASE.glob("*.json"):
        target = season_dir / raw_path.name
        if target.exists():
            skipped += 1
            continue
        shutil.move(str(raw_path), str(target))
        moved += 1

    return {"moved": moved, "skipped": skipped, "directory": str(season_dir)}


def fetch_season_schedule(season, game_type=2, completed_only=True):
    """Fetch each team's season schedule and return unique game records by game id."""
    games_by_id = {}
    for index, team in enumerate(NHL_TEAMS, start=1):
        _progress(index, len(NHL_TEAMS), "Schedule")
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"\nSchedule lookup failed for {team}: {exc}")
            continue

        for game in data.get("games", []):
            if game_type != "all" and game.get("gameType") != int(game_type):
                continue
            if completed_only and game.get("gameState") not in COMPLETED_STATES:
                continue

            game_id = str(game.get("id"))
            if not game_id or game_id == "None":
                continue
            games_by_id[game_id] = game

    return games_by_id


def fetch_and_save_raw_game(game_id, season):
    season_dir = _season_raw_dir(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    raw_path = season_dir / f"{game_id}.json"
    if raw_path.exists():
        return "cached"

    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    data = response.json()
    if not data.get("plays"):
        return "empty"

    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return "fetched"


def scrub_season_raw_games(season, game_type=2, completed_only=True, organize_existing=False):
    if organize_existing:
        result = organize_existing_raw_games(season)
        print(
            f"Organized existing raw games: moved {result['moved']}, "
            f"skipped {result['skipped']} -> {result['directory']}"
        )

    games_by_id = fetch_season_schedule(
        season=season,
        game_type=game_type,
        completed_only=completed_only,
    )
    game_ids = sorted(games_by_id)
    print(f"Unique games found: {len(game_ids)}")

    counts = {"fetched": 0, "cached": 0, "empty": 0, "failed": 0}
    for index, game_id in enumerate(game_ids, start=1):
        _progress(index, len(game_ids), "Raw games")
        try:
            status = fetch_and_save_raw_game(game_id, season)
            counts[status] = counts.get(status, 0) + 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"\nRaw game fetch failed for {game_id}: {exc}")

    return {
        "season": str(season),
        "game_type": game_type,
        "unique_games": len(game_ids),
        **counts,
        "directory": str(_season_raw_dir(season)),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch raw NHL play-by-play for a full season.")
    parser.add_argument("--season", required=True, help="Season id, such as 20242025 or 20252026.")
    parser.add_argument("--game-type", default=2, help="Game type. Use 2 for regular season, 3 for playoffs, or all.")
    parser.add_argument("--include-unfinished", action="store_true", help="Do not filter to completed games only.")
    parser.add_argument(
        "--organize-existing",
        action="store_true",
        help="Move loose JSON files from data/raw_games into data/raw_games/<season> before fetching.",
    )
    args = parser.parse_args()

    game_type = args.game_type if args.game_type == "all" else int(args.game_type)
    summary = scrub_season_raw_games(
        season=args.season,
        game_type=game_type,
        completed_only=not args.include_unfinished,
        organize_existing=args.organize_existing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

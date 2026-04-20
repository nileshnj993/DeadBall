from pathlib import Path
from config.constants.constants import COMPETITIONS_PARQUET

ROOT = Path(__file__).resolve().parents[3]  # adjust once
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

def create_directory(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def get_competitions_parquet_path():
    competitons_path = Path.joinpath(DATA_DIR, "parquets", "competitions")
    create_directory(competitons_path)
    return Path.joinpath(competitons_path, COMPETITIONS_PARQUET)

def get_matches_parquet_path(competition_id: int, season_id: int):
    matches_path = Path.joinpath(DATA_DIR, "parquets", "matches")
    create_directory(matches_path)
    return Path.joinpath(matches_path, f"matches_c{competition_id}_s{season_id}.parquet")

def get_lineups_parquet_path(match_id: int):
    lineups_path = Path.joinpath(DATA_DIR, "parquets", "lineups")
    create_directory(lineups_path)
    return Path.joinpath(lineups_path, f"lineups_m{match_id}.parquet")

def get_events_parquet_path(match_id: int):
    events_path = Path.joinpath(DATA_DIR, "parquets", "events")
    create_directory(events_path)
    return Path.joinpath(events_path, f"events_m{match_id}.parquet")
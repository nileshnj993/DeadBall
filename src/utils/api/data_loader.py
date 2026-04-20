from statsbombpy import sb
import pandas as pd
from config.paths import paths

def getCompetitions(fetch_from_api: bool):
    competitions_parquet_path = paths.get_competitions_parquet_path()
    
    if not fetch_from_api:
        try:
            print("Fetching competitions from parquet...")
            return pd.read_parquet(competitions_parquet_path)
        except FileNotFoundError:
            print("File not found. Fetching competitions from StatsBomB API instead...")

    print("Fetching competitions from StatsBomB API...")
    try:
        competitions = sb.competitions()
        competitions.to_parquet(competitions_parquet_path)
        return competitions
    except Exception as e:
        print(f"An error occurred while fetching competitions: {e}")
        raise e

def getCompetitionById(
        competition_id: int,
        fetch_from_api: bool
    ):
    competitions = getCompetitions(fetch_from_api=fetch_from_api)
    return competitions[competitions["competition_id"] == competition_id]

def getSeasonId_ByCompetitionId(
        competition_id: int,
        fetch_from_api: bool
    ):
    competitions = getCompetitionById(competition_id=competition_id, fetch_from_api=fetch_from_api)
    return competitions[["season_id"]]

def getMatchId_ByCompetitionIdSeasonId(
        competition_id: int,
        season_id: int,
        fetch_from_api: bool
    ):
    matches_parquet_path = paths.get_matches_parquet_path(competition_id=competition_id, season_id=season_id)
    
    if not fetch_from_api:
        try:
            print(f"Fetching matches for competition {competition_id} and season {season_id} from parquet...")
            return pd.read_parquet(matches_parquet_path)[["match_id"]]
        except FileNotFoundError:
            print(f"File not found. Fetching matches for competition {competition_id} and season {season_id} from StatsBomB API instead...")
    
    print(f"Fetching matches for competition {competition_id} and season {season_id} from StatsBomb API...")
    
    try:
        matches = sb.matches(
            competition_id=competition_id, 
            season_id=season_id)
        matches.to_parquet(matches_parquet_path)
        return matches[["match_id"]]
    except Exception as e:
        print(f"An error occurred while fetching matches: {e}")
        raise e

def getLineup_ByMatchId(match_id: int, fetch_from_api: bool):
    lineups_parquet_path = paths.get_lineups_parquet_path(match_id=match_id)

    if not fetch_from_api:
        try:
            print(f"Fetching lineups for match {match_id} from parquet...")
            lineups_df = pd.read_parquet(lineups_parquet_path)
            lineups = {}
            for team_name in lineups_df["team_name"].unique():
                lineup_team = lineups_df[lineups_df["team_name"] == team_name].drop(columns=["team_name"]).reset_index(drop=True)
                lineups[team_name] = lineup_team
            return lineups
        except FileNotFoundError:
            print(f"File not found. Fetching lineups for match {match_id} from StatsBomB API instead...")

    try:
        print(f"Fetching lineups for match {match_id} from StatsBomB API...")
        lineups = sb.lineups(match_id=match_id)
        lineups = {
            team: df.copy().reset_index(drop=True)
            for team, df in lineups.items()
        }
    except Exception as e:
        print(f"An error occurred while fetching lineups: {e}")
        raise e

    lineup_dfs = []
    for team_name, df in lineups.items():
        df = df.copy()
        df["team_name"] = team_name
        lineup_dfs.append(df)

    combined_lineup_df = pd.concat(lineup_dfs, ignore_index=True)
    combined_lineup_df.to_parquet(lineups_parquet_path)
    return lineups

def getEvents_ByMatchId(match_id: int, fetch_from_api: bool):
    events_parquet_path = paths.get_events_parquet_path(match_id=match_id)

    if not fetch_from_api:
        try:
            print(f"Fetching events for match {match_id} from parquet...")
            return pd.read_parquet(events_parquet_path)
        except FileNotFoundError:
            print(f"File not found. Fetching events for match {match_id} from StatsBomB API instead...")

    try:
        print(f"Fetching events for match {match_id} from StatsBomB API...")
        events = sb.events(match_id=match_id)
        events.to_parquet(events_parquet_path)
        return events
    except Exception as e:
        print(f"An error occurred while fetching events: {e}")
        raise e
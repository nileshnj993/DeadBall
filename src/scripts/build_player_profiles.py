import argparse
import pandas as pd

from collections import defaultdict
from utils.api import data_loader
from utils.processing import player_profiles
from config.enums.enums import Competitions

def build_unique_player_df(
        fetch_from_api: bool,
        competition_id: int) -> pd.DataFrame:
    
    unique_player_list = []
    unique_player_set = set()

    print(f"Building Unique Player Dataframe using Statsbomb player data...")

    # get all seasons of competition (La Liga)
    season_ids = data_loader.getSeasonId_ByCompetitionId(fetch_from_api=fetch_from_api, competition_id=competition_id)
    # get all matches of seasons
    for season_id in season_ids:
        match_ids = data_loader.getMatchId_ByCompetitionIdSeasonId(
            fetch_from_api=fetch_from_api, 
            competition_id=competition_id, 
            season_id=season_id)

        for match_id in match_ids:
            # get all lineups of matches
            lineups = data_loader.getLineup_ByMatchId(fetch_from_api=fetch_from_api, match_id=match_id)

            for _, lineup in lineups.items():
                for _, player in lineup.iterrows():
                    player_row = defaultdict()

                    player_name = player["player_name"]
                    if (player_name in unique_player_set):
                        continue
                    
                    unique_player_set.add(player_name)
                    player_row["player_name"] = player_name
                    player_row["statsbomb_id"] = player["player_id"]
                    player_row["country"] = player["country"]

                    unique_player_list.append(player_row)

    unique_player_df = pd.DataFrame(unique_player_list)

    print(f"Number of Unique Players found : {len(unique_player_df)}")
    return unique_player_df

def build_wikidata_player_df(unique_player_df: pd.DataFrame):

    print(f"Building Player Dataframe using Wikidata...")
    wikidata_player_list = []

    for _, row in unique_player_df.iterrows():
        player_name = row["player_name"]

        player_info_dict = player_profiles.fetch_player_stats_from_wikidata(player_name=player_name)
        wikidata_player_list.append(player_info_dict)

    wikidata_player_df = pd.DataFrame(wikidata_player_list)

    print(f"Missing Wikidata Counts per column: {wikidata_player_df.isnull().sum()}")

    return wikidata_player_df

def merge_statsbomb_player_with_wikidata_player(statsbomb_player_df, wikidata_player_df):

# len(df) == len(gbm_df) + len(lgg_df)
# print(f"{len(df) == len(gbm_df) + len(lgg_df)}")
# print(f"Actual Distributon: GBM = {len(gbm_df)} | LGG = {len(lgg_df)}")
# print(f"Merged Distribution using {df['label'].value_counts()}")

    print(f"statsbomb df len: {len(statsbomb_player_df)}")
    print(f"wikidata df len: {len(statsbomb_player_df)}")

    print(f"Merging Statsbomb and Wikidata player data...")
    
    merged = statsbomb_player_df.merge(
        wikidata_player_df,
        on="player_name",
        how="left"
    )

    return merged

def main(args):
    fetch_from_api=args.fetch_from_api
    unique_player_df = build_unique_player_df(fetch_from_api=fetch_from_api, competition_id=Competitions.LA_LIGA.value)
    print("UNIQUE PLAYER DF: ")
    print(unique_player_df.head(5))
    wikidata_player_df = build_wikidata_player_df(unique_player_df=unique_player_df)
    print("WIKIDATA PLAYER DF: ")
    print(wikidata_player_df.head(5))
    merged_df = merge_statsbomb_player_with_wikidata_player(statsbomb_player_df=unique_player_df, wikidata_player_df=wikidata_player_df)
    merged_df.to_csv("test.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-from-api", action="store_true", default=False, help="Whether to fetch data from the StatsBomB API or from parquet files.")
    args = parser.parse_args()
    main(args)


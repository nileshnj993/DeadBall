
import argparse
from utils.api import data_loader

parser = argparse.ArgumentParser()
parser.add_argument("--fetch-from-api", action="store_true", default=False, help="Whether to fetch data from the StatsBomB API or from parquet files.")
args = parser.parse_args()

events = data_loader.getEvents_ByMatchId(match_id=3773386, fetch_from_api=args.fetch_from_api)
print(events["type"].value_counts())

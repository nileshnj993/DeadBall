import pandas as pd
from utils.processing import player_profiles
from utils.api import data_loader

# Script that loops through all lineups, gets a list of players, and fetches their profiles from Wikidata.
# It then merges this data with the available player info in the StatsBomb data and saves a master player profile dataset.
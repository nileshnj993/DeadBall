import time
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
from datetime import datetime

def fetch_player_stats_from_wikidata(player_name):

    time.sleep(0.5)
    print(f"Fetching WikiData for {player_name}")

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    query = f"""
    SELECT ?player ?playerLabel ?height ?dob ?nationalityLabel ?positionLabel ?weight WHERE {{
      ?player wdt:P31 wd:Q5;
              wdt:P106 wd:Q937857.  # footballer

      {{
        ?player rdfs:label "{player_name}"@en.
      }}
      UNION
      {{
        ?player skos:altLabel "{player_name}"@en.
      }}

      OPTIONAL {{ ?player wdt:P2048 ?height. }}   # height (meters)
      OPTIONAL {{ ?player wdt:P569 ?dob. }}       # date of birth
      OPTIONAL {{ ?player wdt:P27 ?nationality. }}
      OPTIONAL {{ ?player wdt:P413 ?position. }}
      OPTIONAL {{ ?player wdt:P2067 ?weight. }}
      OPTIONAL {{ ?player wdt:P21 ?sex. }}        # needed for foot sometimes

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if not results["results"]["bindings"]:
        return None

    r = results["results"]["bindings"][0]

    def get_val(key):
        return r.get(key, {}).get("value")

    # Convert + clean
    height = get_val("height")
    height_cm = None
    if height:
        if float(height) < 3: # Convert to cm if in meters
            height_cm = float(height) * 100
        else:
            height_cm = float(height)
  
    dob = get_val("dob")
    dob_str = None
    age = None

    if dob:
        dob_dt = datetime.fromisoformat(dob.replace("Z", ""))
        dob_str = dob_dt.strftime("%Y-%m-%d")
        age = int((datetime.now() - dob_dt).days / 365.25)

    player_id = get_val("player").split("/")[-1]

    return {
        "player_name": player_name,
        "wikidata_id": player_id,
        "height_cm": height_cm,
        "dob": dob_str,
        "age": age,
        "nationality": get_val("nationalityLabel"),
        "position": get_val("positionLabel"),
        "weight_kg": float(get_val("weight")) if get_val("weight") else None
    }
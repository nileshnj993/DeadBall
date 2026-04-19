from statsbombpy import sb

def getCompetitions(): # MOVE BOOL SOMEWHERE TO GLOBAL VALUE
    use_parquet = "INITIALISE HERE"
    if use_parquet:
        return #something
    else:
        return sb.competitions()

def getCompetitionById(
        competition_id: int
    ):
    competitions = getCompetitions()
    return competitions[competitions["competition_id"] == competition_id]

def getSeasonId_ByCompetitionId(
        competition_id: int
    ):
    competitions = getCompetitionById(competition_id=competition_id)
    return competitions[["season_id"]]


def getMatchId_ByCompetitionIdSeasonId(
        competition_id: int,
        season_id: int
    ):
    matches = sb.matches(
        competition_id=competition_id, 
        season_id=season_id)
    return matches[["match_id"]]

def getLineup_ByMatchId(match_id: int):
    lineups = sb.lineups(match_id=match_id) # 18245
    name_t1, name_t2 = lineups
    lineup_t1 = lineups[name_t1]
    lineup_t2 = lineups[name_t2]
    return (lineup_t1, lineup_t2)

def getEvents_ByMatchId(match_id: int):
    events = None

def main():
    print("START!")
    competitons = getCompetitions()
    print(competitons[['competition_id', 'competition_name']].drop_duplicates())
    # print(getSeasonId_ByCompetitionId(11))
    #print(getMatchId_ByCompetitionIdSeasonId(16, 1))

    # team1, team2 = sb.lineups(match_id=3773386)
    # print(team1)

if __name__ == "__main__":
    main()
1. For the corner taker - how do you define a player profile?
    - Per inidividual
    - Per player 'type' - possibly make a player profile type
        - Do you create a cluster model, to assign player type (or is that too complex?)
        - Do you just have columns for height, strength, weight etc

2. Stage 1 - Yes or No (shot created - if any of these columns has a value 'shot_statsbomb_xg', 'shot_outcome', 'shot_type')

# Plan - 21st April

## Two Stages
1. Stage 1 - predicting whether or not a corner situation results in a shot being taken
- This is a binary classification problem, where the target variable is whether or not a shot is created from a corner situation.
- Features: 
    - Player profile of the corner taker (height, weight, strength, etc.)
    - Corner Details (e.g., whether it's an in-swinger or out-swinger, the side of the pitch etc.)
    - Team tactics (e.g., whether the team typically sends players into the box during corners)
    - Opponent's defensive setup (e.g., how many players they typically have in the box during corners)
    - Historical performance of the corner taker (e.g., past success rate of creating shots from corners)
2. Stage 2 - predicting the expected goals value of a corner situation
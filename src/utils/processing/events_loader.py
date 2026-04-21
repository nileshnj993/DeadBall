def filter_events_by_play_pattern(events_df, play_pattern=None):
    df = events_df
    if play_pattern:
        df = df[df["play_pattern"] == play_pattern]
    return df.copy().reset_index(drop=True)
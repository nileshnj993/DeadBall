"""
DeadBall — Module 1: Trend Analysis
=====================================
"Have set piece styles changed over time?"

What this module does:
  1. Loads multiple seasons of StatsBomb open data
  2. Computes set piece goal % league-wide and per team over time
  3. Flags teams that have shifted to a "set piece heavy" style
  4. Measures long throw-in frequency as a tactical proxy metric
  5. Clusters teams by attacking profile:
       → Open Play Dominant / Transition / Set Piece Reliant / Balanced

Data: StatsBomb Open Data (free, no API key)
Best competition for multi-season trend: La Liga (3 seasons available)

Seasons available for La Liga (competition_id=11):
  season_id=4   → 2018/19
  season_id=42  → 2019/20
  season_id=90  → 2020/21

Usage:
  python trend_analysis.py
  (outputs saved to data/trends/)
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsbombpy import sb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

COMPETITION_ID = 11   # La Liga 

DATA_DIR   = "data"
TRENDS_DIR = os.path.join(DATA_DIR, "trends")

# StatsBomb pitch: 120 × 80 units, left → right
# Final third starts at x = 80
LONG_THROW_X_THRESHOLD = 80

# Clustering
N_CLUSTERS = 4

# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CACHE DATA
# ─────────────────────────────────────────────────────────────────

def load_season(competition_id: int, season_id: int, label: str) -> tuple:
    """
    Load (and cache as parquet) all events + matches for one season.
    Returns (events_df, matches_df).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    ev_path = os.path.join(DATA_DIR, f"events_{competition_id}_{season_id}.parquet")
    ma_path = os.path.join(DATA_DIR, f"matches_{competition_id}_{season_id}.parquet")

    if os.path.exists(ev_path) and os.path.exists(ma_path):
        print(f"  [{label}] Loading from cache...")
        return pd.read_parquet(ev_path), pd.read_parquet(ma_path)

    print(f"  [{label}] Fetching match list...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"  [{label}] {len(matches)} matches found. Downloading events...")

    all_events = []
    for i, row in matches.iterrows():
        ev = sb.events(match_id=row['match_id'])
        ev['match_id'] = row['match_id']
        all_events.append(ev)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(matches)} matches loaded")

    events = pd.concat(all_events, ignore_index=True)
    events.to_parquet(ev_path)
    matches.to_parquet(ma_path)
    print(f"  [{label}] Saved.")
    return events, matches


def load_all_seasons(competition_id: int, seasons: list) -> dict:
    """
    Load all seasons. Returns dict keyed by season label:
      { '2018/19': (events_df, matches_df), ... }
    """
    print(f"\nLoading {len(seasons)} seasons of competition {competition_id}...")
    data = {}
    for s in seasons:
        data[s['season_name']] = load_season(competition_id, s['season_id'], s['season_name'])
    print("All seasons loaded.\n")
    return data


# ─────────────────────────────────────────────────────────────────
# STEP 2 — EXTRACT GOALS BY TYPE
# ─────────────────────────────────────────────────────────────────
# StatsBomb shot events have a `shot_type` field:
#   'Open Play', 'Free Kick', 'Corner', 'Penalty', 'Kick Off'
#
# We define:
#   Direct set piece goal  → shot_type in ['Free Kick', 'Corner']
#   Penalty                → shot_type == 'Penalty'
#   Open play goal         → shot_type == 'Open Play'
#
# Indirect set piece goals (e.g. goal from a long throw sequence) are
# harder to attribute automatically without tracking the preceding event
# chain. We handle these separately via the "sequence attribution" method.

def extract_goals(events: pd.DataFrame) -> pd.DataFrame:
    """
    Return all goal events with a 'goal_type' label.

    Goal types:
      'Direct Set Piece'  — free kick or corner directly converted
      'Penalty'           — penalty kick
      'Open Play'         — open play shot
      'Throw-in Sequence' — goal preceded by a long throw-in within 4 events
      'Other'             — kick off goals etc.
    """
    shots = events[events['type'] == 'Shot'].copy()

    # shot_outcome is a dict column in raw statsbombpy; expand it
    if shots['shot_outcome'].dtype == object:
        shots['shot_outcome_name'] = shots['shot_outcome'].apply(
            lambda x: x.get('name', x) if isinstance(x, dict) else str(x)
        )
        shots['shot_type_name'] = shots['shot_type'].apply(
            lambda x: x.get('name', x) if isinstance(x, dict) else str(x)
        )
    else:
        shots['shot_outcome_name'] = shots['shot_outcome'].astype(str)
        shots['shot_type_name']    = shots['shot_type'].astype(str)

    goals = shots[shots['shot_outcome_name'] == 'Goal'].copy()

    def classify_goal(row):
        t = str(row['shot_type_name'])
        if 'Free Kick' in t:
            return 'Direct Set Piece'
        elif 'Corner' in t:
            return 'Direct Set Piece'
        elif 'Penalty' in t:
            return 'Penalty'
        elif 'Open Play' in t:
            return 'Open Play'
        else:
            return 'Other'

    goals['goal_type'] = goals.apply(classify_goal, axis=1)

    # ── Throw-in sequence attribution ──────────────────────────────
    # For each open-play goal, check if any of the 6 preceding events
    # was a long throw-in. If so, reclassify as 'Throw-in Sequence'.
    events_sorted = events.sort_values(['match_id', 'index']).reset_index(drop=True)

    # Build index lookup: event id → position in sorted list
    id_to_pos = {row['id']: i for i, row in events_sorted.iterrows()}

    def is_from_long_throw(goal_row):
        if goal_row['goal_type'] != 'Open Play':
            return False
        pos = id_to_pos.get(goal_row['id'])
        if pos is None or pos < 6:
            return False
        preceding = events_sorted.iloc[max(0, pos-6):pos]
        # Filter to same match
        preceding = preceding[preceding['match_id'] == goal_row['match_id']]
        for _, ev in preceding.iterrows():
            if ev['type'] != 'Pass':
                continue
            pt = str(ev.get('pass_type', ''))
            if 'Throw-in' not in pt:
                continue
            loc = ev.get('location')
            if isinstance(loc, list) and len(loc) >= 1:
                if float(loc[0]) > LONG_THROW_X_THRESHOLD:
                    return True
        return False

    goals['goal_type'] = goals.apply(
        lambda row: 'Throw-in Sequence' if is_from_long_throw(row) else row['goal_type'],
        axis=1
    )

    return goals


# ─────────────────────────────────────────────────────────────────
# STEP 3 — LEAGUE-WIDE TREND METRICS
# ─────────────────────────────────────────────────────────────────

def compute_league_trends(season_data: dict) -> pd.DataFrame:
    """
    For each season, compute:
    - Total goals
    - Goals by type (direct SP, throw-in seq, open play, penalty)
    - Set piece goal % (direct SP + throw-in seq) / total
    - Long throw-in count league-wide
    - Long throws per match
    """
    rows = []
    for label, (events, matches) in season_data.items():
        goals = extract_goals(events)
        n_matches = len(matches)

        total_goals   = len(goals)
        sp_goals      = (goals['goal_type'] == 'Direct Set Piece').sum()
        throw_goals   = (goals['goal_type'] == 'Throw-in Sequence').sum()
        penalty_goals = (goals['goal_type'] == 'Penalty').sum()
        op_goals      = (goals['goal_type'] == 'Open Play').sum()

        # Long throw-ins
        passes = events[events['type'] == 'Pass'].copy()
        passes['pass_type_name'] = passes['pass_type'].apply(
            lambda x: x.get('name', x) if isinstance(x, dict) else str(x)
        )
        long_throws = passes[
            (passes['pass_type_name'].str.contains('Throw-in', na=False)) &
            (passes['location'].apply(
                lambda l: isinstance(l, list) and float(l[0]) > LONG_THROW_X_THRESHOLD
            ))
        ]

        rows.append({
            'season':            label,
            'n_matches':         n_matches,
            'total_goals':       total_goals,
            'open_play_goals':   op_goals,
            'direct_sp_goals':   sp_goals,
            'throw_seq_goals':   throw_goals,
            'penalty_goals':     penalty_goals,
            'set_piece_goals':   sp_goals + throw_goals,
            'sp_goal_pct':       round((sp_goals + throw_goals) / total_goals * 100, 1) if total_goals else 0,
            'goals_per_match':   round(total_goals / n_matches, 2) if n_matches else 0,
            'long_throws_total': len(long_throws),
            'long_throws_per_match': round(len(long_throws) / n_matches, 1) if n_matches else 0,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# STEP 4 — PER-TEAM METRICS PER SEASON
# ─────────────────────────────────────────────────────────────────

def compute_team_metrics(events: pd.DataFrame, matches: pd.DataFrame, season_label: str) -> pd.DataFrame:
    """
    Per team per season:
    - Games played
    - Total goals scored
    - Set piece goals scored (direct + throw-in sequence)
    - Set piece goal % of own goals
    - Long throw-ins attempted per match
    - Corners per match
    - Free kick (attacking) per match
    - Set piece goals conceded

    Returns one row per team.
    """
    goals = extract_goals(events)

    # Passes for set piece counts
    passes = events[events['type'] == 'Pass'].copy()
    passes['pass_type_name'] = passes['pass_type'].apply(
        lambda x: x.get('name', x) if isinstance(x, dict) else str(x)
    )

    # Team match counts (home + away)
    home = matches[['match_id', 'home_team']].rename(columns={'home_team': 'team'})
    away = matches[['match_id', 'away_team']].rename(columns={'away_team': 'team'})
    team_matches_df = pd.concat([home, away]).drop_duplicates()
    games_played = team_matches_df.groupby('team')['match_id'].count().rename('games_played')

    rows = []
    for team in sorted(goals['team'].unique()):
        team_goals = goals[goals['team'] == team]
        n_games = games_played.get(team, 1)

        total_goals_scored = len(team_goals)
        sp_goals_scored = team_goals['goal_type'].isin(['Direct Set Piece', 'Throw-in Sequence']).sum()
        sp_goal_pct = round(sp_goals_scored / total_goals_scored * 100, 1) if total_goals_scored else 0.0

        # Set piece goals conceded (goals the team let in)
        # We can derive this from shots on goal against this team
        shots_against = events[
            (events['type'] == 'Shot') &
            (events['team'] != team)
        ]

        team_passes = passes[passes['team'] == team]

        long_throws_n = team_passes[
            team_passes['pass_type_name'].str.contains('Throw-in', na=False) &
            team_passes['location'].apply(
                lambda l: isinstance(l, list) and float(l[0]) > LONG_THROW_X_THRESHOLD
            )
        ]

        corners_n  = team_passes[team_passes['pass_type_name'].str.contains('Corner', na=False)]
        freekicks_n = team_passes[team_passes['pass_type_name'].str.contains('Free Kick', na=False)]

        # Open play pass volume (proxy for possession/style)
        op_passes = team_passes[~team_passes['pass_type_name'].isin(['Corner', 'Free Kick', 'Throw-in', 'Kick Off'])]

        rows.append({
            'team':                   team,
            'season':                 season_label,
            'games_played':           int(n_games),
            'goals_scored':           int(total_goals_scored),
            'sp_goals_scored':        int(sp_goals_scored),
            'sp_goal_pct':            sp_goal_pct,
            'long_throws_per_match':  round(len(long_throws_n) / n_games, 2),
            'corners_per_match':      round(len(corners_n) / n_games, 2),
            'freekicks_per_match':    round(len(freekicks_n) / n_games, 2),
            'op_passes_per_match':    round(len(op_passes) / n_games, 1),
            'goals_per_match':        round(total_goals_scored / n_games, 2),
            'sp_goals_per_match':     round(sp_goals_scored / n_games, 3),
        })

    return pd.DataFrame(rows)


def compute_all_team_metrics(season_data: dict) -> pd.DataFrame:
    """Run compute_team_metrics across all seasons and concatenate."""
    frames = []
    for label, (events, matches) in season_data.items():
        print(f"  Computing team metrics: {label}...")
        df = compute_team_metrics(events, matches, label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────
# STEP 5 — "SET PIECE HEAVY" FLAG
# ─────────────────────────────────────────────────────────────────

def flag_set_piece_heavy(team_metrics: pd.DataFrame, threshold_pct: float = 35.0) -> pd.DataFrame:
    """
    Flag teams as 'set piece heavy' if their set piece goal %
    exceeds the threshold. Default: 35% (roughly 1 in 3 goals from SP).

    Also computes whether a team's SP goal % has increased year-on-year,
    indicating a deliberate tactical shift.
    """
    df = team_metrics.copy()
    df['sp_heavy'] = df['sp_goal_pct'] >= threshold_pct

    # Year-on-year change in SP goal % for teams appearing in multiple seasons
    df_sorted = df.sort_values(['team', 'season'])
    df['sp_pct_yoy_change'] = df_sorted.groupby('team')['sp_goal_pct'].diff()

    return df


# ─────────────────────────────────────────────────────────────────
# STEP 6 — CLUSTER TEAMS BY ATTACKING PROFILE
# ─────────────────────────────────────────────────────────────────
# Features used for clustering:
#   - sp_goal_pct          → set piece reliance
#   - long_throws_per_match → long throw aggression
#   - corners_per_match     → corner volume
#   - op_passes_per_match   → open play volume (possession proxy)
#   - goals_per_match       → overall attacking output
#
# Cluster labels (assigned after inspecting centroids):
#   0 → Set Piece Reliant
#   1 → Open Play Dominant
#   2 → Transition / Direct
#   3 → Balanced

CLUSTER_LABELS = {
    0: 'Set Piece Reliant',
    1: 'Open Play Dominant',
    2: 'Transition / Direct',
    3: 'Balanced',
}

CLUSTER_COLORS = {
    'Set Piece Reliant':    '#bc8cff',
    'Open Play Dominant':   '#58a6ff',
    'Transition / Direct':  '#d2a679',
    'Balanced':             '#3fb950',
}

CLUSTERING_FEATURES = [
    'sp_goal_pct',
    'long_throws_per_match',
    'corners_per_match',
    'op_passes_per_match',
    'goals_per_match',
]


def cluster_teams(team_metrics: pd.DataFrame, n_clusters: int = N_CLUSTERS, random_state: int = 42) -> pd.DataFrame:
    """
    Cluster teams by attacking profile using KMeans.
    Adds 'cluster_id', 'cluster_label', and PCA projection columns
    ('pca_x', 'pca_y') for 2D visualisation.

    Note: clusters are re-labelled by interpreting centroids after fitting.
    If you change competitions or seasons, re-examine the centroid printout
    and update CLUSTER_LABELS above accordingly.
    """
    df = team_metrics.dropna(subset=CLUSTERING_FEATURES).copy()

    X = df[CLUSTERING_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    df['cluster_id'] = kmeans.fit_predict(X_scaled)

    # Print centroids (in original scale) so you can interpret them
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids, columns=CLUSTERING_FEATURES)
    centroid_df.index.name = 'cluster_id'
    print("\n── Cluster Centroids (original scale) ──")
    print(centroid_df.round(2).to_string())
    print("\nInspect the above and update CLUSTER_LABELS in the script if needed.\n")

    # Auto-assign labels based on which centroid has highest sp_goal_pct
    sp_idx     = CLUSTERING_FEATURES.index('sp_goal_pct')
    op_idx     = CLUSTERING_FEATURES.index('op_passes_per_match')
    lt_idx     = CLUSTERING_FEATURES.index('long_throws_per_match')
    goals_idx  = CLUSTERING_FEATURES.index('goals_per_match')

    # Rank clusters on each dimension
    sp_rank    = np.argsort(-centroids[:, sp_idx])    # highest SP% first
    op_rank    = np.argsort(-centroids[:, op_idx])    # most open play passes first
    lt_rank    = np.argsort(-centroids[:, lt_idx])    # most long throws first
    goals_rank = np.argsort(-centroids[:, goals_idx]) # most goals first

    auto_labels = {}
    used = set()

    def assign(cluster_id, label):
        auto_labels[cluster_id] = label
        used.add(cluster_id)

    assign(int(sp_rank[0]),    'Set Piece Reliant')
    remaining = [c for c in op_rank if c not in used]
    assign(int(remaining[0]), 'Open Play Dominant')
    remaining = [c for c in lt_rank if c not in used]
    assign(int(remaining[0]), 'Transition / Direct')
    remaining = [c for c in range(n_clusters) if c not in used]
    assign(int(remaining[0]), 'Balanced')

    df['cluster_label'] = df['cluster_id'].map(auto_labels)

    # PCA for 2D plot
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df['pca_x'] = coords[:, 0]
    df['pca_y'] = coords[:, 1]

    variance = pca.explained_variance_ratio_
    print(f"PCA variance explained: PC1={variance[0]*100:.1f}%  PC2={variance[1]*100:.1f}%\n")

    return df


# ─────────────────────────────────────────────────────────────────
# STEP 7 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────

DARK_BG  = '#0d1117'
DARK_FG  = '#e6edf3'
DARK_MID = '#161b22'
DARK_BRD = '#30363d'
BLUE     = '#58a6ff'
AMBER    = '#d29922'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    DARK_BG,
    'axes.edgecolor':    DARK_BRD,
    'axes.labelcolor':   DARK_FG,
    'xtick.color':       DARK_FG,
    'ytick.color':       DARK_FG,
    'text.color':        DARK_FG,
    'grid.color':        DARK_BRD,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.5,
    'font.family':       'DejaVu Sans',
})


def plot_league_trends(league_trends: pd.DataFrame, save_path: str = None):
    """
    3-panel figure:
    Left:   Set piece goal % over seasons
    Middle: Long throw-ins per match over seasons
    Right:  Goal composition stacked bar (open play / SP / penalty)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("League-Wide Set Piece Trends", fontsize=14, fontweight='bold', color=DARK_FG, y=1.02)

    seasons = league_trends['season'].tolist()
    x = np.arange(len(seasons))

    # Panel 1: SP goal %
    ax = axes[0]
    ax.plot(x, league_trends['sp_goal_pct'], marker='o', color=BLUE, linewidth=2.5, markersize=8)
    for xi, yi in zip(x, league_trends['sp_goal_pct']):
        ax.annotate(f'{yi:.1f}%', (xi, yi), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=10, color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_ylabel('Set Piece Goal %'); ax.set_title('SP Goals as % of Total')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, axis='y'); ax.set_ylim(0, max(league_trends['sp_goal_pct']) * 1.4)

    # Panel 2: Long throws per match
    ax = axes[1]
    ax.bar(x, league_trends['long_throws_per_match'], color='#bc8cff', alpha=0.85, width=0.5)
    for xi, yi in zip(x, league_trends['long_throws_per_match']):
        ax.text(xi, yi + 0.2, f'{yi:.1f}', ha='center', fontsize=10, color=DARK_FG)
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_ylabel('Long Throw-ins per Match'); ax.set_title('Long Throw-In Frequency')
    ax.grid(True, axis='y')

    # Panel 3: Goal composition stacked bar
    ax = axes[2]
    op  = league_trends['open_play_goals'].values
    sp  = league_trends['direct_sp_goals'].values
    thr = league_trends['throw_seq_goals'].values
    pen = league_trends['penalty_goals'].values
    oth = (league_trends['total_goals'] - op - sp - thr - pen).clip(0).values

    ax.bar(x, op,  label='Open Play',       color=BLUE,      alpha=0.9)
    ax.bar(x, sp,  label='Direct SP',       color=AMBER,     alpha=0.9, bottom=op)
    ax.bar(x, thr, label='Throw Sequence',  color='#bc8cff', alpha=0.9, bottom=op+sp)
    ax.bar(x, pen, label='Penalty',         color='#f85149', alpha=0.9, bottom=op+sp+thr)
    ax.bar(x, oth, label='Other',           color='#484f58', alpha=0.9, bottom=op+sp+thr+pen)
    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_ylabel('Goals'); ax.set_title('Goal Composition by Season')
    ax.legend(loc='upper left', fontsize=8, facecolor=DARK_MID, edgecolor=DARK_BRD)
    ax.grid(True, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_team_sp_pct_over_time(team_metrics_flagged: pd.DataFrame, top_n: int = 12, save_path: str = None):
    """
    Line plot showing each team's set piece goal % across seasons.
    Highlights 'set piece heavy' teams (dashed line + coloured).
    """
    # Only teams with data in multiple seasons
    team_counts = team_metrics_flagged.groupby('team')['season'].count()
    multi_season_teams = team_counts[team_counts > 1].index

    df = team_metrics_flagged[team_metrics_flagged['team'].isin(multi_season_teams)].copy()

    # Pick top_n teams with highest mean SP goal %
    mean_sp = df.groupby('team')['sp_goal_pct'].mean().nlargest(top_n).index
    df = df[df['team'].isin(mean_sp)]

    seasons = sorted(df['season'].unique())
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Set Piece Goal % Over Time — Top {top_n} Teams", fontsize=13, fontweight='bold', y=1.01)

    for team, grp in df.groupby('team'):
        grp = grp.sort_values('season')
        sp_heavy_any = grp['sp_heavy'].any()
        color  = BLUE if sp_heavy_any else '#484f58'
        lw     = 2.2 if sp_heavy_any else 1.2
        ls     = '--' if sp_heavy_any else '-'
        alpha  = 1.0 if sp_heavy_any else 0.5
        ax.plot(grp['season'], grp['sp_goal_pct'], marker='o', color=color,
                linewidth=lw, linestyle=ls, alpha=alpha, label=team)
        # Label final point
        last = grp.iloc[-1]
        ax.annotate(team, (last['season'], last['sp_goal_pct']),
                    textcoords='offset points', xytext=(5, 0),
                    fontsize=7.5, color=color, alpha=alpha)

    # Threshold line
    ax.axhline(35, color=AMBER, linewidth=1.2, linestyle=':', alpha=0.7)
    ax.text(seasons[-1], 36, 'SP Heavy threshold (35%)', fontsize=8, color=AMBER, ha='right')

    ax.set_ylabel('Set Piece Goal %')
    ax.set_xlabel('Season')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, axis='y')

    legend_elements = [
        Line2D([0], [0], color=BLUE,     lw=2, linestyle='--', label='Set Piece Heavy'),
        Line2D([0], [0], color='#484f58', lw=1, linestyle='-',  label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', facecolor=DARK_MID, edgecolor=DARK_BRD)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_cluster_scatter(clustered: pd.DataFrame, season: str = None, save_path: str = None):
    """
    PCA scatter of team clusters. One point per team per season.
    Colour = cluster label. Annotated with team names.
    """
    df = clustered.copy()
    if season:
        df = df[df['season'] == season]

    fig, ax = plt.subplots(figsize=(11, 7))
    title = f"Team Attacking Profile Clusters — {season or 'All Seasons'}"
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for label, grp in df.groupby('cluster_label'):
        color = CLUSTER_COLORS.get(label, '#888')
        ax.scatter(grp['pca_x'], grp['pca_y'], c=color, s=90,
                   label=label, alpha=0.85, edgecolors='white', linewidth=0.4, zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(row['team'], (row['pca_x'], row['pca_y']),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=7.5, color=color, alpha=0.9)

    ax.set_xlabel('PC1 (Set Piece / Possession axis)', fontsize=10)
    ax.set_ylabel('PC2 (Attacking output axis)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', facecolor=DARK_MID, edgecolor=DARK_BRD, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


def plot_long_throw_heatmap(team_metrics: pd.DataFrame, save_path: str = None):
    """
    Heatmap: teams (rows) × seasons (columns), value = long throws per match.
    Good for spotting which teams increased/decreased long throw usage.
    """
    pivot = team_metrics.pivot_table(
        index='team', columns='season', values='long_throws_per_match', aggfunc='mean'
    )
    pivot = pivot.fillna(0).sort_values(by=pivot.columns[-1], ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.suptitle("Long Throw-Ins per Match — Team Heatmap", fontsize=13, fontweight='bold')

    im = ax.imshow(pivot.values, aspect='auto', cmap='Blues', vmin=0)

    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=9, color='white' if val > pivot.values.max() * 0.6 else DARK_FG)

    plt.colorbar(im, ax=ax, label='Long Throws per Match', fraction=0.03, pad=0.04)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(TRENDS_DIR, exist_ok=True)

    print("=" * 55)
    print("  DeadBall — Module 1: Trend Analysis")
    print("=" * 55)

    competitions_df = sb.competitions()
    print(competitions_df.columns)
    print(competitions_df.shape)
    # match_frames = sb.frames(match_id=3772072, fmt='dataframe')
    count = 0
    for index, row in competitions_df.iterrows():
        #print(row)
        # break
        # comp_frames = sb.competition_frames(
        #     country=row['country_name'],
        #     division= row['division'],
        #     season=row['season_name']
        # )
        # if(len(comp_frames)>0):
        #     count+=1
        if row['match_available_360'] is not None:
            count+=1
            print(f"{row['competition_name']} - {row['season_name']}")
            print(f"{row['match_updated_360']}")
        #else:
            #print(row['match_available_360'])
    print(count)
    raise ValueError("TEST")
    
    competition_df = competitions_df[competitions_df['competition_id'] == COMPETITION_ID]

    # Extract starting year
    competition_df['start_year'] = competition_df['season_name'].str.extract(r'(\d{4})').astype(int)

    # Filter seasons after 2006
    competition_df = competition_df[competition_df['start_year'] > 2006]

    # Convert to desired format
    seasons = competition_df[['season_id', 'season_name']].to_dict(orient='records')

    # 1. Load all seasons
    season_data = load_all_seasons(COMPETITION_ID, seasons)

    # 2. League-wide trends
    print("Computing league-wide trends...")
    league_trends = compute_league_trends(season_data)
    league_trends.to_csv(os.path.join(TRENDS_DIR, "league_trends.csv"), index=False)
    print("\n── League Trends ──")
    print(league_trends.to_string(index=False))

    # 3. Per-team metrics across all seasons
    print("\nComputing per-team metrics...")
    team_metrics = compute_all_team_metrics(season_data)
    team_metrics = flag_set_piece_heavy(team_metrics, threshold_pct=35.0)
    team_metrics.to_csv(os.path.join(TRENDS_DIR, "team_metrics.csv"), index=False)

    # 4. Clustering (on most recent season for clearest signal)
    latest_season = seasons[-1]['season_name']
    print(f"\nClustering teams for {latest_season}...")
    latest_metrics = team_metrics[team_metrics['season'] == latest_season].copy()
    clustered = cluster_teams(latest_metrics)
    clustered.to_csv(os.path.join(TRENDS_DIR, "team_clusters.csv"), index=False)

    print("── Cluster Assignments ──")
    print(
        clustered[['team', 'cluster_label', 'sp_goal_pct', 'long_throws_per_match', 'op_passes_per_match']]
        .sort_values('cluster_label')
        .to_string(index=False)
    )

    # 5. Plots
    print("\nGenerating plots...")
    plot_league_trends(league_trends,
                       save_path=os.path.join(TRENDS_DIR, "league_trends.png"))
    plot_team_sp_pct_over_time(team_metrics,
                               save_path=os.path.join(TRENDS_DIR, "team_sp_pct_trend.png"))
    plot_cluster_scatter(clustered, season=latest_season,
                         save_path=os.path.join(TRENDS_DIR, "cluster_scatter.png"))
    plot_long_throw_heatmap(team_metrics,
                            save_path=os.path.join(TRENDS_DIR, "long_throw_heatmap.png"))

    print(f"\nAll outputs saved to {TRENDS_DIR}/")
    print("Module 1 complete. Next: run app.py for the interactive UI.")
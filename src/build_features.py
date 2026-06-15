"""
Step 2: Feature engineering.

Replays match history through the Elo engine and adds per-team rolling form
features. All features are computed from matches STRICTLY BEFORE the current
one (shift(1)) — no data leakage.

Outputs
-------
data/processed/model_data.csv    one row per match, features + target
data/processed/elo_current.csv   today's Elo ratings for every team
"""

import sys
from pathlib import Path

import pandas as pd

# Allow sibling imports when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
import elo as elo_module

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"


def team_long_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match): points, goals for/against, pre-match Elo."""
    home = pd.DataFrame({
        "match_id": matches.index,
        "team":     matches.home_team,
        "date":     matches.date,
        "gf":       matches.home_score,
        "ga":       matches.away_score,
        "elo":      matches.home_elo,
    })
    away = pd.DataFrame({
        "match_id": matches.index,
        "team":     matches.away_team,
        "date":     matches.date,
        "gf":       matches.away_score,
        "ga":       matches.home_score,
        "elo":      matches.away_elo,
    })
    long = pd.concat([home, away]).sort_values(["team", "date", "match_id"])
    long["points"] = 1.0
    long.loc[long.gf > long.ga, "points"] = 3.0
    long.loc[long.gf < long.ga, "points"] = 0.0
    return long


def rolling_features(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("team")
    long["form5"]       = g.points.transform(lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
    long["form10"]      = g.points.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    long["gf5"]         = g.gf.transform(    lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
    long["ga5"]         = g.ga.transform(    lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
    long["prev_date"]   = g.date.transform(  lambda s: s.shift(1))
    long["rest_days"]   = (long.date - long.prev_date).dt.days.clip(upper=60).fillna(30)
    long["elo_10ago"]   = g.elo.transform(   lambda s: s.shift(10))
    long["elo_delta10"] = long.elo - long.elo_10ago
    return long


def main():
    matches = pd.read_csv(DATA / "matches.csv", parse_dates=["date"])
    matches, final_ratings = elo_module.run_history(matches)

    long = team_long_frame(matches)
    long = rolling_features(long)

    # Pivot back to wide (one row per match)
    for side, opp in (("home", "away"), ("away", "home")):
        team_col = f"{side}_team"
        sub = long.set_index("match_id")[["team", "form5", "form10", "gf5", "ga5",
                                           "rest_days", "elo_delta10"]]
        sub = sub[sub.team == matches[team_col].reindex(sub.index)].drop(columns="team")
        for col in sub.columns:
            matches[f"{side}_{col}"] = sub[col]

    # Simpler per-side merge approach
    home_feats = long[["match_id", "team", "form5", "form10", "gf5", "ga5",
                        "rest_days", "elo_delta10"]].copy()
    home_feats.columns = ["match_id", "check_home"] + [
        f"home_{c}" for c in ["form5", "form10", "gf5", "ga5", "rest_days", "elo_delta10"]
    ]
    away_feats = long[["match_id", "team", "form5", "form10", "gf5", "ga5",
                        "rest_days", "elo_delta10"]].copy()
    away_feats.columns = ["match_id", "check_away"] + [
        f"away_{c}" for c in ["form5", "form10", "gf5", "ga5", "rest_days", "elo_delta10"]
    ]

    model_data = matches.copy().reset_index(names="match_id")
    home_feats = home_feats.groupby("match_id").first().reset_index()
    away_feats = away_feats.groupby("match_id").first().reset_index()

    model_data = model_data.merge(home_feats.drop(columns="check_home"), on="match_id", how="left")
    model_data = model_data.merge(away_feats.drop(columns="check_away"), on="match_id", how="left")

    model_data.to_csv(DATA / "model_data.csv", index=False)
    print(f"Saved {len(model_data):,} rows -> data/processed/model_data.csv")

    elo_df = pd.DataFrame(
        [{"team": t, "elo": round(r, 1)} for t, r in sorted(final_ratings.items(), key=lambda x: -x[1])]
    )
    elo_df.to_csv(DATA / "elo_current.csv", index=False)
    print(f"Saved {len(elo_df)} teams  -> data/processed/elo_current.csv")


if __name__ == "__main__":
    main()

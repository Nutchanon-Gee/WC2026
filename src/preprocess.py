"""
Step 1: Pre-processing.

Reads raw CSVs from data/raw/ and produces:
  data/processed/matches.csv          all played matches, cleaned
  data/processed/wc2026_fixtures.csv  real 2026 WC group-stage fixtures

Cleaning steps
--------------
1. Parse dates, sort chronologically.
2. Standardize team names:
   a. Date-aware mapping of former names -> current names (former_names.csv),
      e.g. Zaire -> DR Congo, Dahomey -> Benin.
   b. Successor mapping for defunct states so a team's rating history carries
      over (Soviet Union -> Russia, Yugoslavia -> Serbia, etc.).
3. Split played matches (score present) from scheduled fixtures (score null).
4. Attach penalty-shootout winners from shootouts.csv. For Elo a match that
   went to penalties counts as a DRAW; the shootout winner is kept separately.
5. Add the regulation-time outcome column used as the prediction target.
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

# Defunct-state successors: carry rating history onto the modern team.
SUCCESSORS = {
    "Soviet Union":           "Russia",
    "Yugoslavia":             "Serbia",
    "Serbia and Montenegro":  "Serbia",
    "Czechoslovakia":         "Czech Republic",
}

# Real 2026 World Cup group draw (12 groups × 4 teams).
GROUPS_2026 = {
    "A": ["Mexico",        "South Korea",            "Czech Republic", "South Africa"],
    "B": ["Canada",        "Bosnia and Herzegovina", "Qatar",          "Switzerland"],
    "C": ["Brazil",        "Morocco",                "Haiti",          "Scotland"],
    "D": ["United States", "Paraguay",               "Australia",      "Turkey"],
    "E": ["Germany",       "Curaçao",                "Ivory Coast",    "Ecuador"],
    "F": ["Netherlands",   "Japan",                  "Sweden",         "Tunisia"],
    "G": ["Belgium",       "Egypt",                  "Iran",           "New Zealand"],
    "H": ["Spain",         "Cape Verde",             "Saudi Arabia",   "Uruguay"],
    "I": ["France",        "Senegal",                "Iraq",           "Norway"],
    "J": ["Argentina",     "Algeria",                "Austria",        "Jordan"],
    "K": ["Portugal",      "DR Congo",               "Uzbekistan",     "Colombia"],
    "L": ["England",       "Croatia",                "Ghana",          "Panama"],
}


def load_raw():
    results   = pd.read_csv(RAW / "results.csv",      parse_dates=["date"])
    shootouts = pd.read_csv(RAW / "shootouts.csv",    parse_dates=["date"])
    former    = pd.read_csv(RAW / "former_names.csv", parse_dates=["start_date", "end_date"])
    return results, shootouts, former


def standardize_names(df: pd.DataFrame, former: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    former = former.copy()
    former["start_date"] = pd.to_datetime(former["start_date"], format="mixed")
    former["end_date"]   = pd.to_datetime(former["end_date"], format="mixed")
    for row in former.itertuples():
        in_window = (df.date >= row.start_date) & (df.date <= row.end_date)
        for col in ("home_team", "away_team"):
            if col in df.columns:
                df.loc[in_window & (df[col] == row.former), col] = row.current
    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].replace(SUCCESSORS)
    return df


def build_fixtures() -> pd.DataFrame:
    """Generate the 2026 WC group-stage fixture list (6 matches per group)."""
    rows = []
    base_date = pd.Timestamp("2026-06-11")
    day = 0
    for group, teams in GROUPS_2026.items():
        # Round-robin: each pair plays once
        pairs = [(teams[i], teams[j]) for i in range(4) for j in range(i+1, 4)]
        for k, (h, a) in enumerate(pairs):
            rows.append({
                "date":      base_date + pd.Timedelta(days=day % 18),
                "home_team": h,
                "away_team": a,
                "tournament": "FIFA World Cup",
                "neutral":   False,
                "group":     group,
            })
            day += 1
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results, shootouts, former = load_raw()

    results   = standardize_names(results,   former)
    shootouts = standardize_names(shootouts, former)
    results   = results.sort_values("date").reset_index(drop=True)

    played   = results.dropna(subset=["home_score", "away_score"]).copy()
    played[["home_score", "away_score"]] = played[["home_score", "away_score"]].astype(int)

    # Attach shootout winners
    shootouts = shootouts.rename(columns={"winner": "shootout_winner"})
    played = played.merge(
        shootouts[["date", "home_team", "away_team", "shootout_winner"]],
        on=["date", "home_team", "away_team"],
        how="left",
    )

    # Regulation-time outcome (target variable)
    played["outcome"] = "draw"
    played.loc[played.home_score > played.away_score, "outcome"] = "home_win"
    played.loc[played.home_score < played.away_score, "outcome"] = "away_win"

    played.to_csv(OUT / "matches.csv", index=False)
    print(f"Saved {len(played):,} played matches -> data/processed/matches.csv")

    fixtures = build_fixtures()
    fixtures.to_csv(OUT / "wc2026_fixtures.csv", index=False)
    print(f"Saved {len(fixtures)} fixtures      -> data/processed/wc2026_fixtures.csv")


if __name__ == "__main__":
    main()

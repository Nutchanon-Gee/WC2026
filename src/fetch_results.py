"""
fetch_results.py — pulls live 2026 World Cup results from football-data.org
and appends any new finished matches into data/raw/results.csv and
data/raw/goalscorers.csv so the simulation pipeline picks them up.

Usage:
    python src/fetch_results.py          # uses FOOTBALL_API_KEY env var
    python src/fetch_results.py <token>  # or pass token directly

The script is rate-limit-aware: it reads the X-RateLimit-Remaining header
and sleeps if needed, as requested by the API author.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
DOCS = ROOT / "docs"

API_BASE    = "https://api.football-data.org/v4"
COMPETITION = "WC"          # FIFA World Cup 2026
SEASON      = 2026

# Name mapping: football-data.org names → our canonical names
NAME_MAP = {
    "USA":                          "United States",
    "Korea Republic":               "South Korea",
    "IR Iran":                      "Iran",
    "Côte d'Ivoire":                "Ivory Coast",
    "Bosnia-Herzegovina":           "Bosnia and Herzegovina",
    "Cape Verde Islands":           "Cape Verde",
    "Congo DR":                     "DR Congo",
    "Czechia":                      "Czech Republic",
}


def canonical(name: str) -> str:
    return NAME_MAP.get(name, name)


def get(path: str, token: str) -> dict:
    """GET with automatic rate-limit handling."""
    url = API_BASE + path
    headers = {"X-Auth-Token": token}
    while True:
        r = requests.get(url, headers=headers, timeout=20)
        remaining = int(r.headers.get("X-RateLimit-Remaining", 10))
        if r.status_code == 429:
            wait = int(r.headers.get("X-RateLimit-Reset", 60))
            print(f"  rate limited — sleeping {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        if remaining < 3:
            print(f"  rate limit low ({remaining} left) — sleeping 12s")
            time.sleep(12)
        return r.json()


def fetch_matches(token: str) -> list[dict]:
    """Return all WC2026 matches from the API."""
    data = get(f"/competitions/{COMPETITION}/matches?season={SEASON}", token)
    return data.get("matches", [])


def update_results(matches: list[dict]) -> tuple[int, list[dict]]:
    """
    Merge finished matches into results.csv.
    Returns (n_new, list of new match dicts for played ticker).
    """
    csv_path = RAW / "results.csv"
    existing = pd.read_csv(csv_path, parse_dates=["date"])

    # Build a set of already-known (date, home, away) keys
    existing_keys = set(
        zip(
            existing["date"].dt.strftime("%Y-%m-%d"),
            existing["home_team"],
            existing["away_team"],
        )
    )

    new_rows = []
    new_played = []

    for m in matches:
        if m["status"] not in ("FINISHED",):
            continue
        date_str = m["utcDate"][:10]
        home = canonical(m["homeTeam"]["name"])
        away = canonical(m["awayTeam"]["name"])
        hs   = m["score"]["fullTime"]["home"]
        as_  = m["score"]["fullTime"]["away"]
        if hs is None or as_ is None:
            continue
        key = (date_str, home, away)
        if key in existing_keys:
            continue
        new_rows.append({
            "date":       date_str,
            "home_team":  home,
            "away_team":  away,
            "home_score": int(hs),
            "away_score": int(as_),
            "tournament": "FIFA World Cup",
            "city":       m.get("venue", ""),
            "country":    "USA",
            "neutral":    False,
        })
        new_played.append({
            "home":  home,
            "away":  away,
            "score": f"{int(hs)}–{int(as_)}",
        })
        print(f"  + {home} {int(hs)}–{int(as_)} {away}  ({date_str})")

    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        updated = pd.concat([existing, new_df], ignore_index=True)
    updated["date"] = pd.to_datetime(updated["date"]).dt.strftime("%Y-%m-%d")
    updated = updated.sort_values("date")
        updated.to_csv(csv_path, index=False)
        print(f"results.csv: added {len(new_rows)} new match(es)")
    else:
        print("results.csv: no new finished matches")

    return len(new_rows), new_played


def update_goalscorers(matches: list[dict], token: str) -> None:
    """Fetch scorers for newly finished matches and append to goalscorers.csv."""
    csv_path = RAW / "goalscorers.csv"
    existing = pd.read_csv(csv_path, parse_dates=["date"])
    existing_keys = set(
        zip(
            existing["date"].dt.strftime("%Y-%m-%d"),
            existing["home_team"],
            existing["away_team"],
        )
    )

    new_rows = []
    for m in matches:
        if m["status"] != "FINISHED":
            continue
        date_str = m["utcDate"][:10]
        home = canonical(m["homeTeam"]["name"])
        away = canonical(m["awayTeam"]["name"])
        if (date_str, home, away) in existing_keys:
            continue
        # Fetch detailed match for goal events
        try:
            detail = get(f"/matches/{m['id']}", token)
            for goal in detail.get("goals", []):
                scorer = goal.get("scorer", {}).get("name", "Unknown")
                team   = canonical(goal.get("team", {}).get("name", ""))
                minute = goal.get("minute", 0) or 0
                own    = goal.get("type") == "OWN"
                pen    = goal.get("type") == "PENALTY"
                new_rows.append({
                    "date":      date_str,
                    "home_team": home,
                    "away_team": away,
                    "team":      team,
                    "scorer":    scorer,
                    "minute":    int(minute),
                    "own_goal":  own,
                    "penalty":   pen,
                })
        except Exception as e:
            print(f"  could not fetch scorers for match {m['id']}: {e}")

    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        updated = pd.concat([existing, new_df], ignore_index=True).sort_values(["date","minute"])
        updated.to_csv(csv_path, index=False)
        print(f"goalscorers.csv: added {len(new_rows)} goal event(s)")
    else:
        print("goalscorers.csv: nothing new")


def write_played_json(played: list[dict]) -> None:
    """Write docs/played.json for the results ticker."""
    out = DOCS / "played.json"
    # Merge with any existing played.json
    existing = []
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            pass
    keys = {(p["home"], p["away"]) for p in existing}
    for p in played:
        if (p["home"], p["away"]) not in keys:
            existing.append(p)
    out.write_text(json.dumps(existing, ensure_ascii=False))
    print(f"played.json: {len(existing)} result(s) total")


def main():
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FOOTBALL_API_KEY", "")
    if not token:
        print("ERROR: no API token. Set FOOTBALL_API_KEY env var or pass as argument.")
        sys.exit(1)

    RAW.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    print(f"Fetching WC2026 matches from football-data.org …")
    matches = fetch_matches(token)
    finished = [m for m in matches if m["status"] == "FINISHED"]
    print(f"  {len(matches)} total matches, {len(finished)} finished")

    n_new, new_played = update_results(matches)
    if n_new:
        update_goalscorers(matches, token)

    write_played_json(new_played if new_played else [])
    print("Done.")


if __name__ == "__main__":
    main()

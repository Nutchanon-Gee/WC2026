"""
Step 4: Monte Carlo simulation of the 2026 FIFA World Cup.

Usage:  python src/simulate.py [n_sims]   (default 10000)

Per simulation:
  1. Group stage: sample scores from double-Poisson + Dixon-Coles model.
  2. After every simulated match, update Elo (K=60) and rolling form.
  3. Group ranking: points → GD → GF → random tiebreak.
     Top 2 advance; 8 best third-placed teams also advance.
  4. Knockouts follow the official FIFA R32 bracket.
     Third-place slots are constraint-matched per FIFA Annex C.
  5. Draws go to penalties: winner sampled from Elo expected score.

Output:
  results/sim_results.csv   per-team stage probabilities
  docs/data.js              full JSON payload for the web dashboard
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).parent))
import elo as elo_module

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "processed"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"

# ── Real 2026 group draw ──────────────────────────────────────────────────────
GROUPS = {
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
TEAM_GROUP = {t: g for g, teams in GROUPS.items() for t in teams}

# Official FIFA R32 bracket (matches 73-88)
R32 = {
    73: ("2A","2B"),  74: ("1E","3?"),  75: ("1F","2C"),  76: ("1C","2F"),
    77: ("1I","3?"),  78: ("2E","2I"),  79: ("1A","3?"),  80: ("1L","3?"),
    81: ("1D","3?"),  82: ("1G","3?"),  83: ("2K","2L"),  84: ("1H","2J"),
    85: ("1B","3?"),  86: ("1J","2H"),  87: ("1K","3?"),  88: ("2D","2G"),
}
SLOT_ALLOWED = {
    74:"ABCDF", 77:"CDFGH", 79:"CEFHI", 80:"EHIJK",
    81:"BEFIJ", 82:"AEHIJ", 85:"EFGIJ", 87:"DEIJL",
}
R16   = {89:(74,77), 90:(73,75), 91:(76,78), 92:(79,80),
         93:(83,84), 94:(81,82), 95:(86,88), 96:(85,87)}
QF    = {97:(89,90), 98:(93,94), 99:(91,92), 100:(95,96)}
SF    = {101:(97,98), 102:(99,100)}
FINAL = {103:(101,102)}

STAGES = ["group","r32","r16","qf","sf","final","champion"]

# ISO-2 codes for flags on the frontend
ISO2 = {
    "Mexico":"MX","South Korea":"KR","Czech Republic":"CZ","South Africa":"ZA",
    "Canada":"CA","Bosnia and Herzegovina":"BA","Qatar":"QA","Switzerland":"CH",
    "Brazil":"BR","Morocco":"MA","Haiti":"HT","Scotland":"GB-SCT",
    "United States":"US","Paraguay":"PY","Australia":"AU","Turkey":"TR",
    "Germany":"DE","Curaçao":"CW","Ivory Coast":"CI","Ecuador":"EC",
    "Netherlands":"NL","Japan":"JP","Sweden":"SE","Tunisia":"TN",
    "Belgium":"BE","Egypt":"EG","Iran":"IR","New Zealand":"NZ",
    "Spain":"ES","Cape Verde":"CV","Saudi Arabia":"SA","Uruguay":"UY",
    "France":"FR","Senegal":"SN","Iraq":"IQ","Norway":"NO",
    "Argentina":"AR","Algeria":"DZ","Austria":"AT","Jordan":"JO",
    "Portugal":"PT","DR Congo":"CD","Uzbekistan":"UZ","Colombia":"CO",
    "England":"GB-ENG","Croatia":"HR","Ghana":"GH","Panama":"PA",
}


class Model:
    def __init__(self, path: Path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.glm      = d["model"]
        self.rho      = d["rho"]
        self.features = d["features"]
        self.max_goals = d["max_goals"]

    def score_matrix(self, lh: float, la: float) -> np.ndarray:
        rho = self.rho
        mg  = self.max_goals
        grid = np.outer(
            poisson.pmf(np.arange(mg + 1), lh),
            poisson.pmf(np.arange(mg + 1), la),
        )
        grid[0, 0] *= max(0.0, 1 - lh * la * rho)
        grid[1, 0] *= max(0.0, 1 + la * rho)
        grid[0, 1] *= max(0.0, 1 + lh * rho)
        grid[1, 1] *= max(0.0, 1 - rho)
        grid = np.maximum(grid, 1e-10)
        grid /= grid.sum()
        return grid

    def predict_lambda(self, elo_gap100, is_home, gf5, opp_ga5, rest_wk):
        x = np.array([[elo_gap100, float(is_home), gf5, opp_ga5, rest_wk]])
        return float(self.glm.predict(x)[0])

    def simulate_match(self, home_state: dict, away_state: dict, neutral: bool, rng: np.random.Generator):
        """Return (home_goals, away_goals)."""
        hfa     = 0.0 if neutral else 1.0
        elo_gap = (home_state["elo"] - away_state["elo"]) / 100

        gf5_h = float(np.mean(home_state.get("gf5", [1.3])))
        ga5_h = float(np.mean(home_state.get("ga5", [1.3])))
        gf5_a = float(np.mean(away_state.get("gf5", [1.3])))
        ga5_a = float(np.mean(away_state.get("ga5", [1.3])))

        lh = self.predict_lambda(
            elo_gap100 =  elo_gap,
            is_home    = hfa,
            gf5        = gf5_h,
            opp_ga5    = ga5_a,
            rest_wk    = home_state.get("rest_days", 14) / 7,
        )
        la = self.predict_lambda(
            elo_gap100 = -elo_gap,
            is_home    = 0.0,
            gf5        = gf5_a,
            opp_ga5    = ga5_h,
            rest_wk    = away_state.get("rest_days", 14) / 7,
        )
        lh = max(lh, 0.01)
        la = max(la, 0.01)
        mat = self.score_matrix(lh, la)
        idx = rng.choice(mat.size, p=mat.ravel())
        hg, ag = divmod(idx, self.max_goals + 1)
        return int(hg), int(ag)


def seed_states(matches: pd.DataFrame) -> dict:
    """Build initial team states from full match history."""
    states = {}
    for m in matches.sort_values("date").itertuples():
        for team, gf, ga in [
            (m.home_team, m.home_score, m.away_score),
            (m.away_team, m.away_score, m.home_score),
        ]:
            if team not in states:
                states[team] = {
                    "elo": elo_module.INITIAL_RATING,
                    "gf5": [1.3]*5, "ga5": [1.3]*5,
                    "form5": [1.0]*5, "elo_history": [],
                    "rest_days": 14, "last_date": None,
                    "elo_delta10": 0.0,
                }
            s = states[team]
            s["gf5"]  = (s["gf5"]  + [gf])[-5:]
            s["ga5"]  = (s["ga5"]  + [ga])[-5:]
            pts = 3 if gf > ga else (1 if gf == ga else 0)
            s["form5"] = (s["form5"] + [pts])[-5:]
            s["elo_history"] = (s["elo_history"] + [s["elo"]])[-10:]
            if len(s["elo_history"]) >= 10:
                s["elo_delta10"] = s["elo"] - s["elo_history"][0]
        # Elo update
        if m.home_team in states and m.away_team in states:
            rh, ra = elo_module.update(
                states[m.home_team]["elo"], states[m.away_team]["elo"],
                int(m.home_score), int(m.away_score), m.tournament, bool(m.neutral)
            )
            states[m.home_team]["elo"] = rh
            states[m.away_team]["elo"] = ra
    return states


def update_state(state: dict, gf: int, ga: int, opp_elo: float,
                 tournament: str, neutral: bool, match_date):
    """Update a team's state after one simulated match."""
    state = dict(state)
    state["gf5"]  = (state["gf5"]  + [gf])[-5:]
    state["ga5"]  = (state["ga5"]  + [ga])[-5:]
    pts = 3 if gf > ga else (1 if gf == ga else 0)
    state["form5"] = (state["form5"] + [pts])[-5:]
    # Elo updated externally; elo_history maintained here
    state["elo_history"] = (state.get("elo_history", []) + [state["elo"]])[-10:]
    if len(state["elo_history"]) >= 10:
        state["elo_delta10"] = state["elo"] - state["elo_history"][0]
    state["rest_days"] = 4
    return state


def simulate_once(model: Model, base_states: dict, fixtures: pd.DataFrame,
                  played: pd.DataFrame, rng: np.random.Generator):
    """Run one full tournament simulation. Returns (reached, group_orders, table, third_qualifiers)."""
    states = {t: dict(s) for t, s in base_states.items()}

    # Ensure all WC teams exist in states
    for g, teams in GROUPS.items():
        for t in teams:
            if t not in states:
                states[t] = {
                    "elo": elo_module.INITIAL_RATING, "gf5": [1.3]*5, "ga5": [1.3]*5,
                    "form5": [1.0]*5, "elo_history": [], "rest_days": 14,
                    "last_date": None, "elo_delta10": 0.0,
                }

    # Group stage table: {group: {team: [pts, gd, gf]}}
    table = {g: {t: [0, 0, 0] for t in teams} for g, teams in GROUPS.items()}

    # Simulate group-stage fixtures
    for row in fixtures.itertuples():
        h, a = row.home_team, row.away_team
        if h not in states or a not in states:
            continue
        hg, ag = model.simulate_match(states[h], states[a], bool(row.neutral), rng)
        # Update table
        g = TEAM_GROUP.get(h)
        if g:
            if hg > ag:
                table[g][h][0] += 3
            elif hg == ag:
                table[g][h][0] += 1; table[g][a][0] += 1
            else:
                table[g][a][0] += 3
            table[g][h][1] += hg - ag; table[g][h][2] += hg
            table[g][a][1] += ag - hg; table[g][a][2] += ag
        # Update Elo
        rh, ra = elo_module.update(states[h]["elo"], states[a]["elo"], hg, ag, "FIFA World Cup", bool(row.neutral))
        states[h]["elo"] = rh; states[a]["elo"] = ra
        states[h] = update_state(states[h], hg, ag, states[a]["elo"], "FIFA World Cup", bool(row.neutral), row.date)
        states[a] = update_state(states[a], ag, hg, states[h]["elo"], "FIFA World Cup", bool(row.neutral), row.date)

    # Rank teams within groups
    def rank_key(t_row):
        return (-t_row[1][0], -t_row[1][1], -t_row[1][2], rng.random())

    group_orders = {}
    for g, t_table in table.items():
        group_orders[g] = [t for t, _ in sorted(t_table.items(), key=rank_key)]

    # Third-place teams ranking
    thirds = [(group_orders[g][2], table[g][group_orders[g][2]]) for g in GROUPS]
    thirds_sorted = sorted(thirds, key=lambda x: (-x[1][0], -x[1][1], -x[1][2], rng.random()))
    # 8 best third-place teams advance
    third_qualifiers = {t for t, _ in thirds_sorted[:8]}
    third_groups     = {t: TEAM_GROUP[t] for t, _ in thirds_sorted[:8]}

    # Position lookup: "1A" -> team, "3?" -> handled via third_groups
    pos = {}
    for g, order in group_orders.items():
        for i, t in enumerate(order, 1):
            pos[f"{i}{g}"] = t

    # Assign third-place teams to bracket slots (constraint matching)
    third_slots = [k for k, (s1, s2) in R32.items() if "3?" in (s1, s2)]
    available_thirds = dict(third_groups)

    def assign_thirds(slots, available):
        """Greedy constraint matching for third-place slots."""
        assignment = {}
        thirds_list = list(available.items())  # (team, group)
        rng.shuffle(thirds_list)
        used = set()
        for slot in slots:
            allowed = SLOT_ALLOWED.get(slot, "ABCDEFGHIJKL")
            for t, g in thirds_list:
                if t not in used and g in allowed:
                    assignment[slot] = t
                    used.add(t)
                    break
            if slot not in assignment and thirds_list:
                # fallback: any unused third
                for t, g in thirds_list:
                    if t not in used:
                        assignment[slot] = t
                        used.add(t)
                        break
        return assignment

    third_assignment = assign_thirds(third_slots, available_thirds)

    # Resolve a bracket slot
    def resolve(slot_key, match_num):
        if slot_key == "3?":
            return third_assignment.get(match_num, list(third_groups.keys())[0] if third_groups else "Unknown")
        return pos.get(slot_key, "Unknown")

    # Run knockouts
    winners = {}
    reached = {t: "group" for t in TEAM_GROUP}

    def ko_match(h, a, neutral=True):
        if h == "Unknown" or a == "Unknown":
            return h if h != "Unknown" else a
        hg, ag = model.simulate_match(states[h], states[a], neutral, rng)
        if hg == ag:  # penalties
            return h if rng.random() < elo_module.expected_score(states[h]["elo"], states[a]["elo"]) else a
        return h if hg > ag else a

    # R32
    for mn, (s1, s2) in R32.items():
        h = resolve(s1, mn)
        a = resolve(s2, mn)
        w = ko_match(h, a)
        winners[mn] = w
        if w in reached: reached[w] = "r32"
        loser = a if w == h else h
        if loser in reached and reached[loser] == "group": reached[loser] = "group"

    # R16
    for mn, (p1, p2) in R16.items():
        h, a = winners.get(p1, "Unknown"), winners.get(p2, "Unknown")
        w = ko_match(h, a)
        winners[mn] = w
        if w in reached: reached[w] = "r16"

    # QF
    for mn, (p1, p2) in QF.items():
        h, a = winners.get(p1, "Unknown"), winners.get(p2, "Unknown")
        w = ko_match(h, a)
        winners[mn] = w
        if w in reached: reached[w] = "qf"

    # SF
    for mn, (p1, p2) in SF.items():
        h, a = winners.get(p1, "Unknown"), winners.get(p2, "Unknown")
        w = ko_match(h, a)
        winners[mn] = w
        if w in reached: reached[w] = "sf"

    # Final
    h, a = winners.get(101, "Unknown"), winners.get(102, "Unknown")
    w = ko_match(h, a)
    winners[103] = w
    if w in reached: reached[w] = "champion"
    # finalist who lost
    finalist = a if w == h else h
    if finalist in reached: reached[finalist] = "final"

    return reached, group_orders, table, third_qualifiers


def main(n_sims=10000):
    RESULTS.mkdir(exist_ok=True)
    rng    = np.random.default_rng(42)
    model  = Model(MODELS / "poisson_model.pkl")
    matches  = pd.read_csv(DATA / "matches.csv",        parse_dates=["date"])
    fixtures = pd.read_csv(DATA / "wc2026_fixtures.csv", parse_dates=["date"])

    print(f"Running {n_sims:,} tournament simulations…")
    base_states = seed_states(matches)

    counts     = {t: dict.fromkeys(STAGES, 0) for t in TEAM_GROUP}
    pos_counts = {t: [0, 0, 0, 0] for t in TEAM_GROUP}
    thirdq     = dict.fromkeys(TEAM_GROUP, 0)
    pts_sum    = dict.fromkeys(TEAM_GROUP, 0.0)
    gd_sum     = dict.fromkeys(TEAM_GROUP, 0.0)

    for i in range(n_sims):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{n_sims:,}")
        reached, orders, table, third_qualifiers = simulate_once(
            model, base_states, fixtures, pd.DataFrame(), rng
        )
        for g, order in orders.items():
            for j, t in enumerate(order):
                if j < 4: pos_counts[t][j] += 1
            for t in GROUPS[g]:
                pts_sum[t] += table[g][t][0]
                gd_sum[t]  += table[g][t][1]
        for t in third_qualifiers:
            thirdq[t] += 1
        for team, stage in reached.items():
            for s in STAGES[:STAGES.index(stage) + 1]:
                counts[team][s] += 1

    out = pd.DataFrame(counts).T / n_sims
    out["won_group"] = pd.Series({t: pos_counts[t][0] for t in TEAM_GROUP}) / n_sims
    out["group"] = pd.Series(TEAM_GROUP)
    out = out.sort_values(["champion", "final", "sf"], ascending=False)
    out.to_csv(RESULTS / "sim_results.csv")
    print(f"\nSaved -> results/sim_results.csv")

    # Print top 20
    show = (out[[s for s in STAGES[::-1] if s != "group"] + ["won_group"]] * 100).round(1)
    show["group"] = out.group
    print("\n=== Probability (%) of reaching each stage — top 20 ===")
    print(show.head(20).to_string())

    global fixtures_global, played_global
    fixtures_global = fixtures
    played_global   = pd.DataFrame()
    export_dashboard(model, base_states, n_sims, out, pos_counts, thirdq, pts_sum, gd_sum)



def compute_upcoming_cards(model, base_states, fixtures):
    """Score grids for unplayed WC fixtures in the next 48 hours."""
    if fixtures is None or len(fixtures) == 0:
        return []
    today  = pd.Timestamp.utcnow().normalize().tz_localize(None)
    cutoff = today + pd.Timedelta(days=2)
    cards  = []
    for m in fixtures.itertuples():
        if not (today <= m.date <= cutoff):
            continue
        h, a = m.home_team, m.away_team
        if h not in base_states or a not in base_states:
            continue
        sh, sa = base_states[h], base_states[a]
        elo_gap = (sh["elo"] - sa["elo"]) / 100
        gf5_h = float(np.mean(sh.get("gf5", [1.3])))
        ga5_h = float(np.mean(sh.get("ga5", [1.3])))
        gf5_a = float(np.mean(sa.get("gf5", [1.3])))
        ga5_a = float(np.mean(sa.get("ga5", [1.3])))
        lh = model.predict_lambda( elo_gap, 1.0 if not m.neutral else 0.0, gf5_h, ga5_a, 2.0)
        la = model.predict_lambda(-elo_gap, 0.0, gf5_a, ga5_h, 2.0)
        lh, la = max(lh, 0.01), max(la, 0.01)
        mat = model.score_matrix(lh, la)
        sub = mat[:6, :6].copy()
        if sub.sum() > 0: sub /= sub.sum()
        from scipy.stats import poisson as _poisson
        import numpy as _np
        cards.append({
            "home":  h, "away": a,
            "date":  str(m.date.date()),
            "ph":    round(float(_np.tril(mat, -1).sum()), 3),
            "pd":    round(float(_np.trace(mat)), 3),
            "pa":    round(float(_np.triu(mat, 1).sum()), 3),
            "expH":  round(float(lh), 2),
            "expA":  round(float(la), 2),
            "grid":  [[round(float(sub[i,j]),4) for j in range(6)] for i in range(6)],
        })
    return cards

def export_dashboard(model, base_states, n, out, pos_counts, thirdq, pts_sum, gd_sum):
    """Write docs/data.js for the HTML dashboard."""
    group_order = {
        g: sorted(GROUPS[g], key=lambda t: (-pts_sum[t], -pos_counts[t][0]))
        for g in GROUPS
    }

    groups_json = {
        g: [
            {
                "team":    t,
                "expPts":  round(pts_sum[t] / n, 2),
                "expGd":   round(gd_sum[t]  / n, 1),
                "p1":      round(pos_counts[t][0] / n, 3),
                "p2":      round(pos_counts[t][1] / n, 3),
                "p3q":     round(thirdq[t]         / n, 3),
                "advance": round(float(out.loc[t, "r32"]), 3) if t in out.index else 0.0,
            }
            for t in group_order[g]
        ]
        for g in GROUPS
    }

    odds = [
        {
            "team":     t,
            "champion": round(float(row.champion), 4),
            "final":    round(float(row.final),    3),
            "sf":       round(float(row.sf),        3),
            "r32":      round(float(row.r32),       3),
        }
        for t, row in out.iterrows()
    ]

    bracket = predicted_bracket(model, base_states, group_order)
    upcoming = compute_upcoming_cards(model, base_states, fixtures_global)
    played_list = [
        {"home": r.home_team, "away": r.away_team,
         "score": f"{int(r.home_score)}–{int(r.away_score)}"}
        for r in played_global.itertuples()
    ] if played_global is not None and len(played_global) else []

    data = {
        "generated":     pd.Timestamp.now().strftime("%Y-%m-%d"),
        "nSims":         n,
        "codes":         ISO2,
        "groups":        groups_json,
        "titleOdds":     odds,
        "bracket":       bracket,
        "upcomingCards": upcoming,
        "played":        played_list,
    }

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    with open(docs / "data.js", "w", encoding="utf-8") as f:
        f.write("const DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n")
    print(f"Dashboard data -> docs/data.js")


def predicted_bracket(model, base_states, group_order):
    """Single deterministic bracket using the Elo favourite."""
    states = {t: dict(s) for t, s in base_states.items()}
    for g, teams in GROUPS.items():
        for t in teams:
            if t not in states:
                states[t] = {"elo": elo_module.INITIAL_RATING, "gf5":[1.3]*5, "ga5":[1.3]*5, "form5":[1.0]*5, "elo_history":[], "rest_days":14, "last_date":None, "elo_delta10":0.0}

    pos = {}
    for g, order in group_order.items():
        for i, t in enumerate(order, 1):
            pos[f"{i}{g}"] = t

    def fav(h, a):
        if h == "Unknown" or a == "Unknown":
            return h if h != "Unknown" else a
        sh, sa = states.get(h, {}), states.get(a, {})
        eh = elo_module.expected_score(sh.get("elo", 1500), sa.get("elo", 1500))
        return h if eh >= 0.5 else a

    winners = {}
    bracket_r32 = []
    for mn, (s1, s2) in R32.items():
        h = pos.get(s1, "TBD") if "?" not in s1 else "TBD"
        a = pos.get(s2, "TBD") if "?" not in s2 else "TBD"
        w = fav(h, a) if h != "TBD" and a != "TBD" else "TBD"
        winners[mn] = w
        bracket_r32.append({"match": mn, "home": h, "away": a, "winner": w})

    bracket_r16 = []
    for mn, (p1, p2) in R16.items():
        h, a = winners.get(p1, "TBD"), winners.get(p2, "TBD")
        w = fav(h, a) if h != "TBD" and a != "TBD" else "TBD"
        winners[mn] = w
        bracket_r16.append({"match": mn, "home": h, "away": a, "winner": w})

    bracket_qf = []
    for mn, (p1, p2) in QF.items():
        h, a = winners.get(p1, "TBD"), winners.get(p2, "TBD")
        w = fav(h, a)
        winners[mn] = w
        bracket_qf.append({"match": mn, "home": h, "away": a, "winner": w})

    bracket_sf = []
    for mn, (p1, p2) in SF.items():
        h, a = winners.get(p1, "TBD"), winners.get(p2, "TBD")
        w = fav(h, a)
        winners[mn] = w
        bracket_sf.append({"match": mn, "home": h, "away": a, "winner": w})

    h, a = winners.get(101, "TBD"), winners.get(102, "TBD")
    champion = fav(h, a)

    return {
        "r32": bracket_r32, "r16": bracket_r16,
        "qf":  bracket_qf,  "sf":  bracket_sf,
        "final": {"home": h, "away": a, "winner": champion},
        "champion": champion,
    }


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10000)

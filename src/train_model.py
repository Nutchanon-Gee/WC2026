"""Step 3: Train double-Poisson GLM with Dixon-Coles correction."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).parent))
DATA   = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS = Path(__file__).resolve().parents[1] / "models"
TRAIN_START, TEST_START = "2000-01-01", "2024-01-01"
MAX_GOALS = 12
FEATURES  = ["elo_gap100","is_home","gf5","opp_ga5","rest_wk"]

def long_design(matches):
    rows = []
    for side, opp in (("home","away"),("away","home")):
        idx = matches.index
        rows.append(pd.DataFrame({
            "goals":      matches[f"{side}_score"],
            "elo_gap100": (matches.get(f"{side}_elo", pd.Series(1500,index=idx)) -
                           matches.get(f"{opp}_elo",  pd.Series(1500,index=idx))) / 100,
            "is_home":    ((side=="home") & ~matches.neutral).astype(float),
            "gf5":        matches.get(f"{side}_gf5",  pd.Series(1.3,index=idx)).fillna(1.3),
            "opp_ga5":    matches.get(f"{opp}_ga5",   pd.Series(1.3,index=idx)).fillna(1.3),
            "rest_wk":    matches.get(f"{side}_rest_days", pd.Series(14,index=idx)).fillna(14)/7,
        }))
    return pd.concat(rows, ignore_index=True)

def score_matrix(lh, la, rho):
    g = np.outer(poisson.pmf(np.arange(MAX_GOALS+1),lh), poisson.pmf(np.arange(MAX_GOALS+1),la))
    g[0,0]*=max(0,1-lh*la*rho); g[1,0]*=max(0,1+la*rho)
    g[0,1]*=max(0,1+lh*rho);    g[1,1]*=max(0,1-rho)
    g=np.maximum(g,1e-10); g/=g.sum(); return g

def wdl(lh,la,rho):
    g=score_matrix(lh,la,rho)
    return np.tril(g,-1).sum(), np.trace(g), np.triu(g,1).sum()

def pred_lam(glm, elo_gap100, is_home, gf5, opp_ga5, rest_wk):
    x = np.array([[elo_gap100, is_home, gf5, opp_ga5, rest_wk]])
    return float(glm.predict(x)[0])

def main():
    MODELS.mkdir(exist_ok=True)
    data  = pd.read_csv(DATA/"model_data.csv", parse_dates=["date"]).dropna(subset=["home_score","away_score"])
    train = data[(data.date>=TRAIN_START)&(data.date<TEST_START)].copy()

    design = long_design(train.reset_index())
    glm = sm.GLM(design["goals"].fillna(0).astype(int),
                 design[FEATURES].fillna(0),
                 family=sm.families.Poisson()).fit()
    print(glm.summary().tables[1])

    best_rho, best_ll = 0.0, np.inf
    for rho in np.arange(0.0, 0.35, 0.02):
        preds=[]
        for _,row in train.iterrows():
            lh=pred_lam(glm,(row.get("home_elo",1500)-row.get("away_elo",1500))/100,
                        float(not row.get("neutral",False)),row.get("home_gf5",1.3) or 1.3,
                        row.get("away_ga5",1.3) or 1.3,(row.get("home_rest_days",14) or 14)/7)
            la=pred_lam(glm,(row.get("away_elo",1500)-row.get("home_elo",1500))/100,0.0,
                        row.get("away_gf5",1.3) or 1.3,row.get("home_ga5",1.3) or 1.3,
                        (row.get("away_rest_days",14) or 14)/7)
            hw,dr,aw=wdl(max(lh,0.01),max(la,0.01),rho)
            outcome=row.get("outcome","draw")
            preds.append(max(hw if outcome=="home_win" else (aw if outcome=="away_win" else dr),1e-10))
        ll=-np.log(preds).mean()
        print(f"  rho={rho:.2f}  ll={ll:.4f}")
        if ll<best_ll: best_ll,best_rho=ll,rho
    print(f"\nBest rho={best_rho:.2f}")

    pickle.dump({"model":glm,"rho":best_rho,"features":FEATURES,"max_goals":MAX_GOALS},
                open(MODELS/"poisson_model.pkl","wb"))
    print("Saved -> models/poisson_model.pkl")

if __name__=="__main__": main()

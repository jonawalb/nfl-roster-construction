"""Cap-dollar and draft-capital construction measures (OverTheCap via nflverse).

Adds to team_season.parquet:
- cap_cov: share of the league cap observed for the team-season (coverage)
- cap_acq_share: share of observed cap dollars on non-homegrown players
- n_acq_stars: acquired (FA/Trade) players carrying >= 5% of the cap
- picks_r1r2_for_vets: round 1-2 picks sent away in trades that returned a
  veteran player, dated to the trade season
"""
import logging

import numpy as np
import pandas as pd

from common import RAW, CLEAN, norm_team

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NICK = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
        "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
        "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
        "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
        "Rams": "LA", "Chargers": "LAC", "Raiders": "LV", "Dolphins": "MIA",
        "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
        "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "Seahawks": "SEA",
        "49ers": "SF", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
        "Redskins": "WAS", "Football Team": "WAS", "Washington": "WAS"}
STAR_CAP = 0.05
# Official NFL salary caps ($M). 2026 falls back to the value implied by
# OTC cap_number / cap_percent.
OFFICIAL_CAP = {2013: 123.0, 2014: 133.0, 2015: 143.28, 2016: 155.27,
                2017: 167.0, 2018: 177.2, 2019: 188.2, 2020: 198.2,
                2021: 182.5, 2022: 208.2, 2023: 224.8, 2024: 255.4,
                2025: 279.2}


def cap_rows() -> pd.DataFrame:
    c = pd.read_parquet(RAW / "contracts.parquet", columns=["otc_id", "gsis_id",
                                                          "season_history"])
    recs = []
    for otc, gsis, hist in c.itertuples(index=False):
        if hist is None:
            continue
        for h in hist:
            if h["year"] == "Total" or h.get("cap_number") is None:
                continue
            recs.append((otc, gsis, int(h["year"]), h["team"], h["cap_number"],
                         h.get("cap_percent")))
    d = pd.DataFrame(recs, columns=["otc_id", "gsis_id", "season", "nick",
                                    "cap", "cap_pct"])
    d = d.drop_duplicates(["otc_id", "season", "nick"])
    d["team"] = d.nick.map(NICK)
    log.info("unmapped team nicknames: %s", d[d.team.isna()].nick.unique()[:10])
    return d.dropna(subset=["team"])


def main() -> None:
    ts = pd.read_parquet(CLEAN / "team_season.parquet")
    p = pd.read_parquet(CLEAN / "player_team_season.parquet")
    d = cap_rows()
    d = d[d.season.between(2013, 2026) & d.cap.gt(0)]
    # League cap implied by cap_number / cap_percent (median across players)
    league_cap = (d[d.cap_pct.gt(0.01)].assign(c=lambda x: x.cap / x.cap_pct)
                  .groupby("season").c.median())
    league_cap.update(pd.Series(OFFICIAL_CAP))
    src = p[["gsis_id", "team", "season", "source", "homegrown"]].drop_duplicates(
        ["gsis_id", "team", "season"])
    d = d.merge(src, on=["gsis_id", "team", "season"], how="left")
    d["cap_share"] = d.cap / d.season.map(league_cap)
    d["acq"] = d.source.isin(["FA", "Trade"])
    g = d[d.source.notna()].groupby(["team", "season"])
    cap = pd.DataFrame({
        "cap_obs": g.cap.sum(),
        "cap_acq_share": g.apply(lambda x: x.loc[x.acq, "cap"].sum() / x.cap.sum()),
        "cap_trade_share": g.apply(lambda x: x.loc[x.source.eq("Trade"), "cap"].sum()
                                   / x.cap.sum()),
        "n_acq_stars": g.apply(lambda x: int((x.acq & x.cap_share.ge(STAR_CAP)).sum())),
        "n_hg_stars": g.apply(lambda x: int((x.homegrown.eq(True)
                                             & x.cap_share.ge(STAR_CAP)).sum())),
    }).reset_index()
    cap["cap_cov"] = cap.cap_obs / cap.season.map(league_cap)

    t = pd.read_csv(RAW / "trades.csv")
    t["gave"], t["received"] = norm_team(t.gave), norm_team(t.received)
    vet = t[t.pick_season.isna() & t.pfr_id.notna()]
    vet_in = set(zip(vet.trade_id, vet.received))  # (trade, team that got a vet)
    picks = t[t.pick_round.isin([1, 2])]
    picks = picks[[k in vet_in for k in zip(picks.trade_id, picks.gave)]]
    pv = picks.assign(r1=picks.pick_round.eq(1).astype(int)).groupby(
        ["gave", "season"]).agg(picks_r1r2_for_vets=("pick_round", "size"),
                                picks_r1_for_vets=("r1", "sum")).reset_index()
    pv = pv.rename(columns={"gave": "team"})

    ts = ts.drop(columns=[c for c in cap.columns.tolist() + pv.columns.tolist()
                          if c in ts.columns and c not in ("team", "season")])
    ts = ts.merge(cap, on=["team", "season"], how="left").merge(
        pv, on=["team", "season"], how="left")
    for c in ["picks_r1r2_for_vets", "picks_r1_for_vets"]:
        ts[c] = ts[c].fillna(0)
    ts = ts.sort_values(["team", "season"])
    # Rolling 3-season stock of R1/R2 picks spent on veterans
    ts["picks_vets_3yr"] = ts.groupby("team").picks_r1r2_for_vets.transform(
        lambda x: x.rolling(3, min_periods=1).sum())
    ts.to_parquet(CLEAN / "team_season.parquet", index=False)
    log.info("league cap by season:\n%s", league_cap.round(1).to_dict())
    log.info("cap coverage by season:\n%s",
             ts.groupby("season").cap_cov.median().round(2).to_dict())


if __name__ == "__main__":
    main()

"""Classify every player-team-season by acquisition source.

Output: data/clean/player_team_season.parquet with one row per
(pfr_id, team, season) appearing in PFR snap counts, carrying
acquisition source (Drafted / UDFA / Trade / FA), tenure, age, and
snap totals.

Definitions
- Drafted: player's draft franchise equals the team.
- UDFA: undrafted, and the team is the first NFL franchise the player appears
  with in any roster (2002+ season rosters, 2011+ weekly rosters).
- Trade: not homegrown, and a player-for-asset trade row sends him to
  the team during the current stint (season >= stint start - 1).
- FA: everything else (free agents, waiver claims, re-signings of
  players first acquired elsewhere).
"""
import logging

import numpy as np
import pandas as pd

from common import (RAW, CLEAN, POS_GROUP, norm_team, read_years,
                    age_on_sept1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_crosswalk() -> pd.DataFrame:
    """pfr_id -> gsis_id, birth_date, draft info; patched from rosters."""
    p = pd.read_parquet(RAW / "players.parquet")
    p = p[["pfr_id", "gsis_id", "display_name", "birth_date", "position",
           "draft_team", "draft_year", "draft_round", "draft_pick",
           "rookie_season"]].dropna(subset=["pfr_id"])
    p = p.drop_duplicates("pfr_id")
    p["draft_team"] = norm_team(p["draft_team"])
    # Patch players absent from players.parquet using weekly rosters.
    wr = read_years("roster_weekly", range(2011, 2027))
    wr = wr.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id", keep="last")
    extra = wr[~wr.pfr_id.isin(p.pfr_id)].rename(
        columns={"full_name": "display_name", "draft_club": "draft_team",
                 "draft_number": "draft_pick", "rookie_year": "rookie_season"})
    extra["draft_team"] = norm_team(extra["draft_team"])
    extra["draft_year"] = np.where(extra.draft_team.notna(),
                                   extra.rookie_season, np.nan)
    cols = ["pfr_id", "gsis_id", "display_name", "birth_date", "position",
            "draft_team", "draft_year", "draft_pick", "rookie_season"]
    return pd.concat([p, extra[cols]], ignore_index=True)


def load_presence(xw: pd.DataFrame) -> pd.DataFrame:
    """(gsis_id, team, season, first_week) for every roster appearance."""
    sr = read_years("roster", range(2002, 2011))[["gsis_id", "team", "season"]]
    sr["week"] = 0
    wr = read_years("roster_weekly", range(2011, 2027))
    # REG only: postseason rows (esp. Super Bowls) carry mislabeled teams.
    wr = wr[~wr.status.isin(["UFA", "RET", "NWT"]) & wr.game_type.eq("REG")]
    wr = wr[["gsis_id", "team", "season", "week"]]
    sc = read_years("snap_counts", range(2012, 2027))
    sc = sc[sc.game_type.eq("REG")]
    sc = sc.merge(xw[["pfr_id", "gsis_id"]], left_on="pfr_player_id",
                  right_on="pfr_id", how="left")[["gsis_id", "team", "season", "week"]]
    pres = pd.concat([sr, wr, sc], ignore_index=True).dropna(subset=["gsis_id"])
    pres["team"] = norm_team(pres["team"])
    return (pres.groupby(["gsis_id", "team", "season"], as_index=False)
            .week.min().rename(columns={"week": "first_week"}))


def first_team(pres: pd.DataFrame) -> pd.Series:
    """First franchise each gsis_id appears with (earliest season, week)."""
    first = pres.sort_values(["season", "first_week"]).drop_duplicates("gsis_id")
    return first.set_index("gsis_id")[["team", "season"]].rename(
        columns={"team": "first_team", "season": "first_season"})


def stint_start(pres: pd.DataFrame) -> pd.DataFrame:
    """For each (gsis, team, season), first season of the current stint.

    A stint breaks only when the player appears with another franchise in
    between; absent seasons (injury, out of league) do not break it, since
    roster files sometimes omit injured players.
    """
    d = pres.sort_values(["gsis_id", "season", "first_week"]).reset_index(drop=True)
    starts = np.empty(len(d))
    gsis, team, season = d.gsis_id.to_numpy(), d.team.to_numpy(), d.season.to_numpy()
    cur_team, cur_start, prev_g = None, None, None
    for i in range(len(d)):
        if gsis[i] != prev_g:
            prev_g, cur_team, cur_start = gsis[i], None, None
        if team[i] != cur_team:
            cur_team, cur_start = team[i], season[i]
        starts[i] = cur_start
    d["stint_start"] = starts
    # Same franchise twice in one season (rare) -> keep earliest start.
    return d.groupby(["gsis_id", "team", "season"], as_index=False).stint_start.min()


def trade_arrivals() -> pd.DataFrame:
    """Player rows from trades: (pfr_id, team received, trade season)."""
    t = pd.read_csv(RAW / "trades.csv")
    t = t[t.pick_season.isna() & t.pfr_id.notna()]
    t["team"] = norm_team(t["received"])
    return t[["pfr_id", "team", "season"]].rename(columns={"season": "trade_season"})


def main() -> None:
    xw = load_crosswalk()
    pres = load_presence(xw)
    ft = first_team(pres)
    st = stint_start(pres)

    sc = read_years("snap_counts", range(2012, 2027))
    sc = sc[sc.game_type == "REG"].copy()
    sc["team"] = norm_team(sc["team"])
    sc["pos_group"] = sc.position.map(POS_GROUP)
    agg = (sc.groupby(["pfr_player_id", "team", "season"], as_index=False)
           .agg(off=("offense_snaps", "sum"), dfn=("defense_snaps", "sum"),
                st=("st_snaps", "sum"), games=("game_id", "nunique"),
                snap_pos=("position", lambda x: x.mode().iat[0]),
                player=("player", "first"))
           .rename(columns={"pfr_player_id": "pfr_id"}))
    agg["pos_group"] = agg.snap_pos.map(POS_GROUP)

    df = agg.merge(xw, on="pfr_id", how="left")
    # Resolve generic "DB" to CB/S from the player-level position.
    db = df.snap_pos.eq("DB")
    df.loc[db, "pos_group"] = np.where(df.loc[db, "position"].eq("CB"), "CB", "S")
    unresolved = df.pos_group.isna()
    log.info("unmapped positions: %s", df.loc[unresolved, "snap_pos"].value_counts().to_dict())

    df = df.merge(ft, left_on="gsis_id", right_index=True, how="left")
    df = df.merge(st, on=["gsis_id", "team", "season"], how="left")
    df["stint_start"] = df.stint_start.fillna(df.season)

    tr = trade_arrivals().merge(df[["pfr_id", "team", "season", "stint_start"]],
                                on=["pfr_id", "team"])
    tr = tr[(tr.trade_season >= tr.stint_start - 1) & (tr.trade_season <= tr.season)]
    traded = set(zip(tr.pfr_id, tr.team, tr.season))

    drafted = df.draft_team.notna() & df.draft_year.notna()
    df["source"] = "FA"
    is_trade = [k in traded for k in zip(df.pfr_id, df.team, df.season)]
    df.loc[is_trade, "source"] = "Trade"
    df.loc[~drafted & df.first_team.eq(df.team), "source"] = "UDFA"
    df.loc[drafted & df.draft_team.eq(df.team), "source"] = "Drafted"
    # Homegrown only if the current stint began at entry: a drafted/UDFA
    # player who left and later re-signed is a re-acquisition (FA).
    entry = np.where(drafted, df.draft_year, df.first_season)
    returnee = (df.source.isin(["Drafted", "UDFA"]) & (df.stint_start > entry + 1)
                & (df.stint_start > 2002))  # 2002 = roster data start (censored)
    df.loc[returnee, "source"] = "FA"
    df["returnee"] = returnee
    log.info("returnees reclassified to FA: %d", int(returnee.sum()))
    # Undrafted with unknown first team (career began pre-2002): leave FA,
    # but flag for sensitivity checks.
    df["first_team_known"] = df.first_season.gt(2002) | drafted
    df["homegrown"] = df.source.isin(["Drafted", "UDFA"])
    df["tenure"] = df.season - df.stint_start + 1
    df["age"] = age_on_sept1(df.birth_date, df.season)
    df["exp"] = df.season - df.rookie_season
    df["draft_r1"] = df.source.eq("Drafted") & df.draft_pick.le(32)

    out = CLEAN / "player_team_season.parquet"
    df.to_parquet(out, index=False)
    log.info("rows %d | missing age %.4f | missing gsis %.4f", len(df),
             df.age.isna().mean(), df.gsis_id.isna().mean())
    log.info("source mix (all seasons, off+def snaps):\n%s",
             df.assign(s=df.off + df.dfn).groupby("source").s.sum()
             .pipe(lambda x: x / x.sum()).round(3))


if __name__ == "__main__":
    main()

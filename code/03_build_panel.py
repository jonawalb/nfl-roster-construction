"""Build the team-season panel: construction, age, special teams, outcomes.

Inputs: data/clean/player_team_season.parquet, raw roster/pbp/games files.
Output: data/clean/team_season.parquet (2013-2026; 2026 partial).
"""
import logging

import numpy as np
import pandas as pd

from common import (RAW, CLEAN, POS_GROUP, OFFENSE, DEFENSE, SPECIALISTS,
                    norm_team, read_years, age_on_sept1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
KEYS = ["team", "season"]


def wavg(df: pd.DataFrame, val: str, w: str) -> float:
    m = df[val].notna() & df[w].gt(0)
    return np.average(df.loc[m, val], weights=df.loc[m, w]) if m.any() else np.nan


def construction(p: pd.DataFrame, prev_presence: set) -> pd.DataFrame:
    """Snap-weighted construction and age measures per team-season."""
    p = p.copy()
    p["od"] = p.off + p.dfn
    p["returning"] = [(g, t, s - 1) in prev_presence
                      for g, t, s in zip(p.gsis_id, p.team, p.season)]
    rows = []
    for (team, season), g in p.groupby(KEYS):
        r = {"team": team, "season": season}
        tot = g.od.sum()
        for src in ["Drafted", "UDFA", "FA", "Trade"]:
            r[f"sh_{src.lower()}"] = g.loc[g.source == src, "od"].sum() / tot
        r["hg_share"] = r["sh_drafted"] + r["sh_udfa"]
        r["hg_share_noqb"] = (g.loc[g.homegrown & g.pos_group.ne("QB"), "od"].sum()
                              / g.loc[g.pos_group.ne("QB"), "od"].sum())
        r["r1_share"] = g.loc[g.draft_r1, "od"].sum() / tot
        r["continuity"] = g.loc[g.returning, "od"].sum() / tot
        r["tenure_w"] = wavg(g, "tenure", "od")
        r["age_w"] = wavg(g, "age", "od")
        r["exp_w"] = wavg(g, "exp", "od")
        r["age_off_w"] = wavg(g, "age", "off")
        r["age_def_w"] = wavg(g, "age", "dfn")
        r["hg_off"] = g.loc[g.homegrown, "off"].sum() / g.off.sum()
        r["hg_def"] = g.loc[g.homegrown, "dfn"].sum() / g.dfn.sum()
        cov = g[~g.pos_group.isin(SPECIALISTS)]
        r["age_st_w"] = wavg(cov, "age", "st")  # ST coverage/return units
        r["hg_st"] = cov.loc[cov.homegrown, "st"].sum() / cov.st.sum()
        for pg in OFFENSE + DEFENSE:
            gg = g[g.pos_group == pg]
            w = "off" if pg in OFFENSE else "dfn"
            r[f"age_w_{pg}"] = wavg(gg, "age", w)
            r[f"hg_{pg}"] = gg.loc[gg.homegrown, w].sum() / gg[w].sum() if gg[w].sum() else np.nan
            r[f"cont_{pg}"] = gg.loc[gg.returning, w].sum() / gg[w].sum() if gg[w].sum() else np.nan
        # QB: primary passer (most offensive snaps among QBs)
        qb = g[g.pos_group == "QB"].sort_values("off", ascending=False).head(1)
        r["qb_name"] = qb.display_name.iat[0] if len(qb) else None
        r["qb_source"] = qb.source.iat[0] if len(qb) else None
        r["qb_age"] = qb.age.iat[0] if len(qb) else np.nan
        for sp in SPECIALISTS:
            s = g[g.pos_group == sp].sort_values("st", ascending=False).head(1)
            r[f"age_{sp}"] = s.age.iat[0] if len(s) else np.nan
            r[f"hg_{sp}"] = float(s.homegrown.iat[0]) if len(s) else np.nan
        r["n_players"] = int((g.od > 0).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def roster_age(years: range) -> pd.DataFrame:
    """Unweighted age of the game-week active roster (status ACT, REG).

    Each player-week counts once, so this is the average age of the
    players on the active list across the season.
    """
    wr = read_years("roster_weekly", years)
    wr = wr[wr.game_type.eq("REG") & wr.status.eq("ACT")].copy()
    wr["team"] = norm_team(wr.team)
    pos = wr.depth_chart_position.fillna(wr.position)
    wr["pos_group"] = pos.map(POS_GROUP)
    wr.loc[pos.eq("DB"), "pos_group"] = "S"
    wr.loc[pos.isin(["OLB", "ILB", "MLB"]), "pos_group"] = "LB"
    wr["age"] = age_on_sept1(wr.birth_date, wr.season)
    wr["unit"] = np.select([wr.pos_group.isin(OFFENSE), wr.pos_group.isin(DEFENSE)],
                           ["off", "def"], "st")
    out = wr.groupby(KEYS).age.mean().rename("age_roster").to_frame()
    out = out.join(wr.pivot_table(index=KEYS, columns="unit", values="age",
                                  aggfunc="mean").add_prefix("age_roster_"))
    out = out.join(wr.pivot_table(index=KEYS, columns="pos_group", values="age",
                                  aggfunc="mean").add_prefix("age_roster_"))
    return out.reset_index()


def pbp_outcomes(years: range) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team-season EPA/success, ST EPA, and primary returner IDs."""
    cols = ["season", "week", "season_type", "posteam", "defteam", "play_type",
            "epa", "success", "pass", "rush", "special", "qb_kneel", "qb_spike",
            "kickoff_returner_player_id", "punt_returner_player_id"]
    frames, rets = [], []
    for y in years:
        d = pd.read_parquet(RAW / f"pbp_{y}.parquet", columns=cols)
        d = d[d.season_type.eq("REG")]
        d["posteam"], d["defteam"] = norm_team(d.posteam), norm_team(d.defteam)
        sc = d[(d["pass"].eq(1) | d["rush"].eq(1)) & d.epa.notna()
               & d.qb_kneel.ne(1) & d.qb_spike.ne(1)]
        off = sc.groupby(["season", "posteam"]).agg(
            off_epa=("epa", "mean"), off_sr=("success", "mean"),
            off_pass_epa=("epa", lambda x: x[sc.loc[x.index, "pass"].eq(1)].mean()),
            off_rush_epa=("epa", lambda x: x[sc.loc[x.index, "rush"].eq(1)].mean()),
            off_plays=("epa", "size"))
        dfn = sc.groupby(["season", "defteam"]).agg(
            def_epa=("epa", "mean"), def_sr=("success", "mean"))
        st = d[d.special.eq(1) & d.epa.notna()]
        st_net = (st.groupby(["season", "posteam"]).epa.sum()
                  .sub(st.groupby(["season", "defteam"]).epa.sum(), fill_value=0))
        for x in (off, dfn, st_net):
            x.index.names = ["season", "team"]
        f = off.join(dfn).join(st_net.rename("st_epa_total"))
        frames.append(f.reset_index())
        for kind, col in [("kr", "kickoff_returner_player_id"),
                          ("pr", "punt_returner_player_id")]:
            r = d.dropna(subset=[col]).groupby(["season", "posteam", col]).size()
            r = r.rename("n").reset_index().sort_values("n", ascending=False)
            r = r.drop_duplicates(["season", "posteam"]).rename(
                columns={"posteam": "team", col: "gsis_id"})
            r["kind"] = kind
            rets.append(r)
    return pd.concat(frames, ignore_index=True), pd.concat(rets, ignore_index=True)


def game_outcomes() -> pd.DataFrame:
    """Wins, point differential, market-implied wins, ATS, playoffs."""
    g = pd.read_csv(RAW / "games.csv")
    g = g[g.season.ge(2013) & g.result.notna()]
    g["home_team"], g["away_team"] = norm_team(g.home_team), norm_team(g.away_team)

    def implied(ml_home: pd.Series, ml_away: pd.Series) -> pd.Series:
        def p(ml):
            return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
        ph, pa = p(ml_home), p(ml_away)
        return ph / (ph + pa)  # remove vig

    g["p_home"] = implied(g.home_moneyline, g.away_moneyline)
    home = pd.DataFrame({"team": g.home_team, "season": g.season, "week": g.week,
                         "game_type": g.game_type, "margin": g.result,
                         "p_win": g.p_home, "ats": g.result - g.spread_line})
    away = pd.DataFrame({"team": g.away_team, "season": g.season, "week": g.week,
                         "game_type": g.game_type, "margin": -g.result,
                         "p_win": 1 - g.p_home, "ats": -(g.result - g.spread_line)})
    tg = pd.concat([home, away], ignore_index=True)
    tg["win"] = np.select([tg.margin > 0, tg.margin == 0], [1.0, 0.5], 0.0)
    tg.to_parquet(CLEAN / "team_games.parquet", index=False)
    reg = tg[tg.game_type == "REG"]
    out = reg.groupby(KEYS).agg(games=("win", "size"), wins=("win", "sum"),
                                pd=("margin", "sum"), exp_wins=("p_win", "sum"),
                                ats_margin=("ats", "mean"),
                                cover_rate=("ats", lambda x: (x > 0).sum() / (x != 0).sum()))
    out["win_pct"] = out.wins / out.games
    out["pd_pg"] = out.pd / out.games
    out["wins_over_exp"] = out.wins - out.exp_wins
    post = tg[tg.game_type != "REG"]
    out["playoffs"] = out.index.isin(post.set_index(KEYS).index).astype(int)
    out["playoff_wins"] = post.groupby(KEYS).win.sum().reindex(out.index).fillna(0)
    sb = post[(post.game_type == "SB") & (post.win == 1)].set_index(KEYS).index
    out["sb_win"] = out.index.isin(sb).astype(int)
    return out.reset_index()


def main() -> None:
    p = pd.read_parquet(CLEAN / "player_team_season.parquet")
    wr = read_years("roster_weekly", range(2011, 2027))
    wr = wr[wr.game_type.eq("REG")]
    prev = set(zip(wr.gsis_id, norm_team(wr.team), wr.season))
    con = construction(p[p.season.ge(2013)], prev)
    ra = roster_age(range(2013, 2027))
    pbp, rets = pbp_outcomes(range(2013, 2027))
    games = game_outcomes()

    # Returner ages and homegrown status
    pl = pd.read_parquet(RAW / "players.parquet")[["gsis_id", "birth_date"]]
    rets = rets.merge(pl, on="gsis_id", how="left")
    rets["team"] = norm_team(rets.team)
    rets["age"] = age_on_sept1(rets.birth_date, rets.season)
    hg = p[["gsis_id", "team", "season", "homegrown"]]
    rets = rets.merge(hg, on=["gsis_id", "team", "season"], how="left")
    rw = rets.pivot_table(index=KEYS, columns="kind", values=["age", "homegrown"],
                          aggfunc="first")
    rw.columns = [f"{a}_{k}" for a, k in rw.columns]

    df = (con.merge(ra, on=KEYS, how="left").merge(pbp, on=KEYS, how="left")
          .merge(games, on=KEYS, how="left").merge(rw.reset_index(), on=KEYS, how="left"))
    df["st_epa_pg"] = df.st_epa_total / df.games
    df["net_epa"] = df.off_epa - df.def_epa
    df = df.sort_values(KEYS)
    for c in ["win_pct", "net_epa", "hg_share", "pd_pg"]:
        df[f"{c}_lag"] = df.groupby("team")[c].shift(1)
        df[f"{c}_lead"] = df.groupby("team")[c].shift(-1)
    df.to_parquet(CLEAN / "team_season.parquet", index=False)
    log.info("panel %s; seasons %s", df.shape, df.season.unique())
    log.info("missing share in key vars:\n%s",
             df[df.season <= 2025][["hg_share", "age_w", "age_roster", "off_epa",
                                     "win_pct", "exp_wins", "age_K", "age_kr"]]
             .isna().mean().round(3))


if __name__ == "__main__":
    main()

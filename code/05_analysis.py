"""All estimates reported in the paper. Writes CSV tables and results.json.

Sample: 416 team-seasons, 2013-2025 regular seasons. SEs clustered by team.
"""
import json
import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import CLEAN, OUT, POS_GROUP, OFFENSE, DEFENSE, norm_team, read_years

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
R: dict = {}


def fit(f: str, data: pd.DataFrame, cluster: str = "team"):
    """OLS with cluster-robust SEs; drops rows with missing values first."""
    base = smf.ols(f, data=data).fit()
    g = data.loc[base.model.data.row_labels, cluster]
    return smf.ols(f, data=data).fit(cov_type="cluster", cov_kwds={"groups": g})


def row(m, var: str) -> dict:
    return {"b": m.params[var], "se": m.bse[var], "p": m.pvalues[var],
            "n": int(m.nobs), "r2": m.rsquared}


def table(models: dict, vars_: list[str]) -> pd.DataFrame:
    out = []
    for name, m in models.items():
        r = {"model": name, "N": int(m.nobs), "R2": round(m.rsquared, 3)}
        for v in vars_:
            if v in m.params:
                r[v] = f"{m.params[v]:.3f} ({m.bse[v]:.3f})" + stars(m.pvalues[v])
        out.append(r)
    return pd.DataFrame(out)


def stars(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def ages(d: pd.DataFrame) -> None:
    groups = OFFENSE + DEFENSE + ["K", "P", "LS"]
    rows = []
    for g in groups:
        r = {"group": g, "roster_age": d[f"age_roster_{g}"].mean()}
        r["snap_age"] = d[f"age_w_{g}"].mean() if f"age_w_{g}" in d else d[f"age_{g}"].mean()
        r["sd_across_teams"] = (d[f"age_w_{g}"] if f"age_w_{g}" in d else d[f"age_{g}"]).std()
        rows.append(r)
    for k, lab in [("kr", "KR (primary)"), ("pr", "PR (primary)")]:
        rows.append({"group": lab, "roster_age": np.nan, "snap_age": d[f"age_{k}"].mean(),
                     "sd_across_teams": d[f"age_{k}"].std()})
    t = pd.DataFrame(rows).round(2)
    t.to_csv(OUT / "t1_age_by_position.csv", index=False)
    units = d.groupby("season")[["age_roster", "age_w", "age_off_w", "age_def_w",
                                 "age_st_w", "age_roster_st"]].mean().round(2)
    units.to_csv(OUT / "t1b_age_by_season.csv")
    R["age"] = {"roster_mean": d.age_roster.mean(), "snap_mean": d.age_w.mean(),
                "off_snap": d.age_off_w.mean(), "def_snap": d.age_def_w.mean(),
                "st_cov_snap": d.age_st_w.mean(),
                "trend_snap_per_yr": np.polyfit(units.index, units.age_w, 1)[0],
                "youngest": d.loc[d.age_w.idxmin(), ["team", "season", "age_w"]].tolist(),
                "oldest": d.loc[d.age_w.idxmax(), ["team", "season", "age_w"]].tolist(),
                "by_group": t.set_index("group").snap_age.to_dict()}


def main_models(d: pd.DataFrame) -> None:
    outs = ["win_pct", "pd_pg", "net_epa", "off_epa", "def_epa", "off_sr", "def_sr",
            "st_epa_pg"]
    res, models = {}, {}
    for y in outs:
        m1 = fit(f"{y} ~ hg_share + age_w + C(season)", d)
        m2 = fit(f"{y} ~ hg_share + age_w + C(season) + C(team)", d)
        models[f"{y} (season FE)"], models[f"{y} (team+season FE)"] = m1, m2
        res[y] = {"between": row(m1, "hg_share"), "within": row(m2, "hg_share"),
                  "age_between": row(m1, "age_w")}
    table(models, ["hg_share", "age_w"]).to_csv(OUT / "t3_main.csv", index=False)
    R["main"] = res
    alt = {}
    for x in ["continuity", "cap_acq_share", "hg_share_noqb", "r1_share", "tenure_w"]:
        alt[x] = row(fit(f"net_epa ~ {x} + age_w + C(season)", d), x)
        alt[x + "_win"] = row(fit(f"win_pct ~ {x} + age_w + C(season)", d), x)
    m = fit("win_pct ~ n_acq_stars + n_hg_stars + C(season)", d)
    alt["acq_stars_win"], alt["hg_stars_win"] = row(m, "n_acq_stars"), row(m, "n_hg_stars")
    m = fit("net_epa ~ n_acq_stars + n_hg_stars + C(season)", d)
    alt["acq_stars_epa"], alt["hg_stars_epa"] = row(m, "n_acq_stars"), row(m, "n_hg_stars")
    wt = m.t_test("n_hg_stars - n_acq_stars = 0")
    alt["stars_diff_epa"] = {"b": float(wt.effect[0]), "se": float(wt.sd[0][0]), "p": float(wt.pvalue)}
    m = fit("win_pct ~ n_acq_stars + n_hg_stars + C(season)", d)
    wt = m.t_test("n_hg_stars - n_acq_stars = 0")
    alt["stars_diff_win"] = {"b": float(wt.effect[0]), "se": float(wt.sd[0][0]), "p": float(wt.pvalue)}
    R["alt"] = alt
    R["sd"] = {v: d[v].std() for v in ["hg_share", "continuity", "cap_acq_share",
                                        "age_w", "win_pct", "net_epa"]}
    R["sd"]["hg_share_within"] = (d.hg_share - d.groupby("team").hg_share.transform("mean")).std()


def shape(d: pd.DataFrame) -> None:
    d = d.copy()
    d["hg_c"] = d.hg_share - d.hg_share.mean()
    m = fit("win_pct ~ hg_c + I(hg_c**2) + age_w + C(season)", d)
    q = fit("playoff_wins ~ hg_c + I(hg_c**2) + age_w + C(season)", d)
    R["quadratic"] = {"win_lin": row(m, "hg_c"), "win_sq": row(m, "I(hg_c ** 2)"),
                      "pw_lin": row(q, "hg_c"), "pw_sq": row(q, "I(hg_c ** 2)"),
                      "win_peak_hg": d.hg_share.mean() - m.params["hg_c"] / (2 * m.params["I(hg_c ** 2)"])}
    d["q"] = pd.qcut(d.hg_share, 4, labels=["Q1 (least homegrown)", "Q2", "Q3",
                                             "Q4 (most homegrown)"])
    qt = d.groupby("q", observed=True).agg(
        hg_range=("hg_share", lambda x: f"{x.min():.2f}-{x.max():.2f}"),
        win_pct=("win_pct", "mean"), win_sd=("win_pct", "std"),
        net_epa=("net_epa", "mean"), playoff_rate=("playoffs", "mean"),
        playoff_wins=("playoff_wins", "mean"), sb_titles=("sb_win", "sum"),
        snap_age=("age_w", "mean"), n=("win_pct", "size")).round(3)
    qt.to_csv(OUT / "t6_quartiles.csv")
    R["quartiles"] = qt.reset_index().to_dict("records")
    R["levene_p"] = stats.levene(*[g.win_pct for _, g in d.groupby("q", observed=True)]).pvalue
    d["hg_pct"] = d.groupby("season").hg_share.rank(pct=True)
    ch = d[d.sb_win == 1][["season", "team", "hg_share", "hg_pct", "cap_acq_share",
                           "n_acq_stars", "picks_vets_3yr", "age_w"]]
    ch.round(3).to_csv(OUT / "t7_champions.csv", index=False)
    R["champs"] = {"mean_pct": ch.hg_pct.mean(), "n_in_middle_band": int(ch.hg_pct.between(.5, .85).sum()),
                   "n": len(ch), "list": ch.round(3).to_dict("records")}
    # Typology: base (homegrown snaps) x apex (acquired cap share), split at medians
    mh, ma = d.hg_share.median(), d.cap_acq_share.median()
    d["type"] = np.select(
        [(d.hg_share >= mh) & (d.cap_acq_share < ma), (d.hg_share >= mh) & (d.cap_acq_share >= ma),
         (d.hg_share < mh) & (d.cap_acq_share >= ma)],
        ["Grown (homegrown base, homegrown stars)", "Stars-and-scrubs (homegrown base, bought stars)",
         "Mercenary (bought base, bought stars)"], "Churn (bought base, cheap stars)")
    ty = d.groupby("type").agg(n=("win_pct", "size"), win_pct=("win_pct", "mean"),
                               net_epa=("net_epa", "mean"), playoff_rate=("playoffs", "mean"),
                               playoff_wins=("playoff_wins", "mean"), sb=("sb_win", "sum"),
                               win_next=("win_pct_lead", "mean")).round(3)
    ty.to_csv(OUT / "t8_typology.csv")
    R["typology"] = ty.reset_index().to_dict("records")
    R["typology_cuts"] = {"hg_median": mh, "capacq_median": ma}
    m = fit("win_pct ~ hg_share * cap_acq_share + age_w + C(season)", d)
    R["interaction"] = {k: row(m, k) for k in ["hg_share", "cap_acq_share", "hg_share:cap_acq_share"]}
    d[["team", "season", "type"]].to_csv(OUT / "typology_assignments.csv", index=False)
    po = d[d.playoffs == 1].assign(ss=lambda x: x.type.str.startswith("Stars").astype(int))
    R["ss_playoff_gap"] = row(fit("playoff_wins ~ ss + win_pct + C(season)", po), "ss")
    R["ss_playoff_gap"]["n_ss"] = int(po.ss.sum())
    R["ss_titles"] = d[(d.sb_win == 1) & d.type.str.startswith("Stars")][["team", "season"]].values.tolist()
    R["cap_cov_season_range"] = d.groupby("season").cap_cov.mean().agg(["min", "max"]).tolist()


def dynamics(d: pd.DataFrame) -> None:
    res, models = {}, {}
    for x in ["hg_share", "continuity", "cap_acq_share", "age_w"]:
        for y in ["win_pct_lead", "net_epa_lead"]:
            ctrl = "win_pct + net_epa + C(season)" + ("" if x == "age_w" else " + age_w")
            m = fit(f"{y} ~ {x} + {ctrl}", d)
            res[f"{x}->{y}"] = row(m, x)
            models[f"{y} ~ {x}"] = m
    table(models, ["hg_share", "continuity", "cap_acq_share", "age_w", "win_pct", "net_epa"]
          ).to_csv(OUT / "t4b_next_season.csv", index=False)
    m = fit("win_pct ~ picks_r1r2_for_vets + win_pct_lag + C(season)", d)
    res["picks_now"] = row(m, "picks_r1r2_for_vets")
    d2 = d.assign(dwin=d.win_pct_lead - d.win_pct)
    res["picks_next_change"] = row(fit("dwin ~ picks_r1r2_for_vets + win_pct + C(season)", d2),
                                   "picks_r1r2_for_vets")
    # Age-decline: next-season change in net EPA by snap-age tercile
    d2["age_t"] = pd.qcut(d2.age_w, 3, labels=["young", "mid", "old"])
    d2["dnet"] = d2.net_epa_lead - d2.net_epa
    res["age_terciles_dnet"] = d2.groupby("age_t", observed=True).dnet.mean().to_dict()
    R["dynamics"] = res


def early_late() -> None:
    """Weeks 1-4 construction predicting weeks 5-18 outcomes (pre-injury-churn)."""
    p = pd.read_parquet(CLEAN / "player_team_season.parquet")[
        ["pfr_id", "team", "season", "homegrown", "age"]]
    sc = read_years("snap_counts", range(2013, 2026))
    sc = sc[sc.game_type == "REG"].copy()
    sc["team"] = norm_team(sc.team)
    sc = sc.merge(p, left_on=["pfr_player_id", "team", "season"],
                  right_on=["pfr_id", "team", "season"], how="left")
    sc["od"] = sc.offense_snaps + sc.defense_snaps
    sc["hg_od"] = sc.od * sc.homegrown.fillna(False).astype(float)
    wk = sc.groupby(["team", "season", "week"]).agg(od=("od", "sum"), hg_od=("hg_od", "sum")).reset_index()
    wk["hg"] = wk.hg_od / wk.od
    tg = pd.read_parquet(CLEAN / "team_games.parquet")
    tg = tg[tg.game_type == "REG"].merge(wk[["team", "season", "week", "hg", "hg_od", "od"]],
                                         on=["team", "season", "week"])
    tg = tg.sort_values(["team", "season", "week"])
    # Pre-game construction: cumulative homegrown share in prior games this season
    grp = tg.groupby(["team", "season"])
    tg["hg_prior"] = grp.hg_od.cumsum().sub(tg.hg_od) / grp.od.cumsum().sub(tg.od)
    tg["games_prior"] = grp.cumcount()
    e = tg[tg.week <= 4].groupby(["team", "season"]).agg(
        hg_e=("hg", "mean"), win_e=("win", "mean"), mar_e=("margin", "mean"))
    l = tg[tg.week >= 5].groupby(["team", "season"]).agg(
        win_l=("win", "mean"), mar_l=("margin", "mean"), pwin_l=("p_win", "mean"),
        ats_l=("ats", "mean"))
    el = e.join(l).reset_index()
    el["woe_l"] = el.win_l - el.pwin_l
    res = {}
    res["win_l"] = row(fit("win_l ~ hg_e + win_e + mar_e + C(season)", el), "hg_e")
    res["mar_l"] = row(fit("mar_l ~ hg_e + mar_e + C(season)", el), "hg_e")
    res["woe_l"] = row(fit("woe_l ~ hg_e + mar_e + C(season)", el), "hg_e")
    res["sd_hg_e"] = el.hg_e.std()
    # Game-level market test: prior-games construction vs closing line
    g = tg[(tg.games_prior >= 3) & tg.ats.notna()].copy()
    g["ts"] = g.team + g.season.astype(str)
    g["cover"] = (g.ats > 0).astype(float)
    g["hg_dm"] = g.hg_prior - g.groupby("season").hg_prior.transform("mean")
    res["ats_game"] = row(fit("ats ~ hg_dm + C(season)", g, cluster="ts"), "hg_dm")
    res["cover_game"] = row(fit("cover ~ hg_dm + C(season)", g[g.ats != 0], cluster="ts"), "hg_dm")
    res["woe_game"] = row(fit("I(win - p_win) ~ hg_dm + C(season)", g, cluster="ts"), "hg_dm")
    g["hq"] = pd.qcut(g.hg_dm, 5, labels=False)
    res["cover_by_quintile"] = g[g.ats != 0].groupby("hq").cover.mean().round(3).to_dict()
    res["n_games"] = int(len(g))
    R["early_late"] = res


def positions(d: pd.DataFrame) -> None:
    rows = []
    for pg in OFFENSE + DEFENSE:
        for kind in ["hg", "cont"]:
            m = fit(f"net_epa ~ {kind}_{pg} + age_w + C(season)", d)
            unit = "off_epa" if pg in OFFENSE else "def_epa"
            mu = fit(f"{unit} ~ {kind}_{pg} + age_w + C(season)", d)
            r = row(m, f"{kind}_{pg}")
            ru = row(mu, f"{kind}_{pg}")
            rows.append({"group": pg, "measure": kind, "b_net": r["b"], "se_net": r["se"],
                         "p_net": r["p"], "b_unit": ru["b"], "se_unit": ru["se"], "p_unit": ru["p"],
                         "sd_x": d[f"{kind}_{pg}"].std()})
    t = pd.DataFrame(rows)
    t["b_net_per_sd"] = t.b_net * t.sd_x
    t["se_net_per_sd"] = t.se_net * t.sd_x
    t.round(4).to_csv(OUT / "t5_positions.csv", index=False)
    R["positions"] = t.round(4).to_dict("records")


def special_teams(d: pd.DataFrame) -> None:
    m = fit("st_epa_pg ~ hg_share + age_w + C(season)", d)
    m2 = fit("st_epa_pg ~ hg_st + age_st_w + age_K + age_P + C(season)", d)
    m3 = fit("win_pct ~ hg_share + age_w + age_K + age_P + age_LS + st_epa_pg + C(season)", d)
    R["st"] = {"placebo_hg": row(m, "hg_share"), "hg_st": row(m2, "hg_st"),
               "age_st": row(m2, "age_st_w"), "age_K": row(m2, "age_K"), "age_P": row(m2, "age_P"),
               "main_with_st_controls": row(m3, "hg_share"),
               "share_hg_K": d.hg_K.mean(), "share_hg_P": d.hg_P.mean(), "share_hg_LS": d.hg_LS.mean(),
               "share_hg_kr": d.homegrown_kr.astype(float).mean(),
               "hg_st_mean": d.hg_st.mean()}


def teams(d: pd.DataFrame) -> None:
    t = d.groupby("team").agg(hg_share=("hg_share", "mean"), continuity=("continuity", "mean"),
                              cap_acq_share=("cap_acq_share", "mean"),
                              picks_r1r2_for_vets=("picks_r1r2_for_vets", "sum"),
                              acq_stars=("n_acq_stars", "mean"), snap_age=("age_w", "mean"),
                              roster_age=("age_roster", "mean"), win_pct=("win_pct", "mean"),
                              net_epa=("net_epa", "mean"), playoff_wins=("playoff_wins", "sum")
                              ).sort_values("hg_share", ascending=False).round(3)
    t.to_csv(OUT / "t2_teams.csv")
    R["team_corr_hg_win"] = t.hg_share.corr(t.win_pct)
    R["team_corr_capacq_win"] = t.cap_acq_share.corr(t.win_pct)
    cases = {"LA": range(2017, 2026), "MIA": range(2022, 2026), "SEA": range(2013, 2026),
             "GB": range(2013, 2026)}
    cols = ["team", "season", "hg_share", "cap_acq_share", "n_acq_stars", "picks_r1r2_for_vets",
            "continuity", "age_w", "wins", "games", "net_epa", "playoff_wins", "qb_name"]
    c = pd.concat([d[(d.team == k) & d.season.isin(v)][cols] for k, v in cases.items()])
    c.round(3).to_csv(OUT / "t9_cases.csv", index=False)


def main() -> None:
    d = pd.read_parquet(CLEAN / "team_season.parquet")
    d = d[d.season.between(2013, 2025)].copy()
    # 2025 leads come from the in-progress 2026 season; treat them as missing.
    for c in [c for c in d.columns if c.endswith("_lead")]:
        d.loc[d.season == 2025, c] = np.nan
    log.info("N team-seasons: %d", len(d))
    R["n"] = len(d)
    ages(d); main_models(d); shape(d); dynamics(d); early_late(); positions(d)
    special_teams(d); teams(d)
    with open(OUT / "results.json", "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log.info("wrote results.json")


if __name__ == "__main__":
    main()

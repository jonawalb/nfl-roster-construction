"""Robustness suite. Reads the main estimates in output/results.json and does not modify them.

Writes output/robustness.json and output/r*_*.csv.
Every block first reproduces the main baseline so the checks are anchored.
"""
import json
import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from common import CLEAN, OUT, RAW, norm_team

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
R: dict = {}
RNG = np.random.default_rng(20261108)
B = 1999  # bootstrap / permutation draws


def fit(f, data, cluster="team"):
    base = smf.ols(f, data=data).fit()
    g = data.loc[base.model.data.row_labels, cluster]
    return smf.ols(f, data=data).fit(cov_type="cluster", cov_kwds={"groups": g})


def row(m, v):
    return {"b": float(m.params[v]), "se": float(m.bse[v]), "p": float(m.pvalues[v]),
            "n": int(m.nobs)}


# ---------- wild cluster bootstrap (restricted, Webb weights) ----------
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def _crv1_t(X, y, gidx, G, j):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, gidx, X * u[:, None])
    n, k = X.shape
    V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    return b[j] / np.sqrt(V[j, j])


def wild_p(f, data, var, cluster="team"):
    m = smf.ols(f, data=data).fit()
    X = pd.DataFrame(m.model.exog, columns=m.model.exog_names)
    y = m.model.endog
    j = list(X.columns).index(var)
    g = data.loc[m.model.data.row_labels, cluster].values
    codes, uniq = pd.factorize(g)
    G = len(uniq)
    Xv = X.values
    t0 = _crv1_t(Xv, y, codes, G, j)
    Xr = np.delete(Xv, j, axis=1)
    br = np.linalg.pinv(Xr) @ y
    fr, ur = Xr @ br, y - Xr @ br
    ts = np.empty(B)
    for i in range(B):
        w = RNG.choice(WEBB, size=G)[codes]
        ts[i] = _crv1_t(Xv, fr + ur * w, codes, G, j)
    return {"t": float(t0), "p_wild": float(np.mean(np.abs(ts) >= abs(t0)))}


# ---------- extra covariates ----------
def add_coaches(d):
    g = pd.read_csv(RAW / "games.csv")
    g = g[g.game_type == "REG"]
    h = g[["season", "home_team", "home_coach"]].rename(columns={"home_team": "team", "home_coach": "coach"})
    a = g[["season", "away_team", "away_coach"]].rename(columns={"away_team": "team", "away_coach": "coach"})
    c = pd.concat([h, a])
    c["team"] = norm_team(c.team)
    hc = c.groupby(["team", "season"]).coach.agg(lambda s: s.value_counts().index[0]).reset_index()
    hc = hc.sort_values(["team", "season"])
    # coach tenure with this franchise (consecutive seasons, counting from 2002 games data)
    hc["new"] = (hc.coach != hc.groupby("team").coach.shift()).astype(int)
    hc["spell"] = hc.groupby("team").new.cumsum()
    hc["coach_tenure"] = hc.groupby(["team", "spell"]).cumcount() + 1
    hc["coach_change"] = hc.new
    return d.merge(hc[["team", "season", "coach", "coach_tenure", "coach_change"]],
                   on=["team", "season"], how="left")


def add_rookie_split(d):
    """Split homegrown snaps into first-contract (exp<=3) and second-contract-plus."""
    p = pd.read_parquet(CLEAN / "player_team_season.parquet")
    p = p[p.season.between(2013, 2025)].copy()
    p["od"] = p.off.fillna(0) + p.dfn.fillna(0)
    hg = p.homegrown.fillna(False).astype(bool)
    p["hg_rook"] = p.od * (hg & (p.exp <= 3))
    p["hg_vet"] = p.od * (hg & (p.exp > 3))
    p["acq_rook"] = p.od * (~hg & (p.exp <= 3))
    p["rook_all"] = p.od * (p.exp <= 3)
    s = p.groupby(["team", "season"])[["od", "hg_rook", "hg_vet", "acq_rook", "rook_all"]].sum()
    for c in ["hg_rook", "hg_vet", "acq_rook", "rook_all"]:
        s[f"sh_{c}"] = s[c] / s.od
    chk = (s.sh_hg_rook + s.sh_hg_vet).reset_index()
    return d.merge(s[[c for c in s.columns if c.startswith("sh_")]].reset_index(),
                   on=["team", "season"], how="left"), chk


# ---------- blocks ----------
def baseline_check(d):
    base = json.load(open(OUT / "results.json"))
    out = {}
    for y in ["win_pct", "net_epa"]:
        b = row(fit(f"{y} ~ hg_share + age_w + C(season)", d), "hg_share")["b"]
        w = row(fit(f"{y} ~ hg_share + age_w + C(season) + C(team)", d), "hg_share")["b"]
        out[y] = {"between_rerun": b, "between_main": base["main"][y]["between"]["b"],
                  "within_rerun": w, "within_main": base["main"][y]["within"]["b"]}
        assert abs(b - base["main"][y]["between"]["b"]) < 1e-9, "baseline mismatch"
        assert abs(w - base["main"][y]["within"]["b"]) < 1e-9, "baseline mismatch"
    R["baseline_reproduced"] = out
    log.info("baseline reproduced exactly")


def inference(d):
    """Small-cluster inference: wild bootstrap, season clustering, two-way, permutation."""
    rows = []
    specs = {"between": "C(season)", "within": "C(season) + C(team)"}
    for y in ["win_pct", "pd_pg", "net_epa", "off_epa", "def_epa"]:
        for k, fe in specs.items():
            f = f"{y} ~ hg_share + age_w + {fe}"
            m = fit(f, d)
            r = row(m, "hg_share")
            r.update(wild_p(f, d, "hg_share"))
            ms = fit(f, d, cluster="season")
            r["p_cl_season"] = float(ms.pvalues["hg_share"])
            # two-way (team, season) clustering via statsmodels groups array
            mm = smf.ols(f, data=d).fit()
            grp = np.column_stack([pd.factorize(d.loc[mm.model.data.row_labels, "team"])[0],
                                   pd.factorize(d.loc[mm.model.data.row_labels, "season"])[0]])
            m2 = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": grp})
            r["p_twoway"] = float(m2.pvalues["hg_share"])
            rows.append({"outcome": y, "spec": k, **r})
    t = pd.DataFrame(rows)
    # Permutation inference: shuffle hg_share within season (between spec)
    for y in ["win_pct", "net_epa"]:
        b0 = row(fit(f"{y} ~ hg_share + age_w + C(season)", d), "hg_share")["b"]
        bs = []
        for _ in range(B):
            dd = d.copy()
            dd["hg_share"] = dd.groupby("season").hg_share.transform(
                lambda s: RNG.permutation(s.values))
            bs.append(smf.ols(f"{y} ~ hg_share + age_w + C(season)", dd).fit().params["hg_share"])
        t.loc[(t.outcome == y) & (t.spec == "between"), "p_perm"] = float(np.mean(np.abs(bs) >= abs(b0)))
    # Multiple-testing adjustment across the family of 10 headline tests
    t["p_holm"] = multipletests(t.p_wild, method="holm")[1]
    t["p_bh"] = multipletests(t.p_wild, method="fdr_bh")[1]
    t.round(4).to_csv(OUT / "r1_inference.csv", index=False)
    R["inference"] = t.round(4).to_dict("records")


def jackknife(d):
    out = {}
    for y in ["win_pct", "net_epa"]:
        for k, fe in {"between": "C(season)", "within": "C(season) + C(team)"}.items():
            f = f"{y} ~ hg_share + age_w + {fe}"
            lot = [row(fit(f, d[d.team != t]), "hg_share") for t in d.team.unique()]
            los = [row(fit(f, d[d.season != s]), "hg_share") for s in d.season.unique()]
            out[f"{y}_{k}"] = {
                "drop_team_b_min": min(r["b"] for r in lot), "drop_team_b_max": max(r["b"] for r in lot),
                "drop_team_p_max": max(r["p"] for r in lot),
                "drop_team_most_influential": d.team.unique()[int(np.argmin([r["b"] for r in lot]))],
                "drop_season_b_min": min(r["b"] for r in los), "drop_season_b_max": max(r["b"] for r in los),
                "drop_season_p_max": max(r["p"] for r in los)}
    pd.DataFrame(out).T.to_csv(OUT / "r2_jackknife.csv")
    R["jackknife"] = out


def confounders(d):
    """Add QB, coach, churn, lagged performance, and rookie-contract controls."""
    rows = []
    base = "hg_share + age_w + C(season)"
    specs = {
        "baseline": base,
        "+ lagged win% & net EPA": base + " + win_pct_lag + net_epa_lag",
        "+ roster churn (n players)": base + " + n_players",
        "+ homegrown QB & QB age": base + " + hg_QB + qb_age",
        "+ QB fixed effects": base + " + C(qb_name)",
        "+ head-coach fixed effects": base + " + C(coach)",
        "+ coach tenure & change": base + " + coach_tenure + coach_change",
        "+ team FE + coach FE": base + " + C(team) + C(coach)",
        "all controls (no FE)": base + " + win_pct_lag + net_epa_lag + n_players + hg_QB + qb_age + coach_tenure + coach_change",
    }
    for y in ["win_pct", "net_epa"]:
        for name, rhs in specs.items():
            dd = d.dropna(subset=["win_pct_lag", "net_epa_lag"]) if "lag" in rhs or name.startswith("all") else d
            m = fit(f"{y} ~ {rhs}", dd)
            rows.append({"outcome": y, "spec": name, **row(m, "hg_share"), "r2": m.rsquared})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(OUT / "r3_confounders.csv", index=False)
    R["confounders"] = t.round(4).to_dict("records")


def rookie_contracts(d):
    """Is the homegrown premium just cheap rookie deals (Mulholland & Jensen 2019)?"""
    out = {}
    for y in ["win_pct", "net_epa"]:
        m = fit(f"{y} ~ sh_hg_rook + sh_hg_vet + age_w + C(season)", d)
        out[y] = {"hg_first_contract": row(m, "sh_hg_rook"), "hg_second_contract": row(m, "sh_hg_vet")}
        w = m.t_test("sh_hg_vet - sh_hg_rook = 0")
        out[y]["diff_p"] = float(w.pvalue)
        m2 = fit(f"{y} ~ hg_share + sh_rook_all + age_w + C(season)", d)
        out[y]["hg_share_ctrl_all_rookie_snaps"] = row(m2, "hg_share")
        out[y]["all_rookie_snaps"] = row(m2, "sh_rook_all")
        m3 = fit(f"{y} ~ sh_hg_vet + sh_hg_rook + age_w + C(season) + C(team)", d)
        out[y]["within_hg_second_contract"] = row(m3, "sh_hg_vet")
    out["means"] = {c: float(d[c].mean()) for c in ["sh_hg_rook", "sh_hg_vet", "sh_acq_rook", "sh_rook_all"]}
    R["rookie_contracts"] = out


def measures(d):
    """Alternative measures, subsamples, and functional form."""
    rows = []
    alts = {"homegrown share (baseline)": "hg_share", "drafted only": "sh_drafted",
            "UDFA only": "sh_udfa", "continuity": "continuity", "tenure (yrs)": "tenure_w",
            "homegrown excl. QB": "hg_share_noqb", "homegrown offense": "hg_off",
            "homegrown defense": "hg_def"}
    for y in ["win_pct", "net_epa"]:
        for lab, x in alts.items():
            m = fit(f"{y} ~ {x} + age_w + C(season)", d)
            r = row(m, x)
            r["b_per_sd"] = r["b"] * d[x].std()
            rows.append({"outcome": y, "measure": lab, **r})
        # horse race: drafted vs continuity together
        m = fit(f"{y} ~ hg_share + continuity + age_w + C(season)", d)
        rows.append({"outcome": y, "measure": "horse race: homegrown", **row(m, "hg_share")})
        rows.append({"outcome": y, "measure": "horse race: continuity", **row(m, "continuity")})
        for lab, sub in {"2013-2019": d.season <= 2019, "2020-2025 (17-game era from 2021)": d.season >= 2020,
                         "cap coverage >= 85%": d.cap_cov >= .85}.items():
            m = fit(f"{y} ~ hg_share + age_w + C(season)", d[sub])
            rows.append({"outcome": y, "measure": f"subsample {lab}", **row(m, "hg_share")})
        m = fit(f"{y} ~ cap_acq_share + age_w + C(season)", d[d.cap_cov >= .85])
        rows.append({"outcome": y, "measure": "acquired-cap share, cap coverage >= 85%", **row(m, "cap_acq_share")})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(OUT / "r4_measures.csv", index=False)
    R["measures"] = t.round(4).to_dict("records")
    # Functional form: decile means and a U-shape test on continuity (Sieweke & Zhao 2015)
    ff = {}
    for x in ["hg_share", "continuity"]:
        dd = d.assign(xc=d[x] - d[x].mean())
        m = fit("net_epa ~ xc + I(xc**2) + age_w + C(season)", dd)
        ff[x] = {"lin": row(m, "xc"), "sq": row(m, "I(xc ** 2)")}
        dd["dec"] = pd.qcut(dd[x], 10, labels=False)
        ff[x]["decile_net_epa"] = dd.groupby("dec").net_epa.mean().round(4).tolist()
    R["functional_form"] = ff


def typology_grid(d):
    """Re-run the base x apex typology over a grid of cutoffs."""
    d = d.copy()
    cuts = [.40, .45, .50, .55, .60]
    rows, rams, merc = [], [], []
    mercs_focal = {"NYJ", "NYG", "LV", "CHI"}
    for qh in cuts:
        for qa in cuts:
            mh, ma = d.hg_share.quantile(qh), d.cap_acq_share.quantile(qa)
            t = np.select([(d.hg_share >= mh) & (d.cap_acq_share < ma),
                           (d.hg_share >= mh) & (d.cap_acq_share >= ma),
                           (d.hg_share < mh) & (d.cap_acq_share >= ma)],
                          ["Grown", "Stars-and-scrubs", "Mercenary"], "Churn")
            d["t"] = t
            g = d.groupby("t").agg(win=("win_pct", "mean"), pw=("playoff_wins", "mean"), n=("win_pct", "size"))
            la = d[(d.team == "LA") & d.season.between(2017, 2025)]
            top_merc = (d[d.t == "Mercenary"].team.value_counts() / d.team.value_counts()).dropna() \
                .sort_values(ascending=False).head(5).index.tolist()
            rows.append({"hg_q": qh, "acq_q": qa,
                         **{f"win_{k}": v for k, v in g.win.items()},
                         **{f"n_{k}": v for k, v in g.n.items()},
                         "merc_lowest_win": g.win.idxmin() == "Mercenary",
                         "rams_ss_share_17_25": float((la.t == "Stars-and-scrubs").mean()),
                         "rams_2021": la.loc[la.season == 2021, "t"].iloc[0],
                         "top5_merc_rate_teams": ",".join(top_merc),
                         "focal_mercs_in_top5": len(mercs_focal & set(top_merc))})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(OUT / "r5_typology_grid.csv", index=False)
    R["typology_grid"] = {
        "n_cells": len(t),
        "merc_lowest_win_cells": int(t.merc_lowest_win.sum()),
        "rams_ss_share_range": [float(t.rams_ss_share_17_25.min()), float(t.rams_ss_share_17_25.max())],
        "rams_2021_types": t.rams_2021.value_counts().to_dict(),
        "focal_mercs_in_top5_mean": float(t.focal_mercs_in_top5.mean()),
        "focal_mercs_in_top5_min": int(t.focal_mercs_in_top5.min())}
    # Champions band: share of champions in alternative percentile bands
    d["hg_pct"] = d.groupby("season").hg_share.rank(pct=True)
    ch = d[d.sb_win == 1].hg_pct
    R["champs_bands"] = {f"{lo}-{hi}": int(ch.between(lo, hi).sum())
                         for lo, hi in [(.5, .85), (.4, .9), (.5, 1.0), (0, .5), (.85, 1.0)]}
    R["champs_pct"] = sorted(ch.round(3).tolist())
    # Null for the band claim: chance 13 random team-seasons land 10+ in a 35-pt band
    from scipy.stats import binom
    R["champs_band_binom_p"] = float(binom.sf(9, 13, .35))


def oster(d):
    """Oster (2019) delta for hg_share on net EPA, Rmax = 1.3 * R-tilde."""
    out = {}
    for y in ["net_epa", "win_pct"]:
        dd = d.dropna(subset=["coach_tenure"])
        short = smf.ols(f"{y} ~ hg_share + C(season)", dd).fit()
        long = smf.ols(f"{y} ~ hg_share + age_w + hg_QB + qb_age + coach_tenure"
                       " + coach_change + C(season)", dd).fit()  # churn excluded: post-treatment
        b0, r0 = short.params["hg_share"], short.rsquared
        b1, r1 = long.params["hg_share"], long.rsquared
        rmax = min(1.0, 1.3 * r1)
        delta = (b1 * (r1 - r0)) / ((b0 - b1) * (rmax - r1)) if b0 != b1 else np.inf
        beta_star = b1 - (b0 - b1) * (rmax - r1) / (r1 - r0)
        out[y] = {"b_short": b0, "r2_short": r0, "b_long": b1, "r2_long": r1, "rmax": rmax,
                  "delta_for_beta0": float(delta), "beta_delta1": float(beta_star)}
    R["oster"] = out


def dynamics_fixed(d):
    """Next-season models with 2025 lead values set to missing (the following season, 2026, is incomplete)."""
    d = d.copy()
    for c in ["win_pct_lead", "net_epa_lead", "hg_share_lead", "pd_pg_lead"]:
        d.loc[d.season == 2025, c] = np.nan
    res = {}
    for x in ["hg_share", "continuity", "cap_acq_share", "age_w"]:
        for y in ["win_pct_lead", "net_epa_lead"]:
            ctrl = "win_pct + net_epa + C(season)" + ("" if x == "age_w" else " + age_w")
            res[f"{x}->{y}"] = row(fit(f"{y} ~ {x} + {ctrl}", d.dropna(subset=[y])), x)
    d2 = d.dropna(subset=["win_pct_lead"]).assign(dwin=lambda z: z.win_pct_lead - z.win_pct)
    res["picks_next_change"] = row(fit("dwin ~ picks_r1r2_for_vets + win_pct + C(season)", d2), "picks_r1r2_for_vets")
    d3 = d.dropna(subset=["net_epa_lead"]).copy()
    d3["age_t"] = pd.qcut(d3.age_w, 3, labels=["young", "mid", "old"])
    res["age_terciles_dnet"] = (d3.net_epa_lead - d3.net_epa).groupby(d3.age_t, observed=True).mean().to_dict()
    ty = pd.read_csv(OUT / "typology_assignments.csv")
    dd = d.merge(ty, on=["team", "season"])
    res["typology_win_next"] = dd.dropna(subset=["win_pct_lead"]).groupby("type").win_pct_lead.mean().round(3).to_dict()
    res["n"] = int(d.win_pct_lead.notna().sum())
    R["dynamics_fixed"] = res


def main():
    d = pd.read_parquet(CLEAN / "team_season.parquet")
    d = d[d.season.between(2013, 2025)].copy()
    baseline_check(d)
    d = add_coaches(d)
    d, chk = add_rookie_split(d)
    R["n"] = len(d)
    R["coach_missing"] = int(d.coach.isna().sum())
    R["rookie_split_check_max_abs_diff_vs_hg_share"] = float(
        (d.sh_hg_rook + d.sh_hg_vet - d.hg_share).abs().max())
    inference(d); jackknife(d); confounders(d); rookie_contracts(d); measures(d)
    typology_grid(d); oster(d); dynamics_fixed(d)
    with open(OUT / "robustness.json", "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log.info("wrote robustness.json")


if __name__ == "__main__":
    main()

"""Decile figure (fig8) and every manuscript table as Markdown, generated from result files.

Tables are written to output/tables/<name>.md, so no table number is typed by hand.
"""
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLEAN, FIG, OUT, ROOT

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
TAB = ROOT / "output" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
R1 = json.load(open(OUT / "results.json"))
R2 = json.load(open(OUT / "robustness.json"))


def stars(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def bse(r: dict, k: int = 3) -> str:
    return f"{r['b']:.{k}f} ({r['se']:.{k}f}){stars(r['p'])}"


def pv(p: float) -> str:
    return "<.001" if p < .001 else f"{p:.3f}"


def md(df: pd.DataFrame, name: str, note: str) -> None:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    (TAB / f"{name}.md").write_text("\n".join(lines) + f"\n\n*Note.* {note}\n")


def fig8() -> None:
    d = pd.read_parquet(CLEAN / "team_season.parquet")
    d = d[d.season.between(2013, 2025)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for x, lab, c in [("hg_share", "Homegrown share", "#1f5fa8"), ("continuity", "Continuity", "#d9822b")]:
        dec = pd.qcut(d[x], 10, labels=False)
        g = d.groupby(dec).net_epa.agg(["mean", "sem"])
        ax.errorbar(np.arange(1, 11) + (-.1 if x == "hg_share" else .1), g["mean"], yerr=1.96 * g["sem"],
                    marker="o", capsize=3, label=lab, color=c)
    ax.axhline(0, color="grey", lw=.8, ls="--")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decile of construction measure (team-seasons, 2013–2025)")
    ax.set_ylabel("Mean net EPA per play")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_deciles.png", dpi=300)


def tables() -> None:
    t = pd.read_csv(OUT / "t1_age_by_position.csv").fillna("")
    t.columns = ["Group", "Active-roster age", "Snap-weighted age", "SD across team-seasons"]
    md(t, "ages", "Team-season means, 2013–2025. Snap-weighted age uses offensive snaps for offensive groups and "
       "defensive snaps for defensive groups. Specialists: player with most special-teams snaps; returners: player "
       "with most returns.")

    lab = {"win_pct": "Win %", "pd_pg": "Point diff. per game", "net_epa": "Net EPA/play",
           "off_epa": "Offensive EPA/play", "def_epa": "Defensive EPA allowed/play", "st_epa_pg": "Special-teams EPA/game"}
    rows = [[lab[k], bse(R1["main"][k]["between"]), bse(R1["main"][k]["within"]), bse(R1["main"][k]["age_between"])]
            for k in lab]
    md(pd.DataFrame(rows, columns=["Outcome", "Homegrown share (season FE)", "Homegrown share (team + season FE)",
                                   "Snap-weighted age (season FE)"]),
       "main", "OLS, N = 416 team-seasons. Each row reports two regressions (Equation 5), each controlling for "
       "snap-weighted age. Standard errors clustered by team in parentheses. *** p<.01, ** p<.05, * p<.1. "
       "Defensive EPA is EPA allowed, so negative coefficients are good.")

    m = pd.DataFrame(R2["measures"])
    keep = ["homegrown share (baseline)", "drafted only", "UDFA only", "continuity", "tenure (yrs)", "homegrown excl. QB",
            "horse race: homegrown", "horse race: continuity"]
    names = {"homegrown share (baseline)": "Homegrown share (baseline)", "drafted only": "Drafted share only",
             "UDFA only": "Own-UDFA share only", "continuity": "Continuity", "tenure (yrs)": "Snap-weighted tenure (years)",
             "homegrown excl. QB": "Homegrown share excl. QBs", "horse race: homegrown": "Joint model: homegrown share",
             "horse race: continuity": "Joint model: continuity"}
    rows = []
    for k in keep:
        a = m[(m.outcome == "net_epa") & (m.measure == k)].iloc[0]
        b = m[(m.outcome == "win_pct") & (m.measure == k)].iloc[0]
        rows.append([names[k], bse(a), bse(b)])
    rc = R2["rookie_contracts"]
    for k, n in [("hg_first_contract", "Split model: homegrown, first contract"),
                 ("hg_second_contract", "Split model: homegrown, second contract or later")]:
        rows.append([n, bse(rc["net_epa"][k]), bse(rc["win_pct"][k])])
    rows.append(["Split model: test first = second (p)", pv(rc["net_epa"]["diff_p"]), pv(rc["win_pct"]["diff_p"])])
    md(pd.DataFrame(rows, columns=["Measure", "Net EPA/play", "Win %"]), "measures",
       "Each line is a regression with season fixed effects and snap-weighted age; joint and split models enter "
       "both measures together. First contract = experience of three or fewer seasons. Standard errors clustered "
       "by team. N = 416.")

    el, dy = R1["early_late"], {**R1["dynamics"], **R2["dynamics_fixed"]}
    rows = [["Weeks 1–4 homegrown share → weeks 5–18 win %", bse(el["win_l"]), "weeks 1–4 win %, margin"],
            ["Weeks 1–4 homegrown share → weeks 5–18 margin/game", bse(el["mar_l"], 2), "weeks 1–4 margin"],
            ["Homegrown share → next-season net EPA", bse(dy["hg_share->net_epa_lead"]), "this season win %, net EPA, age"],
            ["Homegrown share → next-season win %", bse(dy["hg_share->win_pct_lead"]), "same"],
            ["Acquired-cap share → next-season net EPA", bse(dy["cap_acq_share->net_epa_lead"]), "same"],
            ["Acquired-cap share → next-season win %", bse(dy["cap_acq_share->win_pct_lead"]), "same"],
            ["Snap-weighted age → next-season net EPA", bse(dy["age_w->net_epa_lead"]), "this season win %, net EPA"],
            ["R1–2 picks traded for veterans → win %", bse(dy["picks_now"]), "last season win % (N = 384)"],
            ["R1–2 picks traded for veterans → next-season change in win %", bse(dy["picks_next_change"]), "this season win % (N = 384)"]]
    md(pd.DataFrame(rows, columns=["Specification", "Coefficient (SE)", "Controls besides season FE"]), "dynamics",
       "Equations 6 and 7. Standard errors clustered by team. Next-season models use 2013–2024 seasons (N = 384); early-season models N = 416.")

    inf = pd.DataFrame(R2["inference"])
    inf["Outcome"] = inf.outcome.map(lab)
    inf["Design"] = inf.spec.map({"between": "Season FE", "within": "Team + season FE"})
    inf["β (SE)"] = [f"{b:.3f} ({s:.3f})" for b, s in zip(inf.b, inf.se)]
    for c, n in [("p", "p (team)"), ("p_wild", "p (wild bootstrap)"), ("p_twoway", "p (two-way)"),
                 ("p_holm", "p (Holm)"), ("p_bh", "q (BH)")]:
        inf[n] = inf[c].map(pv)
    inf["p (permutation)"] = inf.p_perm.map(lambda x: "—" if pd.isna(x) else pv(x))
    md(inf[["Outcome", "Design", "β (SE)", "p (team)", "p (wild bootstrap)", "p (two-way)", "p (permutation)",
            "p (Holm)", "q (BH)"]], "inference",
       "Coefficient on homegrown share, Equation 5. Wild cluster bootstrap: restricted (null imposed), Webb six-point "
       "weights, 1,999 draws, clusters = 32 teams. Two-way: clustered by team and season. Permutation: homegrown "
       "share shuffled within season, 1,999 draws, season-FE design only. Holm and Benjamini–Hochberg adjustments "
       "applied to the ten wild-bootstrap p-values.")

    c = pd.DataFrame(R2["confounders"])
    rows = []
    for s in c.spec.unique():
        a = c[(c.outcome == "net_epa") & (c.spec == s)].iloc[0]
        b = c[(c.outcome == "win_pct") & (c.spec == s)].iloc[0]
        names = {"baseline": "Baseline (Table 1)", "+ lagged win% & net EPA": "Adds lagged win % and net EPA",
                 "+ roster churn (n players)": "Adds players used (churn)",
                 "+ homegrown QB & QB age": "Adds homegrown QB and QB age", "+ QB fixed effects": "Adds QB fixed effects",
                 "+ head-coach fixed effects": "Adds head-coach fixed effects",
                 "+ coach tenure & change": "Adds coach tenure and coaching change",
                 "+ team FE + coach FE": "Adds team and head-coach fixed effects",
                 "all controls (no FE)": "All controls (no team or coach FE)"}
        rows.append([names[s], bse(a), bse(b), int(a.n)])
    md(pd.DataFrame(rows, columns=["Specification", "Net EPA/play", "Win %", "N"]), "confounders",
       "Coefficient on homegrown share. All models include season fixed effects and snap-weighted age. Coach = head "
       "coach with most regular-season games for the team that season. 'All controls' adds lagged win % and net "
       "EPA, players used, homegrown QB, QB age, coach tenure and coaching change. Standard errors clustered by team.")

    ty = pd.read_csv(OUT / "t8_typology.csv")
    ty["win_next"] = ty.type.map(R2["dynamics_fixed"]["typology_win_next"])
    ty.columns = ["Type", "N", "Win %", "Net EPA", "Playoff rate", "Playoff wins/season", "Titles", "Win % next season"]
    md(ty, "typology", "Team-seasons split at the medians of homegrown snap share (61.6%) and acquired-cap share (45.1%). Types are defined by the two shares, not by counts of stars. Next-season win % uses 2013–2024 seasons.")

    ch = pd.read_csv(OUT / "t7_champions.csv").sort_values("season")
    ch["hg_pct"] = (ch.hg_pct * 100).round().astype(int)
    ch.columns = ["Season", "Champion", "Homegrown share", "Percentile (season)", "Acquired-cap share",
                  "Acquired stars", "R1–2 picks for vets (3 yrs)", "Snap age"]
    md(ch, "champions", "Percentile of homegrown share within the champion's season.")

    rows = [["Season level: wins over market expectation", bse(el["woe_l"]), "416 team-seasons"],
            ["Game level: margin against closing spread", bse(el["ats_game"], 2), f"{el['n_games']:,} team-games"],
            ["Game level: cover (excl. pushes)", bse(el["cover_game"]), "5,424 team-games"],
            ["Game level: win minus vig-free moneyline probability", bse(el["woe_game"]), "5,564 team-games"]]
    md(pd.DataFrame(rows, columns=["Outcome", "Homegrown share (SE)", "Sample"]), "market",
       "Equation 10. Game-level construction is the homegrown share of a team's prior games that season, from each team's "
       "fourth game on, demeaned by season; standard errors clustered by team-season. Season-level row uses weeks 1–4 share "
       "against weeks 5–18 outcomes.")

    jk = pd.DataFrame(R2["jackknife"]).T
    jk = jk[["drop_team_b_min", "drop_team_b_max", "drop_team_p_max", "drop_season_b_min", "drop_season_b_max",
             "drop_season_p_max"]].astype(float).round(3).reset_index()
    jk.columns = ["Outcome / design", "β min (drop team)", "β max (drop team)", "max p (drop team)",
                  "β min (drop season)", "β max (drop season)", "max p (drop season)"]
    md(jk, "jackknife", "Conventional team-clustered p-values.")

    g = pd.read_csv(OUT / "r5_typology_grid.csv")
    g = g[["hg_q", "acq_q", "win_Grown", "win_Stars-and-scrubs", "win_Mercenary", "win_Churn", "rams_2021",
           "rams_ss_share_17_25", "focal_mercs_in_top5"]].round(3)
    g.columns = ["Homegrown cut (quantile)", "Acquired-cap cut (quantile)", "Win % Grown", "Win % S&S",
                 "Win % Mercenary", "Win % Churn", "Rams 2021 type", "Rams S&S share 2017–25", "NYJ/NYG/LV/CHI in top 5"]
    md(g, "grid", "Each row re-assigns all 416 team-seasons using the stated quantile cutoffs. Last column: how many "
       "of the four franchises have a top-five Mercenary rate.")

    p = pd.read_csv(OUT / "t5_positions.csv")
    p = p.assign(est=[f"{b:.3f} ({s:.3f})" for b, s in zip(p.b_net_per_sd, p.se_net_per_sd)])
    pw = p.pivot(index="group", columns="measure", values="est").reindex(
        ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "CB", "S"]).reset_index()
    pw.columns = ["Position group", "Continuity (per SD)", "Homegrown share (per SD)"]
    md(pw, "positions", "Net EPA per play per one-SD change in the group's measure; separate regressions with season "
       "FE and snap-weighted age; team-clustered SE.")
    log.info("tables written to %s", TAB)


if __name__ == "__main__":
    fig8()
    tables()
